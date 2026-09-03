# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from huggingface_hub import HfApi, get_token, snapshot_download  # type: ignore
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # type: ignore

from infinity_emb._optional_imports import CHECK_ONNX, CHECK_ONNXRUNTIME, CHECK_TRANSFORMERS
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device

if CHECK_ONNXRUNTIME.is_available:
    try:
        import onnxruntime as ort  # type: ignore
    except (ImportError, RuntimeError, Exception) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoConfig  # type: ignore[import-untyped]

OPTIMIZED_SUFFIX = "_optimized.onnx"

TENSORRT_PROVIDER_OPTIONS = {
    "trt_fp16_enable": True,
    "trt_layer_norm_fp32_fallback": True,
    "trt_cuda_graph_enable": True,  # helps small layers
    "trt_builder_optimization_level": 3,  # select between 3-5
    # int8, not working, needs calibration table.
    # "trt_int8_use_native_calibration_table": True,
}

# onnxruntime.transformers fusion recipes; every encoder (bert, roberta, xlm-roberta, ...) is "bert"
ORT_MODEL_TYPES = {
    "bart": "bart",
    "clip": "clip",
    "gpt2": "gpt2",
    "gpt_neox": "gpt_neox",
    "swin": "swin",
    "t5": "t5",
    "vit": "vit",
}


def mean_pooling(last_hidden_states: np.ndarray, attention_mask: np.ndarray):
    input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

    sum_embeddings = np.sum(last_hidden_states.astype(float) * input_mask_expanded, axis=1)
    mask_sum = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)

    return sum_embeddings / mask_sum


def cls_token_pooling(model_output, *args):
    return model_output[:, 0]


def normalize(input_array, p=2, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def device_to_onnx(device: Device) -> str:
    CHECK_ONNXRUNTIME.mark_required()
    available = ort.get_available_providers()

    if device == Device.cpu:
        if "OpenVINOExecutionProvider" in available:
            return "OpenVINOExecutionProvider"
        return "CPUExecutionProvider"
    elif device == Device.cuda:
        if "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
        elif "MIGraphXExecutionProvider" in available:
            return "MIGraphXExecutionProvider"
        return "CUDAExecutionProvider"
    elif device == Device.mps:
        return "CoreMLExecutionProvider"
    elif device == Device.tensorrt:
        return "TensorrtExecutionProvider"
    elif device is None or device == Device.auto:
        if "TensorrtExecutionProvider" in available:
            return "TensorrtExecutionProvider"
        elif "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        elif "MIGraphXExecutionProvider" in available:
            return "MIGraphXExecutionProvider"  # swapped order of ROCM and MIGraphX
        elif "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
        elif "CoreMLExecutionProvider" in available:
            return "CoreMLExecutionProvider"
        elif "OpenVINOExecutionProvider" in available:
            return "OpenVINOExecutionProvider"
        else:
            return "CPUExecutionProvider"
    else:
        raise ValueError(f"Unknown device {device}")


class OnnxOutputs(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class OnnxModel:
    """One onnxruntime session plus the transformers config of the model, callable with the
    tokenizer output. This is the part of optimum's ORTModel that infinity used."""

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Any,
        execution_provider: str,
        provider_options: Optional[dict] = None,
    ):
        CHECK_ONNXRUNTIME.mark_required()
        self.model_path = Path(model_path)
        self.config = config
        self.execution_provider = execution_provider
        self.provider_options = dict(provider_options or {})
        providers: list[Any] = [execution_provider]
        if self.provider_options:
            providers = [(execution_provider, self.provider_options)]
        self.session = ort.InferenceSession(self.model_path.as_posix(), providers=providers)
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]

    def __call__(self, **inputs: np.ndarray) -> OnnxOutputs:
        feed = {name: inputs[name] for name in self.input_names if name in inputs}
        if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        missing = [name for name in self.input_names if name not in feed]
        if missing:
            raise ValueError(
                f"{self.model_path.name} expects inputs {missing} the tokenizer did not produce"
            )
        return OnnxOutputs(zip(self.output_names, self.session.run(None, feed)))


def symlink_free_model_dir(
    model_name_or_path: str, file_name: str, revision: Optional[str] = None
) -> str:
    """onnxruntime >= 1.23 rejects external data whose canonical path leaves the model
    directory (tensorprotoutils.cc, ValidateExternalDataPath). The huggingface cache stores
    every file as a symlink into blobs/, so a `*.onnx_data` next to the model always "escapes".
    Hardlink the snapshot into a plain directory instead; copy when hardlinks are impossible.
    """
    if Path(model_name_or_path).exists():
        return model_name_or_path
    onnx_file = Path(file_name).as_posix()
    snapshot = Path(
        snapshot_download(
            model_name_or_path,
            revision=revision,
            token=get_token(),
            allow_patterns=["*.json", onnx_file, f"{onnx_file}*"],
        )
    )
    if not (snapshot / onnx_file).is_symlink():
        return snapshot.as_posix()
    target = (
        Path(HUGGINGFACE_HUB_CACHE)
        / "infinity_onnx"
        / "materialized"
        / model_name_or_path
        / snapshot.name
    )
    for src in snapshot.rglob("*"):
        if src.is_dir():
            continue
        dst = target / src.relative_to(snapshot)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        real = src.resolve()
        try:
            os.link(real, dst)
        except OSError:
            shutil.copy2(real, dst)
    logger.info(f"loading {model_name_or_path} from the symlink-free copy {target}")
    return target.as_posix()


def _resolve_model_file(model_dir: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    if file_path.is_absolute() and file_path.exists():
        return file_path
    try:
        file_path = file_path.relative_to(model_dir)
    except ValueError:
        pass
    return model_dir / file_path


def optimize_graph(
    model_path: Path, output_path: Path, config: Any, execution_provider: str
) -> Path:
    """Offline graph fusion with onnxruntime's own transformers optimizer, same recipe optimum's
    ORTOptimizer used (opt_level 99, transformers specific fusions, fp16 on GPU providers)."""
    CHECK_ONNX.mark_required()
    from onnxruntime.transformers.fusion_options import FusionOptions  # type: ignore
    from onnxruntime.transformers.optimizer import (  # type: ignore
        optimize_model as ort_optimize_model,
    )

    is_gpu = not ("cpu" in execution_provider.lower() or "openvino" in execution_provider.lower())
    model_type = ORT_MODEL_TYPES.get(getattr(config, "model_type", ""), "bert")
    optimizer = ort_optimize_model(
        model_path.as_posix(),
        model_type=model_type,
        num_heads=0,
        hidden_size=0,
        optimization_options=FusionOptions(model_type),
        opt_level=99,
        use_gpu=is_gpu,
    )
    if is_gpu:
        optimizer.convert_float_to_float16(keep_io_types=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer.save_model_to_file(output_path.as_posix(), use_external_data_format=True)
    return output_path


def load_onnx_model(
    model_name_or_path: Union[str, Path],
    execution_provider: str,
    file_name: str,
    optimize_model: bool = False,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    provider_options: Optional[dict] = None,
) -> OnnxModel:
    """
    Downloads, optionally optimizes and loads an ONNX model for the execution provider.

    Args:
        model_name_or_path (Union[str, Path]): The model name or path
        execution_provider (str): The execution provider to use, e.g. "CUDAExecutionProvider"
        file_name (str): The onnx file name to use, e.g. "onnx/model.onnx"
        optimize_model (bool, optional): Whether to optimize the graph offline. Defaults to False.
        revision (Optional[str], optional): The revision to use. Defaults to None.
        trust_remote_code (bool, optional): Whether to trust the remote code. Defaults to True.
        provider_options (Optional[dict], optional): Options forwarded to the execution
            provider session. Merged over the TensorRT defaults. Defaults to None.
    """
    CHECK_ONNXRUNTIME.mark_required()
    CHECK_TRANSFORMERS.mark_required()
    provider_options = dict(provider_options or {})
    if execution_provider == "TensorrtExecutionProvider":
        provider_options = {**TENSORRT_PROVIDER_OPTIONS, **provider_options}
        optimize_model = False

    repo_id = str(model_name_or_path)
    model_dir = Path(symlink_free_model_dir(repo_id, file_name, revision))
    model_path = _resolve_model_file(model_dir, file_name)
    config = AutoConfig.from_pretrained(model_dir.as_posix(), trust_remote_code=trust_remote_code)

    if optimize_model:
        optimized = (
            Path(HUGGINGFACE_HUB_CACHE)
            / "infinity_onnx"
            / execution_provider
            / repo_id.lstrip("/")
            / model_path.name.replace(".onnx", OPTIMIZED_SUFFIX)
        )
        try:
            if optimized.exists():
                logger.info(f"Optimized model found at {optimized}, skipping optimization")
            else:
                logger.info(f"Optimizing {model_path} for {execution_provider}")
                optimize_graph(model_path, optimized, config, execution_provider)
            return OnnxModel(optimized, config, execution_provider, provider_options)
        except Exception as e:
            logger.warning(
                f"The optimized model {optimized} could not be used: {e}. "
                "Going to use the unoptimized model."
            )

    return OnnxModel(model_path, config, execution_provider, provider_options)


def _list_all_repo_files(
    model_name_or_path: str,
    revision: Union[str, None] = None,
    use_auth_token: Union[bool, str] = True,
):
    if not Path(model_name_or_path).exists():
        if isinstance(use_auth_token, bool):
            token = get_token()
        else:
            token = use_auth_token
        return list(
            map(
                Path,
                HfApi().list_repo_files(model_name_or_path, revision=revision, token=token),
            )
        )
    else:
        return list(Path(model_name_or_path).glob("**/*"))


def get_onnx_files(
    *,
    model_name_or_path: str,
    revision: Union[str, None] = None,
    use_auth_token: Union[bool, str] = True,
    prefer_quantized=False,
) -> Path:
    """gets the onnx files from the repo"""
    repo_files = _list_all_repo_files(
        model_name_or_path=model_name_or_path,
        revision=revision,
        use_auth_token=use_auth_token,
    )
    pattern = "**.onnx"
    onnx_files = [p for p in repo_files if p.match(pattern)]

    prefered_regex = "quantize" if prefer_quantized else "model.onnx"
    prefered_onnx = [f for f in onnx_files if prefered_regex in f.name]
    if len(onnx_files) > 1:
        logger.info(f"Found {len(onnx_files)} onnx files: {onnx_files}")
        if prefered_onnx:
            onnx_files = prefered_onnx
        onnx_file = onnx_files[-1]
        logger.info(f"Using {onnx_file} as the model")
        return onnx_file
    elif len(onnx_files) == 1:
        return onnx_files[0]
    else:
        raise ValueError(f"No onnx files found for {model_name_or_path} and revision {revision}")
