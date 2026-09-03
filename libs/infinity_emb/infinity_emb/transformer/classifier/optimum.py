# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import copy
from typing import Any

import numpy as np

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.transformer.abstract import BaseClassifer
from infinity_emb.transformer.classifier import classification_activation
from infinity_emb.transformer.padding import padding_bucket
from infinity_emb.transformer.utils_optimum import (
    device_to_onnx,
    get_onnx_files,
    load_onnx_model,
)

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]


class OptimumClassifier(BaseClassifer):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_ONNXRUNTIME.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        provider = device_to_onnx(engine_args.device)

        onnx_file = get_onnx_files(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            use_auth_token=True,
            prefer_quantized=("cpu" in provider.lower() or "openvino" in provider.lower())
            and not engine_args.onnx_do_not_prefer_quantized,
        )

        self.model = load_onnx_model(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not engine_args.onnx_disable_optimize,
            provider_options=engine_args.onnx_provider_options_dict(),
        )
        self.config = self.model.config
        self.classification_activation = classification_activation(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.engine_args = engine_args

    def encode_pre(self, sentences: list[str]) -> dict[str, np.ndarray]:
        pad_to_multiple_of, max_length = padding_bucket(
            self.config.max_position_embeddings, self.engine_args.pad_to_multiple_of
        )
        encoded = self.tokenizer(
            sentences,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=True,
            return_tensors="np",
        )
        return {k: v.astype(np.int64) for k, v in encoded.items()}

    def encode_core(self, features: dict[str, np.ndarray]) -> np.ndarray:
        return self.model(**features)["logits"]

    def encode_post(self, logits: np.ndarray) -> list[Any]:
        """one list per sentence, every label with its raw logit, best first"""
        id2label = self.config.id2label
        results = []
        for row in np.asarray(logits, dtype=np.float32):
            order = np.argsort(-row)
            results.append([{"label": id2label[int(i)], "score": float(row[i])} for i in order])
        return results

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
        tks = self._infinity_tokenizer(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
        ).encodings
        return [len(t.tokens) for t in tks]
