from pathlib import Path
from typing import ClassVar

import pytest

from infinity_emb.transformer import utils_optimum
from infinity_emb.transformer.utils_optimum import optimize_model

PROVIDER = "CPUExecutionProvider"
MODEL = "fake-org/fake-model"


class _RecordingModel:
    calls: ClassVar[list[dict]] = []
    fail_on: ClassVar[str] = ""

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        if cls.fail_on and kwargs["file_name"].endswith(cls.fail_on):
            raise RuntimeError(f"cannot load {kwargs['file_name']}")
        return (model_name_or_path, kwargs["file_name"])


@pytest.fixture
def recording_model(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_optimum, "HUGGINGFACE_HUB_CACHE", tmp_path.as_posix())
    monkeypatch.setattr(utils_optimum, "symlink_free_model_dir", lambda repo, *_, **__: repo)
    _RecordingModel.calls = []
    _RecordingModel.fail_on = ""
    return _RecordingModel


def test_symlink_free_model_dir_hardlinks_the_snapshot(monkeypatch, tmp_path):
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc123"
    (blobs).mkdir()
    (snapshot / "onnx").mkdir(parents=True)
    (blobs / "b1").write_bytes(b"model")
    (blobs / "b2").write_bytes(b"weights")
    (blobs / "b3").write_text("{}")
    (snapshot / "onnx" / "model.onnx").symlink_to("../../../blobs/b1")
    (snapshot / "onnx" / "model.onnx_data").symlink_to("../../../blobs/b2")
    (snapshot / "config.json").symlink_to("../../blobs/b3")
    monkeypatch.setattr(utils_optimum, "HUGGINGFACE_HUB_CACHE", (tmp_path / "hub").as_posix())
    monkeypatch.setattr(utils_optimum, "snapshot_download", lambda *a, **k: snapshot.as_posix())

    target = Path(utils_optimum.symlink_free_model_dir(MODEL, "onnx/model.onnx", None))

    assert target == tmp_path / "hub" / "infinity_onnx" / "materialized" / MODEL / "abc123"
    for rel, content in [
        ("onnx/model.onnx", b"model"),
        ("onnx/model.onnx_data", b"weights"),
        ("config.json", b"{}"),
    ]:
        assert not (target / rel).is_symlink()
        assert (target / rel).read_bytes() == content
    assert (target / "onnx" / "model.onnx_data").stat().st_ino == (blobs / "b2").stat().st_ino


def test_symlink_free_model_dir_keeps_local_and_plain_paths(monkeypatch, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    assert (
        utils_optimum.symlink_free_model_dir(local.as_posix(), "model.onnx", None)
        == local.as_posix()
    )

    plain = tmp_path / "plain-snapshot"
    plain.mkdir()
    (plain / "model.onnx").write_bytes(b"x")
    monkeypatch.setattr(utils_optimum, "snapshot_download", lambda *a, **k: plain.as_posix())
    assert utils_optimum.symlink_free_model_dir(MODEL, "model.onnx", None) == plain.as_posix()


def _cache_optimized_file(tmp_path):
    folder = tmp_path / "infinity_onnx" / PROVIDER / MODEL
    folder.mkdir(parents=True)
    optimized = folder / "model_optimized.onnx"
    optimized.write_bytes(b"")
    return optimized


def test_cached_optimized_model_is_used(recording_model, tmp_path):
    optimized = _cache_optimized_file(tmp_path)

    model = optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider=PROVIDER,
        file_name="model.onnx",
        optimize_model=True,
    )

    assert model == (optimized.parent.as_posix(), "model_optimized.onnx")
    assert len(recording_model.calls) == 1


def test_broken_cached_optimized_model_falls_back(recording_model, tmp_path):
    _cache_optimized_file(tmp_path)
    recording_model.fail_on = "_optimized.onnx"

    model = optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider=PROVIDER,
        file_name="model.onnx",
        optimize_model=True,
    )

    assert model == (MODEL, "model.onnx")
    assert [c["file_name"] for c in recording_model.calls] == [
        "model_optimized.onnx",
        "model.onnx",
    ]


def test_onnx_file_in_subfolder_is_split(recording_model):
    optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider=PROVIDER,
        file_name="onnx/model_quantized.onnx",
        optimize_model=False,
    )

    call = recording_model.calls[0]
    assert call["file_name"] == "model_quantized.onnx"
    assert call["subfolder"] == "onnx"


def test_provider_options_are_forwarded(recording_model):
    optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider=PROVIDER,
        file_name="model.onnx",
        optimize_model=False,
        provider_options={"num_of_threads": 4},
    )

    assert recording_model.calls[0]["provider_options"] == {"num_of_threads": 4}


def test_empty_provider_options_pass_none(recording_model):
    optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider=PROVIDER,
        file_name="model.onnx",
        optimize_model=False,
    )

    assert recording_model.calls[0]["provider_options"] is None


def test_provider_options_override_tensorrt_defaults(recording_model):
    optimize_model(
        MODEL,
        model_class=recording_model,
        execution_provider="TensorrtExecutionProvider",
        file_name="model.onnx",
        provider_options={"trt_fp16_enable": False, "trt_max_workspace_size": 1 << 30},
    )

    options = recording_model.calls[0]["provider_options"]
    assert options["trt_fp16_enable"] is False
    assert options["trt_max_workspace_size"] == 1 << 30
    assert options["trt_cuda_graph_enable"] is True
