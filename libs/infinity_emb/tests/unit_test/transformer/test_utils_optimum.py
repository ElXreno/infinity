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
    _RecordingModel.calls = []
    _RecordingModel.fail_on = ""
    return _RecordingModel


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
