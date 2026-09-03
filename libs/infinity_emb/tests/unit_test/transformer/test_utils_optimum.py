from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from infinity_emb.transformer import utils_optimum
from infinity_emb.transformer.utils_optimum import OnnxOutputs, load_onnx_model

PROVIDER = "CPUExecutionProvider"
MODEL = "fake-org/fake-model"


class _RecordingSession:
    calls: ClassVar[list[dict]] = []
    fail_on: ClassVar[str] = ""

    def __init__(self, model_path, config, execution_provider, provider_options=None):
        self.model_path = Path(model_path)
        self.config = config
        self.execution_provider = execution_provider
        self.provider_options = dict(provider_options or {})
        type(self).calls.append(
            {
                "model_path": self.model_path,
                "execution_provider": execution_provider,
                "provider_options": self.provider_options,
            }
        )
        if type(self).fail_on and self.model_path.name.endswith(type(self).fail_on):
            raise RuntimeError(f"cannot load {self.model_path.name}")


@pytest.fixture
def model_dir(tmp_path):
    model = tmp_path / "model"
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model.onnx").write_bytes(b"")
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"")
    return model


@pytest.fixture
def recording_session(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_optimum, "HUGGINGFACE_HUB_CACHE", (tmp_path / "hub").as_posix())
    monkeypatch.setattr(utils_optimum, "OnnxModel", _RecordingSession)
    monkeypatch.setattr(
        utils_optimum,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_type="bert")),
    )
    _RecordingSession.calls = []
    _RecordingSession.fail_on = ""
    return _RecordingSession


def _load(model_dir, **kwargs):
    defaults = {
        "execution_provider": PROVIDER,
        "file_name": "onnx/model.onnx",
        "optimize_model": False,
    }
    return load_onnx_model(model_dir.as_posix(), **{**defaults, **kwargs})


def test_onnx_file_in_subfolder_resolves_inside_the_model_dir(recording_session, model_dir):
    _load(model_dir, file_name="onnx/model_quantized.onnx")

    assert recording_session.calls[0]["model_path"] == model_dir / "onnx" / "model_quantized.onnx"


def test_absolute_onnx_file_is_kept(recording_session, model_dir):
    _load(model_dir, file_name=(model_dir / "onnx" / "model.onnx").as_posix())

    assert recording_session.calls[0]["model_path"] == model_dir / "onnx" / "model.onnx"


def test_provider_options_are_forwarded(recording_session, model_dir):
    _load(model_dir, provider_options={"num_of_threads": 4})

    assert recording_session.calls[0]["provider_options"] == {"num_of_threads": 4}


def test_empty_provider_options_stay_empty(recording_session, model_dir):
    _load(model_dir)

    assert recording_session.calls[0]["provider_options"] == {}


def test_provider_options_override_tensorrt_defaults(recording_session, model_dir):
    _load(
        model_dir,
        execution_provider="TensorrtExecutionProvider",
        optimize_model=True,
        provider_options={"trt_fp16_enable": False, "trt_max_workspace_size": 1 << 30},
    )

    call = recording_session.calls[0]
    assert call["provider_options"]["trt_fp16_enable"] is False
    assert call["provider_options"]["trt_max_workspace_size"] == 1 << 30
    assert call["provider_options"]["trt_cuda_graph_enable"] is True
    assert call["model_path"].name == "model.onnx"


def _optimized_path(tmp_path, model_dir):
    return (
        tmp_path
        / "hub"
        / "infinity_onnx"
        / PROVIDER
        / model_dir.as_posix().lstrip("/")
        / "model_optimized.onnx"
    )


def test_optimizes_once_and_reuses_the_cached_graph(recording_session, model_dir, monkeypatch):
    optimized = []

    def fake_optimize_graph(model_path, output_path, config, execution_provider):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"optimized")
        optimized.append(output_path)
        return output_path

    monkeypatch.setattr(utils_optimum, "optimize_graph", fake_optimize_graph)

    _load(model_dir, optimize_model=True)
    _load(model_dir, optimize_model=True)

    assert len(optimized) == 1
    assert optimized[0].name == "model_optimized.onnx"
    assert [c["model_path"].name for c in recording_session.calls] == [
        "model_optimized.onnx",
        "model_optimized.onnx",
    ]


def test_broken_cached_optimized_model_falls_back(
    recording_session, model_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(utils_optimum, "HUGGINGFACE_HUB_CACHE", (tmp_path / "hub").as_posix())
    optimized = tmp_path / "hub" / "infinity_onnx" / PROVIDER / model_dir.as_posix().lstrip("/")
    optimized.mkdir(parents=True)
    (optimized / "model_optimized.onnx").write_bytes(b"broken")
    recording_session.fail_on = "_optimized.onnx"

    model = _load(model_dir, optimize_model=True)

    assert model.model_path == model_dir / "onnx" / "model.onnx"
    assert [c["model_path"].name for c in recording_session.calls] == [
        "model_optimized.onnx",
        "model.onnx",
    ]


def test_failed_optimization_falls_back(recording_session, model_dir, monkeypatch):
    def broken_optimize_graph(*args, **kwargs):
        raise RuntimeError("fusion failed")

    monkeypatch.setattr(utils_optimum, "optimize_graph", broken_optimize_graph)

    model = _load(model_dir, optimize_model=True)

    assert model.model_path == model_dir / "onnx" / "model.onnx"


def test_onnx_outputs_attribute_access():
    outputs = OnnxOutputs(logits=[[1.0]])

    assert outputs.logits == [[1.0]]
    assert outputs["logits"] == [[1.0]]
    with pytest.raises(AttributeError):
        outputs.last_hidden_state


def test_onnx_outputs_token_embeddings_naming():
    assert OnnxOutputs(last_hidden_state="hf", pooler_output="p").token_embeddings() == "hf"
    assert OnnxOutputs(token_embeddings="st", sentence_embedding="s").token_embeddings() == "st"
    assert OnnxOutputs(hidden="first", other="second").token_embeddings() == "first"


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
