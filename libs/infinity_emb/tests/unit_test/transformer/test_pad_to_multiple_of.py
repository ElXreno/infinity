import numpy as np
import pytest
import torch

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.crossencoder.optimum import OptimumCrossEncoder
from infinity_emb.transformer.crossencoder.torch import CrossEncoderPatched
from infinity_emb.transformer.embedder.optimum import OptimumEmbedder
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
    pad_features_to_multiple_of,
)
from infinity_emb.transformer.padding import padding_bucket

SENTENCES = [
    "Short.",
    "A slightly longer sentence with a few more tokens in it.",
    (
        "The third sentence is the longest of them all and keeps going for a while "
        "so that the padded length lands on an odd token count."
    ),
]
QUERY_DOCS = [
    ("Where is Paris?", doc)
    for doc in [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish, it comes with a side of fries and a drink.",
    ]
]


def _run(model, inputs):
    features = model.encode_pre(inputs)
    return features["input_ids"].shape[1], model.encode_post(model.encode_core(features))


def _assert_padded_matches(model, base_kwargs, inputs, multiple, compare):
    model.engine_args = EngineArgs(**base_kwargs)
    plain_len, plain_out = _run(model, inputs)
    model.engine_args = EngineArgs(**base_kwargs, pad_to_multiple_of=multiple)
    padded_len, padded_out = _run(model, inputs)

    assert padded_len % multiple == 0
    assert plain_len <= padded_len < plain_len + multiple
    compare(plain_out, padded_out)


def _assert_embeddings_close(plain, padded):
    for a, b in zip(plain, padded):
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine > 0.99


def _sigmoid(logits):
    return 1 / (1 + np.exp(-np.array(logits)))


def _assert_scores_close(plain, padded):
    np.testing.assert_allclose(_sigmoid(padded), _sigmoid(plain), atol=0.02)
    assert list(np.argsort(padded)) == list(np.argsort(plain))
    assert np.argmax(padded) == 0


def test_engine_args_pad_to_multiple_of():
    assert EngineArgs().pad_to_multiple_of == 0
    assert EngineArgs(pad_to_multiple_of=16).pad_to_multiple_of == 16
    with pytest.raises(ValueError):
        EngineArgs(pad_to_multiple_of=-1)


def test_engine_args_onnx_provider_options():
    assert EngineArgs().onnx_provider_options_dict() == {}
    args = EngineArgs(onnx_provider_options='{"num_of_threads": 8, "load_config": "{}"}')
    assert args.onnx_provider_options_dict() == {"num_of_threads": 8, "load_config": "{}"}
    with pytest.raises(ValueError):
        EngineArgs(onnx_provider_options="not json")
    with pytest.raises(TypeError):
        EngineArgs(onnx_provider_options="[1, 2]")


class _Tokenizer:
    pad_token_id = 7

    def __init__(self, padding_side):
        self.padding_side = padding_side


def test_pad_features_to_multiple_of_helper():
    features = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 7]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }

    right = pad_features_to_multiple_of(features, 4, _Tokenizer("right"))
    assert right["input_ids"].tolist() == [[1, 2, 3, 7], [4, 5, 7, 7]]
    assert right["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]

    left = pad_features_to_multiple_of(features, 4, _Tokenizer("left"))
    assert left["input_ids"].tolist() == [[7, 1, 2, 3], [7, 4, 5, 7]]
    assert left["attention_mask"].tolist() == [[0, 1, 1, 1], [0, 1, 1, 0]]

    assert pad_features_to_multiple_of(features, 3, _Tokenizer("right")) is features
    assert pad_features_to_multiple_of(features, 4, _Tokenizer("right"), max_length=3) is features
    capped = pad_features_to_multiple_of(features, 8, _Tokenizer("right"), max_length=4)
    assert capped["input_ids"].shape[1] == 4


@pytest.mark.parametrize(
    "max_length,multiple,expected",
    [
        (514, 32, (32, 512)),
        (8194, 64, (64, 8192)),
        (512, 64, (64, 512)),
        (32, 64, (32, 32)),
        (514, 0, (None, 514)),
    ],
)
def test_padding_bucket(max_length, multiple, expected):
    assert padding_bucket(max_length, multiple) == expected


@pytest.mark.parametrize("multiple", [16, 64])
def test_optimum_embedder(multiple):
    base = {"model_name_or_path": "Xenova/bge-small-en-v1.5", "device": "cpu"}
    model = OptimumEmbedder(engine_args=EngineArgs(**base))
    _assert_padded_matches(model, base, SENTENCES, multiple, _assert_embeddings_close)


@pytest.mark.parametrize("multiple", [32, 64])
def test_optimum_crossencoder(multiple):
    base = {"model_name_or_path": "Xenova/bge-reranker-base", "device": "cpu"}
    model = OptimumCrossEncoder(engine_args=EngineArgs(**base))
    _assert_padded_matches(model, base, QUERY_DOCS, multiple, _assert_scores_close)


@pytest.mark.parametrize("multiple", [16, 64])
def test_torch_embedder(multiple):
    base = {
        "model_name_or_path": "michaelfeil/bge-small-en-v1.5",
        "engine": "torch",
        "device": "cpu",
    }
    model = SentenceTransformerPatched(engine_args=EngineArgs(**base))
    _assert_padded_matches(model, base, SENTENCES, multiple, _assert_embeddings_close)


@pytest.mark.parametrize("multiple", [32, 64])
def test_torch_crossencoder(multiple):
    base = {
        "model_name_or_path": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "engine": "torch",
        "device": "cpu",
    }
    model = CrossEncoderPatched(engine_args=EngineArgs(**base))
    _assert_padded_matches(model, base, QUERY_DOCS, multiple, _assert_scores_close)
