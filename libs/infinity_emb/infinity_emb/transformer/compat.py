# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from typing import Any, Optional


def replace_underlying_model(module: Any, model: Any) -> None:
    """sentence-transformers >= 5 stores the transformers model as `Transformer.model` and
    exposes `auto_model` only as a read-only alias; older releases have `auto_model` alone."""
    if hasattr(module, "model"):
        module.model = model
    else:
        module.auto_model = model


def pooled_features(output: Any) -> Any:
    """transformers >= 5 returns a BaseModelOutputWithPooling from `get_text_features`,
    `get_image_features` and `get_audio_features`; the projected embedding is `pooler_output`.
    Older releases returned the tensor itself."""
    return getattr(output, "pooler_output", output)


def text_max_length(config: Any, tokenizer: Any) -> Optional[int]:
    """Longest text the text tower accepts. transformers >= 5 dropped `config.max_length`, so
    the limit comes from the text config's positions and the tokenizer; the smaller one wins
    because RoBERTa style towers reserve two positions and only the tokenizer knows that."""
    candidates = []
    for source in (getattr(config, "text_config", None), config):
        for attribute in ("max_position_embeddings", "max_length"):
            value = getattr(source, attribute, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 1_000_000:
        candidates.append(model_max)
    return min(candidates) if candidates else None
