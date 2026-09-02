# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from typing import Any


def replace_underlying_model(module: Any, model: Any) -> None:
    """sentence-transformers >= 5 stores the transformers model as `Transformer.model` and
    exposes `auto_model` only as a read-only alias; older releases have `auto_model` alone."""
    if hasattr(module, "model"):
        module.model = model
    else:
        module.auto_model = model
