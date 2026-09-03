# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_TORCH

if CHECK_TORCH.is_available:
    import torch

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
        # allow TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

if TYPE_CHECKING:
    from logging import Logger

    from transformers import PreTrainedModel  # type: ignore[import-untyped]

    from infinity_emb.args import EngineArgs


def check_if_bettertransformer_possible(engine_args: "EngineArgs") -> bool:
    """BetterTransformer was removed from transformers 5 (#41367) and optimum 2; torch's
    scaled dot product attention, the default since transformers 4.36, covers the same models.
    The `bettertransformer` flag stays accepted so existing deployments keep starting."""
    return False


def to_bettertransformer(model: "PreTrainedModel", engine_args: "EngineArgs", logger: "Logger"):
    if engine_args.bettertransformer:
        logger.debug("bettertransformer is a no-op: BetterTransformer no longer exists upstream.")
    return model
