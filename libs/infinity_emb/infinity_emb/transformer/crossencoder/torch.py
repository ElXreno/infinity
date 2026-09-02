# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from infinity_emb._optional_imports import CHECK_SENTENCE_TRANSFORMERS, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, RerankLimits
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.crossencoder import truncate_texts_to_tokens
from infinity_emb.transformer.quantization.interface import (
    quant_interface,
)
from infinity_emb.transformer.padding import padding_bucket
from infinity_emb.transformer.st_compat import replace_underlying_model

if CHECK_TORCH.is_available and CHECK_SENTENCE_TRANSFORMERS.is_available:
    import torch
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
else:

    class CrossEncoder:  # type: ignore[no-redef]
        pass


if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel  # type: ignore[import-untyped]


from infinity_emb.transformer.acceleration import (
    to_bettertransformer,
    check_if_bettertransformer_possible,
)

__all__ = [
    "CrossEncoderPatched",
]


class CrossEncoderPatched(CrossEncoder, BaseCrossEncoder):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_SENTENCE_TRANSFORMERS.mark_required()

        model_kwargs = {}
        attempt_bt = check_if_bettertransformer_possible(engine_args)
        if engine_args.bettertransformer and attempt_bt:
            model_kwargs["attn_implementation"] = "eager"

        ls = engine_args._loading_strategy
        assert ls is not None

        if ls.loading_dtype is not None:  # type: ignore
            model_kwargs["torch_dtype"] = ls.loading_dtype

        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=ls.device_placement,
            model_kwargs=model_kwargs,
        )
        model = self._require_model()
        model.to(ls.device_placement)  # type: ignore[arg-type]

        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.engine_args = engine_args
        model.eval()
        if engine_args.bettertransformer and attempt_bt:
            self._replace_model(to_bettertransformer(model, engine_args, logger))

        self._require_model().to(ls.loading_dtype)  # type: ignore[arg-type]

        if ls.quantization_dtype is not None:
            model = self._require_model()
            self._replace_model(
                quant_interface(  # TODO: add ls.quantization_dtype and ls.placement
                    model, engine_args.dtype, device=Device[model.device.type]
                )
            )

        if engine_args.compile:
            logger.info("using torch.compile(dynamic=True)")
            self._replace_model(torch.compile(self._require_model(), dynamic=True))

    def _require_model(self) -> "PreTrainedModel":
        model = self.model
        assert model is not None
        return model

    def _replace_model(self, model: Any) -> None:
        replace_underlying_model(self._first_module(), model)

    def encode_pre(self, input_tuples: list[tuple[str, str, RerankLimits]]):
        # return input_tuples
        queries = [t[0].strip() for t in input_tuples]
        documents = [t[1].strip() for t in input_tuples]
        limits = [t[2] if len(t) > 2 else RerankLimits() for t in input_tuples]

        pad_to_multiple_of, model_max = padding_bucket(
            getattr(self._require_model().config, "max_position_embeddings", None)
            or self.tokenizer.model_max_length,
            self.engine_args.pad_to_multiple_of,
        )

        def pair_max_length(limit: RerankLimits) -> int:
            # Always cap at the model's positional limit: max_pair_tokens may exceed it
            # (e.g. a 768-token request against a 512-position model), which would produce
            # a sequence the model cannot process in encode_core.
            if limit.max_pair_tokens:
                return min(limit.max_pair_tokens, model_max)
            return model_max

        # 1) head-truncate the query and the document independently, then
        # 2) cap the joined pair (longest side trimmed first) to max_pair_tokens.
        queries = truncate_texts_to_tokens(
            self.tokenizer, queries, [lim.max_query_tokens for lim in limits]
        )
        documents = truncate_texts_to_tokens(
            self.tokenizer, documents, [lim.max_tokens_per_doc for lim in limits]
        )
        encodings = [
            self.tokenizer(
                q,
                d,
                truncation="longest_first",
                max_length=pair_max_length(lim),
            )
            for q, d, lim in zip(queries, documents, limits)
        ]
        return self.tokenizer.pad(
            encodings,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def encode_core(self, features: dict[str, "Tensor"]):
        """
        Computes sentence embeddings
        """
        model = self._require_model()
        with torch.no_grad():
            features = {k: v.to(model.device) for k, v in features.items()}
            out_features = model(**features, return_dict=True)["logits"]

        return out_features.detach().cpu()

    def encode_post(self, out_features) -> list[float]:
        return out_features.flatten().to(torch.float32).numpy()

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]
