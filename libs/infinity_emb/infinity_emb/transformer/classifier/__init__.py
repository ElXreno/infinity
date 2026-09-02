# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

__all__ = ["classification_activation"]


def classification_activation(config) -> str:
    """Mirrors the default of transformers' TextClassificationPipeline: sigmoid for
    multi-label or single-logit heads, softmax for single-label heads, otherwise the
    config's own `function_to_apply`."""
    problem_type = getattr(config, "problem_type", None)
    num_labels = getattr(config, "num_labels", None) or 0
    if problem_type == "multi_label_classification" or num_labels == 1:
        return "sigmoid"
    if problem_type == "single_label_classification" or num_labels > 1:
        return "softmax"
    return str(getattr(config, "function_to_apply", None) or "none").lower()
