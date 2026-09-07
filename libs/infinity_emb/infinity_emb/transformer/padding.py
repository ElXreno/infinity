# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil



def padding_bucket(max_length: int, pad_to_multiple_of: int) -> tuple[int | None, int]:
    if not pad_to_multiple_of:
        return None, max_length
    multiple = min(pad_to_multiple_of, max_length)
    return multiple, max_length - max_length % multiple
