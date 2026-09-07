# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import importlib.metadata

from infinity_emb.args import EngineArgs
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray
from infinity_emb.env import MANAGER

# reexports
from infinity_emb.infinity_server import create_server
from infinity_emb.log_handler import logger
from infinity_emb.sync_engine import SyncEngineArray

__version__: str = importlib.metadata.version("infinity_emb")

__all__ = [
    "MANAGER",
    "AsyncEmbeddingEngine",
    "AsyncEngineArray",
    "EngineArgs",
    "SyncEngineArray",
    "__version__",
    "create_server",
    "logger",
]
