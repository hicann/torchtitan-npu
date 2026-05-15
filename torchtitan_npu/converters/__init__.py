# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

__all__ = [
    "registry",
    "register_model_converter",
    "get_model_converter_config",
    "register_npu_converter",
    "get_npu_converter_config",
    "ConverterRegistry",
    "BaseConverter",
    "NPUConverter",
    "ParallelizePlanUpdater",
    "StateDictUpdater",
    "ModelCustomConfig",
]

import importlib
import pkgutil
from pathlib import Path

from .base_converter import BaseConverter
from .model_custom_interface import (
    ModelCustomConfig,
    ParallelizePlanUpdater,
    StateDictUpdater,
)
from .npu_converter import NPUConverter
from .npu_registry import get_model_converter_config, register_model_converter

from .registry import (
    ConverterRegistry,
    get_npu_converter_config,
    register_npu_converter,
    registry,
)


def _auto_search_conveter():
    package_dir = Path(__file__).parent

    for subdir in ["kernels", "features"]:
        subdir_path = package_dir / subdir
        if subdir_path.exists():
            for _, name, _ in pkgutil.iter_modules([str(subdir_path)]):
                importlib.import_module(f".{subdir}.{name}", package=__package__)


_auto_search_conveter()
