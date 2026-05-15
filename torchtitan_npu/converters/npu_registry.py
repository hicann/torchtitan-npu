# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .framework.model_custom_config_registry import registry as npu_registry


def registry():
    return npu_registry


def register_model_converter(name: str):
    return npu_registry.register(name)


def get_model_converter_config(name: str) -> Configurable.Config | None:
    return npu_registry.get_config(name)


G_USING_TRAIN_CONFIG: Trainer.Config | None = None


def get_using_model_spec() -> ModelSpec:
    if G_USING_TRAIN_CONFIG is None:
        raise RuntimeError("G_USING_TRAIN_CONFIG must be set before using.")
    return G_USING_TRAIN_CONFIG.model_spec


_original_init_distributed = Trainer.init_distributed


def init_distributed_wrapper(self) -> ParallelDims:
    global G_USING_TRAIN_CONFIG
    G_USING_TRAIN_CONFIG = self.config
    return _original_init_distributed(self)


Trainer.init_distributed = init_distributed_wrapper
