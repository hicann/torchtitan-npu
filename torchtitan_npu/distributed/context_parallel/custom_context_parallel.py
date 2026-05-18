# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plugin registry for custom Context Parallel strategies.

CP strategies are registered via ``register_cp_strategy(detector, applier)``.
When ``apply_cp_to_attention_module`` walks the model, each module is tested
against registered detectors in insertion order.  The first matching strategy is
applied.  Unmatched modules fall through to the original torchtitan CP logic.
"""

from collections.abc import Callable, Sequence

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

_cp_strategies: list[
    tuple[Callable[[nn.Module, str], bool], Callable[[nn.Module, DeviceMesh], None]]
] = []


def register_cp_strategy(
    detector: Callable[[nn.Module, str], bool],
    applier: Callable[[nn.Module, DeviceMesh], None],
) -> None:
    _cp_strategies.append((detector, applier))


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    attention_type: str,
) -> None:
    for module in attention_modules:
        applied = False
        for detector, applier in _cp_strategies:
            if detector(module, attention_type):
                applier(module, cp_mesh)
                applied = True
                break
        if not applied:
            raise NotImplementedError(
                f"No custom CP strategy found for module "
                f"{type(module).__name__} with attention_type={attention_type!r}"
            )


def _is_dsv4_attention(module: nn.Module, _attention_type: str) -> bool:
    return hasattr(module, "compress_ratio") and hasattr(module, "pre_attention")


def _apply_dsv4_cp(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    from .deepseek_v4_cp import DeepSeekV4PreAttentionCP

    parallelize_module(
        # pyrefly: ignore [bad-argument-type]
        module.pre_attention,
        cp_mesh,
        # pyrefly: ignore [bad-argument-type]
        DeepSeekV4PreAttentionCP(compress_ratio=module.compress_ratio),
    )
