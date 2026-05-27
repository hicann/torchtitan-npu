# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import torch.nn as nn
from torchtitan.components.optimizer import OptimizersContainer

from torchtitan_npu.patches.optimizer import swap_optimizer


def test_unwrap_dtensor_returns_plain_tensor_for_non_dtensor():
    tensor = torch.randn(2, 2)

    result = swap_optimizer.unwrap_dtensor(tensor)

    assert result is tensor


def test_swap_config_delegates_to_base_container_when_disabled():
    """swap_optimizer=False → Config.build() returns a plain OptimizersContainer."""
    model = nn.Linear(2, 2)
    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=False, name="AdamW", lr=1e-3, implementation="for-loop"
    )

    result = config.build(model_parts=[model])

    assert isinstance(result, OptimizersContainer)
    assert not isinstance(result, swap_optimizer.SwapOptimizersContainer)


def test_swap_config_rejects_unknown_optimizer():
    """Unknown optimizer name raises NotImplementedError before any NPU access."""
    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="SGD",
        lr=1e-3,
        implementation="for-loop",
        swap_optimizer_times=8,
    )

    with pytest.raises(NotImplementedError, match="Optimizer SGD not added"):
        config.build(model_parts=[nn.Linear(2, 2)])


def test_swap_config_routes_to_swap_container_when_enabled(monkeypatch):
    """swap_optimizer=True → Config.build() instantiates the _owner class.

    Mocks _owner so the test verifies dispatch routing without running the
    real SwapOptimizersContainer.__init__ (which touches NPU streams).
    """
    instantiated = []

    class FakeOwner:
        def __init__(self, *, config, model_parts):
            instantiated.append((config, model_parts))

    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer.Config, "_owner", FakeOwner
    )

    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        implementation="fused",
        swap_optimizer_times=16,
    )

    result = config.build(model_parts=["model_part"])

    assert isinstance(result, FakeOwner)
    assert len(instantiated) == 1
    cfg, parts = instantiated[0]
    assert cfg.swap_optimizer is True
    assert cfg.swap_optimizer_times == 16
    assert cfg.name == "AdamW"
    assert parts == ["model_part"]
