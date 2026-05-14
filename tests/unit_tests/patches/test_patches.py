# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import torch

from torchtitan_npu.patches.quantization import quantize
from torchtitan_npu.patches.torch import clip_grad


def test_register_quantize_module_handler_registers_handler():
    class DummyConfig:
        pass

    def handler(module, config):
        return module

    handler_registry = {}
    with patch.object(quantize, "_QUANTIZE_CONFIG_HANDLER", handler_registry):
        decorated = quantize.register_quantize_module_handler(DummyConfig)(handler)
        assert decorated is handler
        assert handler_registry.get(DummyConfig) is handler


def test_group_dtensors_by_layout_groups_non_dtensors_together():
    tensor_a = torch.randn(2, 2)
    tensor_b = torch.randn(2, 2)

    grouped = clip_grad.group_dtensors_by_layout([tensor_a, tensor_b])

    assert len(grouped) == 1
    assert ("non_dtensor", None) in grouped
    assert grouped[("non_dtensor", None)] == [tensor_a, tensor_b]


def test_group_dtensors_by_layout_handles_empty_input():
    grouped = clip_grad.group_dtensors_by_layout([])

    assert grouped == {}


def test_register_quantize_module_handler_overrides_existing_handler():
    class DummyConfig:
        pass

    def old_handler(module, config):
        return module

    def new_handler(module, config):
        return module

    handler_registry = {DummyConfig: old_handler}
    with patch.object(quantize, "_QUANTIZE_CONFIG_HANDLER", handler_registry):
        quantize.register_quantize_module_handler(DummyConfig)(new_handler)
        assert handler_registry[DummyConfig] is new_handler
