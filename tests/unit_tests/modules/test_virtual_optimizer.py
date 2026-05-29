# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torchtitan_npu.patches.optimizer.virtual_optimizer import (
    is_swap_tensor,
    patched_state_dict,
    unwrap_dtensor,
    virtual_optimizer_step_impl,
    VirtualAllocator,
    VirtualOptimizersContainer,
    wrap_like_param,
)


@pytest.fixture(scope="module", autouse=True)
def setup_dist_env():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_unwrap_dtensor_logic():
    plain_tensor = torch.randn(4, 4)
    assert unwrap_dtensor(plain_tensor) is plain_tensor

    mesh = DeviceMesh("cpu", [0])
    dtensor = DTensor.from_local(plain_tensor, mesh, [Replicate()])
    result = unwrap_dtensor(dtensor)
    assert not isinstance(result, DTensor)
    assert torch.equal(result, plain_tensor)


def test_wrap_like_param():
    mesh = DeviceMesh("cpu", [0])
    local_tensor = torch.randn(4, 4)
    p = DTensor.from_local(torch.randn(4, 4), mesh, [Shard(0)])

    wrapped = wrap_like_param(local_tensor, p)

    assert isinstance(wrapped, DTensor)
    assert wrapped.device_mesh == p.device_mesh
    assert wrapped.placements == p.placements
    assert wrapped.size() == p.size()


def test_is_swap_tensor():
    t1 = torch.randn(2, 2)
    assert not is_swap_tensor(t1)

    t2 = torch.randn(2, 2)
    t2.swap_tensor = True
    assert is_swap_tensor(t2)


def test_virtual_allocator_memory_logic():
    alloc = VirtualAllocator(pp_rank=0, pp_stages=2, virtual_optimizer_size=10.0)
    assert alloc.get_swap_memory_sizes() == [10.0, 10.0]

    alloc_list = VirtualAllocator(
        pp_rank=1, pp_stages=2, virtual_optimizer_size=[5.0, 15.0]
    )
    assert alloc_list.get_swap_memory_sizes() == [5.0, 15.0]

    p = torch.randn(2, 2)
    with patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu", spec=[]):
        mem = alloc.create(p)
        assert mem.shape == p.shape
        assert torch.all(mem == 0)


@patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu")
def test_virtual_allocator_npu_swap(mock_npu):
    mock_swap_tensor = torch.randn(2, 2)
    mock_npu.empty_with_swapped_memory.return_value = mock_swap_tensor

    p = torch.randn(2, 2)
    alloc = VirtualAllocator(pp_rank=0, pp_stages=1, virtual_optimizer_size=1.0)

    with patch(
        "torchtitan_npu.patches.optimizer.virtual_optimizer.hasattr", return_value=True
    ):
        res = alloc._get_swap_memory(p)
        assert getattr(res, "swap_tensor", False) is True


@patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu")
def test_virtual_optimizer_step_impl_logic(mock_npu, monkeypatch):
    mock_npu.npu.current_device.return_value = torch.device("cpu")

    def mock_npu_method(self, *args, **kwargs):
        return self

    monkeypatch.setattr(torch.Tensor, "npu", mock_npu_method)

    mesh = DeviceMesh("cpu", [0])
    local_p = torch.randn(2, 2, requires_grad=True)
    local_p.grad = torch.randn(2, 2)

    p_dtensor = DTensor.from_local(local_p.detach(), mesh, [Replicate()])
    p_dtensor.grad = DTensor.from_local(local_p.grad, mesh, [Replicate()])

    class MockOptimizer:
        def __init__(self):
            self.param_groups = [
                {
                    "params": [p_dtensor],
                    "lr": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                    "decoupled_weight_decay": True,
                }
            ]
            self.state = {p_dtensor: {}}
            self.virtual_allocator = MagicMock()
            self.virtual_allocator.init_exp.return_value = (
                torch.zeros(2, 2),
                torch.zeros(2, 2),
            )
            self.print_swap_flag = False

    opt = MockOptimizer()

    with patch("torch._fused_adamw_") as mock_fused:
        virtual_optimizer_step_impl(opt)
        assert "exp_avg" in opt.state[p_dtensor]
        assert isinstance(opt.state[p_dtensor]["exp_avg"], DTensor)
        assert mock_fused.called


def create_zero_tensor(x):
    return torch.zeros_like(x)


def test_patched_state_dict_basic():
    model = torch.nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    opt._original_state_dict = opt.state_dict
    opt.virtual_allocator = MagicMock()
    opt.virtual_allocator.create = create_zero_tensor

    state_dict = patched_state_dict(opt)
    assert "state" in state_dict
    assert "param_groups" in state_dict


def test_virtual_optimizers_container():
    model_part = torch.nn.Linear(10, 10)

    config = VirtualOptimizersContainer.Config(
        name="AdamW",
        lr=0.001,
        weight_decay=0.01,
        virtual_optimizer=True,
        virtual_optimizer_size=20.0,
    )

    container = VirtualOptimizersContainer(config, model_parts=[model_part])

    assert len(container.optimizers) == 1
    opt = container.optimizers[0]

    assert hasattr(opt, "_allocator_config")
    pp_rank, pp_size, virtual_size = opt._allocator_config
    assert pp_size == 1
    assert virtual_size == 20.0
