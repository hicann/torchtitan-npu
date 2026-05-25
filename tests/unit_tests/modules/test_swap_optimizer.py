# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import types
from unittest.mock import patch

import torch
import torch.distributed as dist
import torchtitan.components.optimizer as tt_optimizer
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate

from torchtitan_npu.patches.optimizer import swap_optimizer


def _make_cpu_mesh():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return DeviceMesh("cpu", [0])


def test_unwrap_dtensor_returns_plain_tensor_for_non_dtensor():
    tensor = torch.randn(2, 2)

    result = swap_optimizer.unwrap_dtensor(tensor)

    assert result is tensor


def test_build_optimizers_wrapper_delegates_when_swap_disabled():
    sentinel = object()
    calls = []
    with patch.object(
        swap_optimizer,
        "_original_build_optimizers",
        lambda model_parts, optimizer_config, parallel_dims, ft_manager: calls.append(
            (model_parts, optimizer_config, parallel_dims, ft_manager)
        )
        or sentinel,
    ):
        optimizer_config = types.SimpleNamespace(swap_optimizer=False)

        result = tt_optimizer.build_optimizers(
            model_parts=["model"],
            optimizer_config=optimizer_config,
            parallel_dims="parallel_dims",
            ft_manager="ft_manager",
        )

    assert result is sentinel
    assert calls == [(["model"], optimizer_config, "parallel_dims", "ft_manager")]


def test_build_optimizers_wrapper_rejects_unknown_optimizer(monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)

    optimizer_config = types.SimpleNamespace(
        swap_optimizer=True,
        name="SGD",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=8,
    )

    try:
        tt_optimizer.build_optimizers(
            model_parts=[],
            optimizer_config=optimizer_config,
            parallel_dims=None,
            ft_manager=None,
        )
        raise AssertionError("Expected NotImplementedError for unsupported optimizer")
    except NotImplementedError as exc:
        assert "Optimizer SGD not added" in str(exc)


def test_build_optimizers_wrapper_uses_swap_container(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)
    monkeypatch.setattr(
        swap_optimizer,
        "SwapOptimizersContainer",
        lambda model_parts, optimizer_cls, optimizer_kwargs, swap_times: calls.append(
            (model_parts, optimizer_cls, optimizer_kwargs, swap_times)
        )
        or "swap_container",
    )

    optimizer_config = types.SimpleNamespace(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=16,
    )

    result = tt_optimizer.build_optimizers(
        model_parts=["model_part"],
        optimizer_config=optimizer_config,
        parallel_dims=None,
        ft_manager=None,
    )

    assert result == "swap_container"
    assert calls[0][0] == ["model_part"]
    assert calls[0][1] is torch.optim.AdamW
    assert calls[0][2]["lr"] == 1e-3
    assert calls[0][3] == 16


class _TiedEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.embedding.weight


def _build_swapped_optimizer(
    monkeypatch,
    *,
    exp_avg,
    exp_avg_sq,
    step=None,
    model=None,
    param=None,
):
    if model is None:
        model = torch.nn.Linear(2, 2, bias=False)
    if param is None:
        param = model.weight
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    if step is not None:
        optimizer.param_groups[0]["step"] = torch.tensor(float(step))
    optimizer.param_to_group_map = {param: optimizer.param_groups[0]}

    state = optimizer.state[param]
    live_exp_avg = torch.zeros_like(param)
    live_exp_avg_sq = torch.zeros_like(param)
    live_exp_avg.untyped_storage().resize_(0)
    live_exp_avg_sq.untyped_storage().resize_(0)
    state["exp_avg"] = live_exp_avg
    state["exp_avg_sq"] = live_exp_avg_sq
    state["max_exp_avg_sq"] = None

    cpu_state = {
        "exp_avg": torch.full_like(param, exp_avg, device="cpu"),
        "exp_avg_sq": torch.full_like(param, exp_avg_sq, device="cpu"),
        "max_exp_avg_sq": None,
    }
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_cpu_states_map",
        {param: cpu_state},
    )
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_device_states_map",
        {param: state},
    )

    container = swap_optimizer.SwapOptimizersContainer.__new__(
        swap_optimizer.SwapOptimizersContainer
    )
    container.model_parts = [model]
    container.optimizers = [optimizer]

    return types.SimpleNamespace(
        container=container,
        model=model,
        optimizer=optimizer,
        param=param,
        cpu_state=cpu_state,
        live_exp_avg=live_exp_avg,
        live_exp_avg_sq=live_exp_avg_sq,
    )


def test_state_dict_uses_cpu_cache_snapshot_and_restores_swap_state(monkeypatch):
    fixture = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=1.25,
        exp_avg_sq=2.5,
        step=5,
    )

    state_dict = fixture.container.state_dict()
    fixture.optimizer.param_groups[0]["step"].add_(1)

    assert torch.equal(
        state_dict["state.weight.exp_avg"],
        fixture.cpu_state["exp_avg"],
    )
    assert torch.equal(
        state_dict["state.weight.exp_avg_sq"],
        fixture.cpu_state["exp_avg_sq"],
    )
    assert state_dict["state.weight.exp_avg"].device.type == "cpu"
    assert not isinstance(state_dict["state.weight.exp_avg"], DTensor)
    assert "state.weight.max_exp_avg_sq" not in state_dict
    assert state_dict["state.weight.step"].item() == 5
    assert state_dict["param_groups.weight.step"].item() == 5
    assert (
        state_dict["state.weight.step"] is not fixture.optimizer.param_groups[0]["step"]
    )
    assert (
        state_dict["param_groups.weight.step"]
        is not fixture.optimizer.param_groups[0]["step"]
    )

    state = fixture.optimizer.state[fixture.param]
    assert state["exp_avg"] is fixture.live_exp_avg
    assert state["exp_avg"].untyped_storage().size() == 0
    assert state["exp_avg_sq"] is fixture.live_exp_avg_sq
    assert state["exp_avg_sq"].untyped_storage().size() == 0
    assert state["max_exp_avg_sq"] is None
    assert "step" not in state


def test_state_dict_preserves_dtensor_layout_for_cpu_cache(monkeypatch):
    mesh = _make_cpu_mesh()
    local_param = torch.randn(2, 2)
    param = DTensor.from_local(local_param, mesh, [Replicate()])
    live_exp_avg = DTensor.from_local(
        torch.zeros_like(local_param), mesh, [Replicate()]
    )
    cpu_exp_avg = torch.ones_like(local_param, device="cpu")
    state = {"exp_avg": live_exp_avg}

    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_cpu_states_map",
        {param: {"exp_avg": cpu_exp_avg}},
    )
    monkeypatch.setattr(
        swap_optimizer.DTensor,
        "from_local",
        staticmethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CPU-cache state_dict must not call DTensor.from_local")
            )
        ),
    )

    value = swap_optimizer.SwapOptimizersContainer._state_value_for_state_dict(
        param,
        state,
        "exp_avg",
    )

    assert isinstance(value, DTensor)
    assert value.device_mesh == param.device_mesh
    assert value.placements == param.placements
    local_value = value.to_local()
    assert local_value.device.type == "cpu"
    assert (
        local_value.untyped_storage().data_ptr()
        == cpu_exp_avg.untyped_storage().data_ptr()
    )


def test_state_dict_allows_zero_numel_cpu_cache_shard():
    empty_shard = torch.empty(0, 16384, device="cpu")

    value = swap_optimizer.SwapOptimizersContainer._tensor_for_state_dict(empty_shard)

    assert value.shape == torch.Size([0, 16384])
    assert value.numel() == 0
    assert value.untyped_storage().size() == 0


def test_state_dict_rejects_nonempty_cache_placeholder_without_storage():
    placeholder = torch.empty(4, 16384, device="cpu")
    placeholder.untyped_storage().resize_(0)

    try:
        swap_optimizer.SwapOptimizersContainer._tensor_for_state_dict(placeholder)
        raise AssertionError("Expected zero-storage nonempty tensor to be rejected")
    except RuntimeError as exc:
        assert "without CPU cache" in str(exc)


def test_load_state_dict_rebuilds_swap_state_without_mutating_checkpoint(monkeypatch):
    source = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=3.0,
        exp_avg_sq=4.0,
        step=5,
    )
    checkpoint_state_dict = source.container.state_dict()
    original_exp_avg = checkpoint_state_dict["state.weight.exp_avg"].clone()
    original_exp_avg_sq = checkpoint_state_dict["state.weight.exp_avg_sq"].clone()
    original_exp_avg_storage = (
        checkpoint_state_dict["state.weight.exp_avg"].untyped_storage().size()
    )
    original_exp_avg_sq_storage = (
        checkpoint_state_dict["state.weight.exp_avg_sq"].untyped_storage().size()
    )

    target = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=0.0,
        exp_avg_sq=0.0,
    )

    target.container.load_state_dict(checkpoint_state_dict)

    exp_avg_storage = (
        checkpoint_state_dict["state.weight.exp_avg"].untyped_storage().size()
    )
    exp_avg_sq_storage = (
        checkpoint_state_dict["state.weight.exp_avg_sq"].untyped_storage().size()
    )
    if exp_avg_storage != original_exp_avg_storage:
        raise AssertionError(f"exp_avg checkpoint storage mutated: {exp_avg_storage}")
    if exp_avg_sq_storage != original_exp_avg_sq_storage:
        raise AssertionError(
            f"exp_avg_sq checkpoint storage mutated: {exp_avg_sq_storage}"
        )
    if not torch.equal(checkpoint_state_dict["state.weight.exp_avg"], original_exp_avg):
        raise AssertionError("exp_avg checkpoint value mutated")
    if not torch.equal(
        checkpoint_state_dict["state.weight.exp_avg_sq"], original_exp_avg_sq
    ):
        raise AssertionError("exp_avg_sq checkpoint value mutated")

    assert (
        target.optimizer.param_to_group_map[target.param]
        is target.optimizer.param_groups[0]
    )
    assert target.optimizer.param_groups[0]["step"].item() == 5
    assert "step" not in target.optimizer.state[target.param]
    assert (
        swap_optimizer.SwapOptimizersContainer.param_to_device_states_map[target.param]
        is target.optimizer.state[target.param]
    )
    target_cpu_state = swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map[
        target.param
    ]
    assert torch.equal(target_cpu_state["exp_avg"], original_exp_avg)
    assert torch.equal(target_cpu_state["exp_avg_sq"], original_exp_avg_sq)
    assert target_cpu_state["max_exp_avg_sq"] is None
    assert target.optimizer.state[target.param]["exp_avg"].untyped_storage().size() == 0
    assert (
        target.optimizer.state[target.param]["exp_avg_sq"].untyped_storage().size() == 0
    )
    assert target.optimizer.state[target.param]["max_exp_avg_sq"] is None


def test_shared_parameter_aliases_use_canonical_save_and_alias_load(monkeypatch):
    source_model = _TiedEmbeddingModel()
    source = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=3.0,
        exp_avg_sq=4.0,
        step=5,
        model=source_model,
        param=source_model.embedding.weight,
    )

    fqns_by_param = swap_optimizer.SwapOptimizersContainer._fqns_by_param(source_model)
    assert fqns_by_param[source_model.embedding.weight] == (
        "embedding.weight",
        "lm_head.weight",
    )

    checkpoint_state_dict = source.container.state_dict()
    assert "state.embedding.weight.exp_avg" in checkpoint_state_dict
    assert "state.lm_head.weight.exp_avg" not in checkpoint_state_dict
    assert "param_groups.embedding.weight.lr" in checkpoint_state_dict
    assert "param_groups.lm_head.weight.lr" not in checkpoint_state_dict

    alias_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("state.embedding.weight."):
            key = key.replace("state.embedding.weight.", "state.lm_head.weight.", 1)
        elif key.startswith("param_groups.embedding.weight."):
            key = key.replace(
                "param_groups.embedding.weight.", "param_groups.lm_head.weight.", 1
            )
        alias_state_dict[key] = value

    target_model = _TiedEmbeddingModel()
    target = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=0.0,
        exp_avg_sq=0.0,
        model=target_model,
        param=target_model.embedding.weight,
    )
    target.optimizer.param_groups[0]["lr"] = 0.5

    target.container.load_state_dict(alias_state_dict)

    assert target.optimizer.param_groups[0]["lr"] == 1e-3
    assert target.optimizer.param_groups[0]["step"].item() == 5
    target_cpu_state = swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map[
        target.param
    ]
    assert torch.equal(
        target_cpu_state["exp_avg"],
        alias_state_dict["state.lm_head.weight.exp_avg"],
    )
    assert torch.equal(
        target_cpu_state["exp_avg_sq"],
        alias_state_dict["state.lm_head.weight.exp_avg_sq"],
    )


def test_load_state_dict_clears_device_cache_after_rebuilding_swap_state(monkeypatch):
    source = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=3.0,
        exp_avg_sq=4.0,
        step=5,
    )
    checkpoint_state_dict = source.container.state_dict()
    target = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=0.0,
        exp_avg_sq=0.0,
    )
    calls = []

    class Device:
        def empty_cache(self):
            calls.append("empty_cache")

    monkeypatch.setattr(swap_optimizer, "get_torch_device", lambda: Device())

    target.container.load_state_dict(checkpoint_state_dict)

    assert calls == ["empty_cache"]


def test_state_dict_builds_flat_dict_without_upstream_optimizer_conversion(
    monkeypatch,
):
    fixture = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=1.0,
        exp_avg_sq=2.0,
        step=3,
    )

    def fail_upstream_state_dict(self):
        raise AssertionError("upstream state_dict should not be called")

    monkeypatch.setattr(
        tt_optimizer.OptimizersContainer,
        "state_dict",
        fail_upstream_state_dict,
    )

    state_dict = fixture.container.state_dict()

    assert set(state_dict) >= {
        "state.weight.exp_avg",
        "state.weight.exp_avg_sq",
        "state.weight.step",
        "param_groups.weight.lr",
        "param_groups.weight.betas",
        "param_groups.weight.amsgrad",
    }
    assert torch.equal(state_dict["state.weight.exp_avg"], fixture.cpu_state["exp_avg"])
    assert state_dict["state.weight.step"].item() == 3


def test_move_step_to_device_uses_configured_torch_device(monkeypatch):
    class Device:
        def current_device(self):
            return torch.device("cpu")

    monkeypatch.setattr(swap_optimizer, "get_torch_device", lambda: Device())

    step = torch.tensor(5, dtype=torch.int64)
    moved_step = swap_optimizer.SwapOptimizersContainer._move_step_to_device(step)

    assert moved_step.device.type == "cpu"
    assert moved_step.item() == 5


def test_loaded_plain_cpu_state_rebuilds_dtensor_runtime_placeholder():
    mesh = _make_cpu_mesh()
    local_param = torch.randn(2, 2)
    param = DTensor.from_local(local_param, mesh, [Replicate()])
    cpu_exp_avg = torch.ones_like(local_param, device="cpu")
    state = {}

    placeholder = swap_optimizer.SwapOptimizersContainer._clone_loaded_state_for_device_placeholder(
        param,
        cpu_exp_avg,
    )

    assert isinstance(placeholder, DTensor)
    assert placeholder.device_mesh == param.device_mesh
    assert placeholder.placements == param.placements
    assert placeholder.to_local().untyped_storage().size() == 0


def test_loaded_cpu_state_for_non_cpu_param_keeps_plain_zero_storage_placeholder(
    monkeypatch,
):
    param = types.SimpleNamespace(device=torch.device("meta"))
    cpu_exp_avg = torch.ones(2, 2, device="cpu")
    wrap_calls = []

    def forbid_wrap(local_tensor, like_tensor):
        wrap_calls.append(True)
        return local_tensor

    monkeypatch.setattr(swap_optimizer, "wrap_like_param", forbid_wrap)

    placeholder = swap_optimizer.SwapOptimizersContainer._clone_loaded_state_for_device_placeholder(
        param,
        cpu_exp_avg,
    )
    placeholder_device_type = placeholder.device.type
    placeholder_storage_size = placeholder.untyped_storage().size()
    del placeholder

    assert placeholder_device_type == "cpu"
    assert placeholder_storage_size == 0
    assert wrap_calls == []
