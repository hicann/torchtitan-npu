# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from MindSpeed,
# https://gitcode.com/Ascend/MindSpeed/blob/master/mindspeed/core/optimizer/swap_optimizer/swap_optimizer.py
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
from typing import Any, TypeVar

import torch
import torch.nn as nn
import torchtitan
from torch.distributed.checkpoint.state_dict import _get_fqns
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim import Optimizer
from torch.optim.optimizer import _use_grad_for_differentiable
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.utils import get_device_info

from torchtitan_npu.patches.optimizer.muon_optimizer import build_muon_hybrid_optimizers


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Optimizer)
_original_build_optimizers = (
    torchtitan.components.optimizer.build_optimizers  # pyrefly: ignore[implicit-import]
)


def get_torch_device():
    # get torch.device
    return get_device_info()[1]


def unwrap_dtensor(tensor):
    """get normal tensor"""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def wrap_like_param(local_tensor: torch.Tensor, tensor):
    if isinstance(tensor, DTensor):
        return DTensor.from_local(
            local_tensor,
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
            shape=tensor.size(),
            stride=tensor.stride(),
            run_check=False,
        )
    return local_tensor


def wrap_like_param_without_device_move(local_tensor: torch.Tensor, tensor):
    if isinstance(tensor, DTensor):
        spec = DTensorSpec(
            tensor.device_mesh,
            tensor.placements,
            tensor_meta=TensorMeta(
                shape=tensor.size(),
                stride=tensor.stride(),
                dtype=local_tensor.dtype,
            ),
        )
        return DTensor(
            local_tensor.view_as(local_tensor),
            spec,
            requires_grad=local_tensor.requires_grad,
        )
    return local_tensor


class SwapOptimizersContainer(OptimizersContainer):
    """A container for optimizers which can be swapped between host and device to save memory during training.

    It will offload the optimizer states to the host (CPU) during the forward and backward passes.
    During the optimizer.step(), it will load, update, and offload these states in slices.
    This pipelined approach significantly reduces GPU memory pressure during the optimizer step,
    making it highly beneficial for memory-intensive scenarios.
    """

    swap_to_device_stream = None
    swap_to_host_stream = None

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}

    swap_to_host_events_map = {}
    swap_to_device_events_map = {}
    param_update_events_map = {}

    state_keys = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
    _MISSING = object()
    MISSING = _MISSING

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        swap_optimizer_times: int,
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # create streams for swapping
        if SwapOptimizersContainer.swap_to_device_stream is None:
            SwapOptimizersContainer.swap_to_device_stream = get_torch_device().Stream()
            SwapOptimizersContainer.swap_to_host_stream = get_torch_device().Stream()

        # initialize states and cpu counterparts for each device param
        for idx, optim in enumerate(self.optimizers):
            optim.param_to_group_map = {}
            for group in optim.param_groups:
                for p in group["params"]:
                    optim.param_to_group_map[p] = group
                    SwapOptimizersContainer.param_state_initialization(p, optim)
            swap_num = sum(
                [
                    sum([unwrap_dtensor(p).numel() for p in group["params"]])
                    for group in optim.param_groups
                ]
            )
            optim.swap_numel = swap_num // swap_optimizer_times
            logger.info(
                f"Swap param numel for optimizer_{idx}: {optim.swap_numel} / {swap_num}\n"
            )

    @staticmethod
    def _restore_states(original_states):
        for state, original_state in original_states:
            state.clear()
            state.update(original_state)

    @staticmethod
    def _empty_device_cache():
        try:
            get_torch_device().empty_cache()
        except Exception as exc:
            logger.debug("Failed to empty device cache after optimizer load: %s", exc)

    @staticmethod
    def _move_step_to_device(step):
        if torch.is_tensor(step) and step.device.type == "cpu":
            return step.to(get_torch_device().current_device())
        return step

    @staticmethod
    def _save_optimizer_states():
        original_states = []
        for cpu_state in SwapOptimizersContainer.param_to_cpu_states_map.values():
            original_states.append((cpu_state, dict(cpu_state)))
        for state in SwapOptimizersContainer.param_to_device_states_map.values():
            original_states.append((state, dict(state)))
        return original_states

    @staticmethod
    def _save_param_groups(optimizers):
        return [
            (group, dict(group)) for optim in optimizers for group in optim.param_groups
        ]

    @staticmethod
    def _restore_param_groups(original_param_groups):
        for group, original_group in original_param_groups:
            group.clear()
            group.update(original_group)

    @staticmethod
    def _param_group_value_for_state_dict(value):
        if torch.is_tensor(value):
            return unwrap_dtensor(value).detach().cpu().clone()
        return value

    @staticmethod
    def _param_names_by_param(model_part):
        param_names_by_param: dict[Any, list[str]] = {}
        try:
            named_parameters = model_part.named_parameters(remove_duplicate=False)
        except TypeError:
            named_parameters = model_part.named_parameters()
        for name, param in named_parameters:
            param_names = param_names_by_param.get(param)
            if param_names is None:
                param_names = []
                param_names_by_param[param] = param_names
            param_names.append(name)
        return param_names_by_param

    @staticmethod
    def _fqns_by_param(model_part):
        fqns_by_param = {}
        param_names_by_param = SwapOptimizersContainer._param_names_by_param(model_part)
        for name, param in model_part.named_parameters():
            ordered_fqns = []
            for param_name in param_names_by_param.get(param, [name]):
                fqns = set(_get_fqns(model_part, param_name))
                if not fqns:
                    raise AssertionError(
                        f"Expected at least 1 FQN for parameter '{param_name}', got 0"
                    )
                if len(fqns) > 1 and param_name not in fqns:
                    raise NotImplementedError(
                        "Swap optimizer checkpoint does not support saving "
                        f"flattened parameter '{param_name}' that maps to multiple "
                        f"FQNs: {sorted(fqns)}"
                    )
                ordered_fqns.extend(
                    fqn for fqn in [param_name, *sorted(fqns)] if fqn in fqns
                )
            ordered_fqns = list(dict.fromkeys(ordered_fqns))
            fqns_by_param[param] = tuple(ordered_fqns)
        return fqns_by_param

    @classmethod
    def param_state_initialization(cls, param, optim):
        cls.swap_to_host_events_map[param] = None

        device_state = optim.state[param]
        cls.param_to_device_states_map[param] = device_state
        cpu_state = {}
        cls.param_to_cpu_states_map[param] = cpu_state

        amsgrad = optim.param_to_group_map[param]["amsgrad"]

        for key in cls.state_keys:
            if key in device_state:
                continue
            if key == "max_exp_avg_sq" and not amsgrad:
                device_state[key] = None
                cpu_state[key] = None
            else:
                local_param = unwrap_dtensor(param)
                cpu_state[key] = torch.zeros_like(
                    local_param, pin_memory=True, device="cpu"
                )
                device_state[key] = cls._clone_loaded_state_for_device_placeholder(
                    param,
                    cpu_state[key],
                )

    @classmethod
    def swap_states_to_device(cls, param):
        if param not in cls.param_to_cpu_states_map:
            return

        cpu_state = cls.param_to_cpu_states_map[param]
        device_state = cls.param_to_device_states_map[param]
        local_param = unwrap_dtensor(param)
        for key in cls.state_keys:
            if key not in cpu_state or cpu_state[key] is None:
                continue
            local_state = unwrap_dtensor(device_state[key])
            if local_state.untyped_storage().size() == 0:
                if local_state.device != local_param.device:
                    local_state = torch.empty_strided(
                        cpu_state[key].size(),
                        cpu_state[key].stride(),
                        dtype=cpu_state[key].dtype,
                        layout=cpu_state[key].layout,
                        device=local_param.device,
                    )
                    device_state[key] = wrap_like_param(local_state, param)
                else:
                    local_state.untyped_storage().resize_(
                        cpu_state[key].untyped_storage().size()
                    )
                local_state.copy_(cpu_state[key], non_blocking=True)

        cls.swap_to_device_events_map[param] = (
            get_torch_device().current_stream().record_event()
        )

    @classmethod
    def swap_states_to_host(cls, param):
        if param not in cls.param_to_device_states_map:
            return

        device_state = cls.param_to_device_states_map[param]
        cpu_state = cls.param_to_cpu_states_map[param]
        for key in cls.state_keys:
            if key not in device_state or device_state[key] is None:
                continue
            local_state = unwrap_dtensor(device_state[key])
            if local_state.untyped_storage().size() != 0:
                cpu_state[key].copy_(local_state, non_blocking=True)
                local_state.untyped_storage().resize_(0)

        cls.swap_to_host_events_map[param] = (
            get_torch_device().current_stream().record_event()
        )

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            get_torch_device().current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def wait_param_update_event(cls, param):
        event = cls.param_update_events_map.get(param, None)
        if event is not None:
            get_torch_device().current_stream().wait_event(event)
            cls.param_update_events_map[param] = None

    @classmethod
    def _tensor_for_state_dict(cls, tensor, like_param=None):
        local_tensor = unwrap_dtensor(tensor)
        if local_tensor.numel() != 0 and local_tensor.untyped_storage().size() == 0:
            raise RuntimeError(
                "Cannot checkpoint a swapped optimizer state without CPU cache."
            )
        if local_tensor.device.type == "cpu":
            cpu_tensor = local_tensor.detach()
        else:
            cpu_tensor = local_tensor.detach().cpu()
        if isinstance(like_param, DTensor):
            return wrap_like_param_without_device_move(cpu_tensor, like_param)
        return cpu_tensor

    @classmethod
    def _state_value_for_state_dict(cls, param, state, key):
        cpu_state = cls.param_to_cpu_states_map.get(param)
        if cpu_state is not None:
            cpu_value = cpu_state.get(key)
            if cpu_value is not None:
                return cls._tensor_for_state_dict(cpu_value, param)

        value = state.get(key)
        if value is None:
            return None
        return cls._tensor_for_state_dict(value, param)

    @classmethod
    def _state_step_for_state_dict(cls, optim, param, state):
        if "step" in state:
            step = state["step"]
            if torch.is_tensor(step):
                return step.detach().cpu().clone()
            return step

        group = getattr(optim, "param_to_group_map", {}).get(param)
        if group is not None and "step" in group:
            step = group["step"]
            if torch.is_tensor(step):
                return step.detach().cpu().clone()
            return step
        if any(state.get(key) is not None for key in cls.state_keys):
            return torch.tensor(0, dtype=torch.int64, device="cpu")
        return None

    @classmethod
    def _add_param_group_to_state_dict(cls, state_dict, group, fqn):
        for key, value in group.items():
            if key == "params":
                continue
            state_dict[
                f"param_groups.{fqn}.{key}"
            ] = cls._param_group_value_for_state_dict(value)

    @classmethod
    def _add_param_state_to_state_dict(cls, state_dict, optim, param, fqn):
        state = optim.state[param]
        for key in cls.state_keys:
            value = cls._state_value_for_state_dict(param, state, key)
            if value is not None:
                state_dict[f"state.{fqn}.{key}"] = value

        step = cls._state_step_for_state_dict(optim, param, state)
        if step is not None:
            state_dict[f"state.{fqn}.step"] = step

    @classmethod
    def _optimizer_state_dict(cls, model_part, optim):
        fqns_by_param = cls._fqns_by_param(model_part)
        state_dict = {}
        for group in optim.param_groups:
            for param in group["params"]:
                fqn = fqns_by_param[param][0]
                cls._add_param_state_to_state_dict(state_dict, optim, param, fqn)
                cls._add_param_group_to_state_dict(state_dict, group, fqn)
        return state_dict

    @classmethod
    def _wait_pending_swap_to_host(cls):
        if cls.swap_to_host_stream is None:
            return
        get_torch_device().current_stream().wait_stream(cls.swap_to_host_stream)

    @classmethod
    def _clone_to_cpu_cache(cls, tensor):
        cpu_tensor = unwrap_dtensor(tensor).detach().cpu()
        try:
            cached_tensor = torch.empty_like(
                cpu_tensor,
                pin_memory=True,
                device="cpu",
            )
            cached_tensor.copy_(cpu_tensor, non_blocking=True)
            return cached_tensor
        except RuntimeError:
            return cpu_tensor.clone()

    @classmethod
    def _clone_loaded_state_for_device_placeholder(cls, param, tensor):
        local_tensor = unwrap_dtensor(tensor)
        local_param = unwrap_dtensor(param)
        placeholder = torch.empty_strided(
            local_tensor.size(),
            local_tensor.stride(),
            dtype=local_tensor.dtype,
            layout=local_tensor.layout,
            device=local_tensor.device,
        )
        placeholder.untyped_storage().resize_(0)
        if placeholder.device != local_param.device:
            return placeholder
        return wrap_like_param(placeholder, param)

    @classmethod
    def _clone_loaded_value(cls, value):
        if torch.is_tensor(value):
            return unwrap_dtensor(value).detach().clone()
        return value

    @classmethod
    def _state_dict_value_for_fqns(cls, state_dict, prefix, fqns, key):
        for fqn in fqns:
            flat_key = f"{prefix}.{fqn}.{key}"
            if flat_key in state_dict:
                return state_dict[flat_key]
        return cls._MISSING

    @classmethod
    def _load_param_group(cls, group, fqns, state_dict):
        for key in group:
            if key == "params":
                continue
            value = cls._state_dict_value_for_fqns(
                state_dict, "param_groups", fqns, key
            )
            if value is not cls._MISSING:
                group[key] = cls._clone_loaded_value(value)

    @classmethod
    def _load_param_state(cls, optim, param, fqns, state_dict):
        group = optim.param_to_group_map[param]
        state = optim.state[param]
        state.clear()
        cpu_state = cls.param_to_cpu_states_map.setdefault(param, {})
        cls.param_to_device_states_map[param] = state

        for key in cls.state_keys:
            value = cls._state_dict_value_for_fqns(state_dict, "state", fqns, key)
            if value is not cls._MISSING:
                cpu_state[key] = cls._clone_to_cpu_cache(value)
                state[key] = cls._clone_loaded_state_for_device_placeholder(
                    param,
                    cpu_state[key],
                )
                unwrap_dtensor(state[key]).untyped_storage().resize_(0)
            elif key == "max_exp_avg_sq" and not group["amsgrad"]:
                state[key] = None
                cpu_state[key] = None
            else:
                cpu_state.pop(key, None)

        step = cls._state_dict_value_for_fqns(state_dict, "state", fqns, "step")
        if step is cls._MISSING:
            step = cls._state_dict_value_for_fqns(
                state_dict, "param_groups", fqns, "step"
            )
        if step is not cls._MISSING:
            group["step"] = cls._clone_loaded_value(step)

    @classmethod
    def _load_optimizer_state_dict(cls, model_part, optim, state_dict):
        fqns_by_param = cls._fqns_by_param(model_part)
        optim.param_to_group_map = {}

        for group in optim.param_groups:
            loaded_group = False
            for param in group["params"]:
                optim.param_to_group_map[param] = group
                fqns = fqns_by_param[param]
                if not loaded_group:
                    cls._load_param_group(group, fqns, state_dict)
                    loaded_group = True
                cls._load_param_state(optim, param, fqns, state_dict)

    def state_dict(self) -> dict[str, Any]:
        self._wait_pending_swap_to_host()
        state_dict = {}
        for model_part, optim in zip(self.model_parts, self.optimizers):
            state_dict.update(self._optimizer_state_dict(model_part, optim))
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        original_states = self._save_optimizer_states()
        original_param_groups = self._save_param_groups(self.optimizers)
        try:
            for model_part, optim in zip(self.model_parts, self.optimizers):
                self._load_optimizer_state_dict(model_part, optim, state_dict)
        except Exception:
            self._restore_states(original_states)
            self._restore_param_groups(original_param_groups)
            raise
        self._empty_device_cache()

    # Public aliases for classmethods used by subclasses
    fqns_by_param = _fqns_by_param
    clone_to_cpu_cache = _clone_to_cpu_cache
    state_step_for_state_dict = _state_step_for_state_dict
    add_param_state_to_state_dict = _add_param_state_to_state_dict
    add_param_group_to_state_dict = _add_param_group_to_state_dict
    optimizer_state_dict = _optimizer_state_dict
    wait_pending_swap_to_host = _wait_pending_swap_to_host
    load_param_group = _load_param_group
    load_param_state = _load_param_state
    state_dict_value_for_fqns = _state_dict_value_for_fqns
    clone_loaded_value = _clone_loaded_value
    load_optimizer_state_dict = _load_optimizer_state_dict
    empty_device_cache = _empty_device_cache


def param_update(param, state, param_group):
    beta1, beta2 = param_group["betas"]
    step_func = (
        torch._fused_adamw_
        if param_group["decoupled_weight_decay"]
        else torch._fused_adam_
    )
    step_func(
        [param],
        [param.grad],
        [state["exp_avg"]],
        [state["exp_avg_sq"]],
        [state["max_exp_avg_sq"]] if param_group["amsgrad"] else [],
        [param_group["step"]],
        amsgrad=param_group["amsgrad"],
        lr=param_group["lr"],
        beta1=beta1,
        beta2=beta2,
        weight_decay=param_group["weight_decay"],
        eps=param_group["eps"],
        maximize=param_group["maximize"],
    )


def pipeline_load_param(swap_numel, params_list, start_index, current_swap_count):
    torch_device = get_torch_device()
    torch_device.current_stream().wait_stream(
        SwapOptimizersContainer.swap_to_host_stream
    )

    with torch_device.stream(SwapOptimizersContainer.swap_to_device_stream):
        torch_device.current_stream().wait_stream(
            SwapOptimizersContainer.swap_to_host_stream
        )

        idx = start_index
        while idx < len(params_list):
            param_local = unwrap_dtensor(params_list[idx])
            if params_list[idx].grad is None:
                idx += 1
                continue  # skip no grad param

            numel = param_local.numel()
            if current_swap_count > 0 and current_swap_count + numel > swap_numel:
                break  # stop load params when the buffer is full

            SwapOptimizersContainer.swap_states_to_device(params_list[idx])
            current_swap_count += numel
            idx += 1

    return current_swap_count


@_use_grad_for_differentiable
def swap_optimizer_step(self, closure=None):
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "SwapOptimizer does not support torch.jit.script by now."
        )

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if "step" in group:
            group["step"] += 1
            group["step"] = SwapOptimizersContainer._move_step_to_device(group["step"])
        else:
            group["step"] = torch.tensor(
                1,
                dtype=torch.int64,
                device=get_torch_device().current_device(),
            )

    swap_count = 0
    params_list = [p for group in self.param_groups for p in group["params"]]
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError(
                "SwapOptimizer step function does not support sparse gradients for now."
            )

        state = self.state[param]
        group = self.param_to_group_map[param]
        amsgrad = group["amsgrad"]

        # state initialization
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        if "max_exp_avg_sq" not in state:
            state["max_exp_avg_sq"] = (
                torch.zeros_like(param, memory_format=torch.preserve_format)
                if amsgrad
                else None
            )

        # pipelined swap update (load -> update -> offload)
        # load
        if swap_count == 0:
            swap_count = pipeline_load_param(
                self.swap_numel, params_list, i, swap_count
            )

        # update
        SwapOptimizersContainer.wait_swap_to_device_event(param)
        param_update(param, state, group)
        SwapOptimizersContainer.param_update_events_map[param] = (
            get_torch_device().current_stream().record_event()
        )
        # offload
        with get_torch_device().stream(SwapOptimizersContainer.swap_to_host_stream):
            SwapOptimizersContainer.wait_param_update_event(param)
            swap_count -= unwrap_dtensor(param).numel()
            SwapOptimizersContainer.swap_states_to_host(param)

    return loss


@functools.wraps(_original_build_optimizers)
def _build_optimizers_wrapper(
    model_parts, optimizer_config, parallel_dims, ft_manager=None
):
    if getattr(optimizer_config, "name", None) == "Muon":
        swap_optimizer = getattr(optimizer_config, "swap_optimizer", False)
        virtual_allocator = getattr(optimizer_config, "virtual_allocator", False)

        if swap_optimizer and virtual_allocator:
            raise ValueError(
                "Cannot use both swap_optimizer and virtual_allocator for Muon. "
                "Please set one of them to false."
            )

        if swap_optimizer:
            from torchtitan_npu.patches.optimizer.swap_muon_optimizer import (
                build_swap_muon_hybrid_optimizers,
            )

            return build_swap_muon_hybrid_optimizers(
                model_parts,
                optimizer_config,
                parallel_dims,
                ft_manager,
            )

        return build_muon_hybrid_optimizers(
            model_parts,
            optimizer_config,
            parallel_dims,
            ft_manager,
            virtual_allocator=virtual_allocator,
        )

    if getattr(optimizer_config, "swap_optimizer", False):
        # patch optimizer step functions
        torch.optim.AdamW.step = swap_optimizer_step
        torch.optim.Adam.step = swap_optimizer_step

        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }

        name = optimizer_config.name
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        optimizer_cls = optimizer_classes[name]

        optimizer_kwargs = {
            "lr": optimizer_config.lr,
            "betas": (optimizer_config.beta1, optimizer_config.beta2),
            "eps": optimizer_config.eps,
            "weight_decay": optimizer_config.weight_decay,
            "fused": optimizer_config.implementation == "fused",
            "foreach": optimizer_config.implementation == "foreach",
        }

        logger.info(f"[Patch] Building SwapOptimizersContainer with {name}")
        return SwapOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            optimizer_config.swap_optimizer_times,
        )

    # original optimizers
    return _original_build_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )


# patch build_optimizers function
torchtitan.components.optimizer.build_optimizers = (  # pyrefly: ignore[implicit-import]
    _build_optimizers_wrapper
)
