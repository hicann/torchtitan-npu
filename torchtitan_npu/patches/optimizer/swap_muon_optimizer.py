# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/components/optimizer.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Swap Muon optimizer: offloads Muon momentum_buffer to CPU and swaps on demand.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torchtitan.components.ft import FTManager
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    build_muon_hybrid_optimizers,
    DistributedMuon,
    MuonHybridOptimizersContainer,
)
from torchtitan_npu.patches.optimizer.swap_optimizer import (
    get_torch_device,
    SwapOptimizersContainer,
    unwrap_dtensor,
    wrap_like_param,
    wrap_like_param_without_device_move,
)


class SwapMuonState:
    """Per-parameter swap state for Muon momentum_buffer.

    Manages CPU↔GPU swap lifecycle for a single parameter's momentum_buffer.
    Uses delete-and-recreate pattern: swap_to_host copies to CPU then deletes
    the GPU tensor (sets state to None); swap_to_device creates a fresh tensor.
    This avoids resize_(0)→empty_strided roundtrip which crashes on NPU after
    HCCL all_to_all / all_reduce.
    """

    def __init__(self, param, device_module):
        self.param = param
        self.device_module = device_module
        self._cpu_momentum = None
        self._on_device = True
        self._swap_event = None
        self._optim_state: dict | None = None
        self._buf_shape = None
        self._buf_dtype = None

    @property
    def cpu_momentum(self):
        return self._cpu_momentum

    @cpu_momentum.setter
    def cpu_momentum(self, value):
        self._cpu_momentum = value

    @property
    def on_device(self):
        return self._on_device

    @on_device.setter
    def on_device(self, value):
        self._on_device = value

    @property
    def buf_shape(self):
        return self._buf_shape

    @buf_shape.setter
    def buf_shape(self, value):
        self._buf_shape = value

    @property
    def buf_dtype(self):
        return self._buf_dtype

    @buf_dtype.setter
    def buf_dtype(self, value):
        self._buf_dtype = value

    @property
    def optim_state(self):
        return self._optim_state

    @optim_state.setter
    def optim_state(self, value):
        self._optim_state = value

    def init_from_momentum_buffer(self, momentum_buffer):
        local_buf = unwrap_dtensor(momentum_buffer)
        self._cpu_momentum = torch.zeros_like(local_buf, pin_memory=True, device="cpu")
        self._cpu_momentum.copy_(local_buf, non_blocking=False)
        self._buf_shape = local_buf.shape
        self._buf_dtype = local_buf.dtype
        self._set_momentum_buffer(None)
        self._on_device = False

    def swap_to_device(self, stream=None):
        if self._cpu_momentum is None or self._on_device:
            return

        state = self._get_momentum_buffer()
        cpu = self._cpu_momentum

        if state is None:
            local_param = unwrap_dtensor(self.param)
            local_state = torch.empty(  # pyrefly: ignore[no-matching-overload]
                self._buf_shape,
                dtype=self._buf_dtype,
                device=local_param.device,
            )
            self._set_momentum_buffer(
                wrap_like_param(local_state, self.param)
                if isinstance(self.param, DTensor)
                else local_state
            )

        local_state = unwrap_dtensor(self._get_momentum_buffer())
        local_state.copy_(cpu, non_blocking=True)
        self._on_device = True

        if stream is not None:
            self._swap_event = stream.record_event()
        else:
            self._swap_event = self.device_module.current_stream().record_event()

    def swap_to_host(self, stream=None):
        if self._cpu_momentum is None or not self._on_device:
            return

        state = self._get_momentum_buffer()
        if state is not None:
            local_state = unwrap_dtensor(state)
            if local_state.untyped_storage().size() != 0:
                self._cpu_momentum.copy_(local_state, non_blocking=True)
            self._set_momentum_buffer(None)
        self._on_device = False

        if stream is not None:
            self._swap_event = stream.record_event()
        else:
            self._swap_event = self.device_module.current_stream().record_event()

    def wait_swap(self):
        if self._swap_event is not None:
            self.device_module.current_stream().wait_event(self._swap_event)
            self._swap_event = None

    def _get_momentum_buffer(self):
        if self._optim_state is not None:
            return self._optim_state.get("momentum_buffer")
        return None

    def _set_momentum_buffer(self, value):
        if self._optim_state is not None:
            self._optim_state["momentum_buffer"] = value


class SwapMuonHybridOptimizersContainer(MuonHybridOptimizersContainer):
    """Container for Muon + AdamW hybrid optimizers with swap support.

    Extends MuonHybridOptimizersContainer to add:
    - Muon momentum_buffer swap to/from CPU (via SwapMuonState)
    - Sliced (pipelined per-bucket) swap mode only
    - AdamW swap delegation to SwapOptimizersContainer (Phase 5)

    Uses delete-and-recreate pattern: swap_to_host copies to CPU then deletes
    the GPU tensor (sets state to None); swap_to_device creates a fresh tensor.
    This avoids resize_(0)→empty_strided roundtrip which crashes on NPU after
    HCCL all_to_all / all_reduce.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: list[Optimizer],
        muon_adjust_lr_fn: str | None = None,
        swap_optimizer_times: int = 16,
        swap_merge_buckets: int = 1,
    ) -> None:
        super().__init__(model_parts, optimizers, muon_adjust_lr_fn)

        if swap_optimizer_times <= 0:
            raise ValueError(
                f"swap_optimizer_times must be positive, got {swap_optimizer_times}"
            )
        if swap_merge_buckets <= 0:
            raise ValueError(
                f"swap_merge_buckets must be positive, got {swap_merge_buckets}"
            )

        self._swap_optimizer_times = swap_optimizer_times

        # Get device module
        self._device_module = get_torch_device()

        # Create swap streams (reuse class-level streams from SwapOptimizersContainer
        # if they already exist, otherwise create new ones)
        if SwapOptimizersContainer.swap_to_device_stream is None:
            SwapOptimizersContainer.swap_to_device_stream = self._device_module.Stream()
            SwapOptimizersContainer.swap_to_host_stream = self._device_module.Stream()
        self._swap_to_device_stream = SwapOptimizersContainer.swap_to_device_stream
        self._swap_to_host_stream = SwapOptimizersContainer.swap_to_host_stream

        # Initialize per-parameter swap states for Muon momentum_buffer
        self._muon_swap_states: dict[int, SwapMuonState] = {}
        muon_optim = self.optimizers[0]
        if not isinstance(muon_optim, DistributedMuon):
            raise TypeError(
                f"First optimizer must be DistributedMuon, got {type(muon_optim)}"
            )

        if not muon_optim.fsdp_enabled:
            raise RuntimeError(
                "Swap optimizer requires FSDP to be enabled; "
                "DDP is not supported with swap optimizer"
            )

        # Set swap attributes on DistributedMuon
        muon_optim._swap_enabled = True
        muon_optim._swap_container = self
        muon_optim._swap_merge_buckets = swap_merge_buckets

        muon_optim._device_module = self._device_module
        muon_optim._swap_to_device_stream = self._swap_to_device_stream
        muon_optim._swap_to_host_stream = self._swap_to_host_stream

        self._enable_adamw_swap(swap_optimizer_times)

        self._pre_init_swap_states()

        logger.info(
            f"[SwapMuon] Built SwapMuonHybridOptimizersContainer "
            f"swap_optimizer_times={swap_optimizer_times} | "
            f"swap_merge_buckets={swap_merge_buckets} | "
            f"muon_swap_states={len(self._muon_swap_states)}"
        )

    @property
    def muon_swap_states(self):
        return self._muon_swap_states

    def step(self, *args, **kwargs) -> None:
        muon_optim = self.optimizers[0]

        muon_optim.step(*args, **kwargs)

        if len(self.optimizers) > 1:
            self.optimizers[1].step(*args, **kwargs)

        if not self._muon_swap_states:
            logger.warning(
                "[SwapMuon] _muon_swap_states is empty after step(). "
                "This should not happen if _pre_init_swap_states ran in __init__."
            )

    def get_swap_state(self, param_id: int):
        return self._muon_swap_states.get(param_id)

    def state_dict(self) -> dict[str, Any]:
        self._wait_pending_swap_to_host()

        merged = {}
        muon_optim = self.optimizers[0]

        for model in self.model_parts:
            fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
            for group in muon_optim.param_groups:
                for param in group["params"]:
                    if param not in fqns_by_param:
                        continue
                    fqn = fqns_by_param[param][0]
                    self._serialize_param_state(param, fqn, group, muon_optim, merged)

        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is not None:
            for model in self.model_parts:
                merged.update(self._adamw_state_dict_for_model(model, adamw_optim))

        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        muon_optim = self.optimizers[0]

        if not self._muon_swap_states:
            self._ensure_swap_states_initialized()

        for model in self.model_parts:
            fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)

            for group in muon_optim.param_groups:
                loaded_group = False
                for param in group["params"]:
                    if param not in fqns_by_param:
                        continue
                    fqns = fqns_by_param[param]

                    if not loaded_group:
                        SwapOptimizersContainer.load_param_group(
                            group, fqns, state_dict
                        )
                        loaded_group = True

                    self._load_param_state(param, fqns, group, muon_optim, state_dict)

        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is not None:
            adamw_optim.param_to_group_map = {}
            for model in self.model_parts:
                self._load_adamw_state_dict_for_model(model, adamw_optim, state_dict)

        SwapOptimizersContainer.empty_device_cache()

    def _adamw_state_dict_for_model(self, model, optim):
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        state_dict = {}
        for group in optim.param_groups:
            for param in group["params"]:
                if param not in fqns_by_param:
                    continue
                fqn = fqns_by_param[param][0]
                SwapOptimizersContainer.add_param_state_to_state_dict(
                    state_dict, optim, param, fqn
                )
                SwapOptimizersContainer.add_param_group_to_state_dict(
                    state_dict, group, fqn
                )
        return state_dict

    def _load_adamw_state_dict_for_model(self, model, optim, state_dict):
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        for group in optim.param_groups:
            loaded_group = False
            for param in group["params"]:
                if param not in fqns_by_param:
                    continue
                optim.param_to_group_map[param] = group
                fqns = fqns_by_param[param]
                if not loaded_group:
                    SwapOptimizersContainer.load_param_group(group, fqns, state_dict)
                    loaded_group = True
                SwapOptimizersContainer.load_param_state(optim, param, fqns, state_dict)

    def _enable_adamw_swap(self, swap_optimizer_times: int):
        from torchtitan_npu.patches.optimizer.swap_optimizer import swap_optimizer_step

        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is None:
            return

        adamw_optim.param_to_group_map = {}
        for group in adamw_optim.param_groups:
            for p in group["params"]:
                adamw_optim.param_to_group_map[p] = group
                SwapOptimizersContainer.param_state_initialization(p, adamw_optim)

        swap_num = sum(
            unwrap_dtensor(p).numel()
            for group in adamw_optim.param_groups
            for p in group["params"]
        )
        adamw_optim.swap_numel = swap_num // swap_optimizer_times

        adamw_optim.step = swap_optimizer_step.__get__(adamw_optim, type(adamw_optim))

        logger.info(
            f"[SwapMuon] AdamW swap enabled: "
            f"{len(adamw_optim.param_groups[0]['params'])} params | "
            f"swap_numel={adamw_optim.swap_numel}"
        )

    def _pre_init_swap_states(self):
        muon_optim = self.optimizers[0]
        pre_init_count = 0
        for group in muon_optim.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = muon_optim.state.setdefault(p, {})
                local_p = unwrap_dtensor(p)
                zero_buf = torch.zeros_like(local_p)
                if isinstance(p, DTensor):
                    zero_buf = wrap_like_param(zero_buf, p)
                state["momentum_buffer"] = zero_buf

                swap_state = SwapMuonState(p, self._device_module)
                swap_state.optim_state = state
                swap_state.init_from_momentum_buffer(zero_buf)
                self._muon_swap_states[id(p)] = swap_state
                pre_init_count += 1

        logger.info(
            f"[SwapMuon] Pre-init swap states: {pre_init_count} momentum_buffers "
            f"created (zeros) and offloaded to CPU"
        )

    def _wait_pending_swap_to_host(self):
        SwapOptimizersContainer.wait_pending_swap_to_host()

    def _copy_device_momentum_to_cpu(self, swap_state, muon_optim, param):
        if not swap_state.on_device:
            return
        buf = muon_optim.state.get(param, {}).get("momentum_buffer")
        if buf is not None:
            local_buf = unwrap_dtensor(buf)
            if local_buf.untyped_storage().size() != 0:
                swap_state.cpu_momentum.copy_(local_buf, non_blocking=False)

    def _serialize_momentum_buffer(
        self, param: torch.Tensor, muon_optim: Optimizer
    ) -> torch.Tensor:
        swap_state = self._muon_swap_states.get(id(param))
        if swap_state is not None and swap_state.cpu_momentum is not None:
            self._copy_device_momentum_to_cpu(swap_state, muon_optim, param)
            cpu_tensor = SwapOptimizersContainer.clone_to_cpu_cache(
                swap_state.cpu_momentum
            )
            if isinstance(param, DTensor):
                cpu_tensor = wrap_like_param_without_device_move(cpu_tensor, param)
            return cpu_tensor

        local_p = unwrap_dtensor(param)
        placeholder = torch.zeros_like(local_p, device="cpu")
        if isinstance(param, DTensor):
            placeholder = wrap_like_param_without_device_move(placeholder, param)
        return placeholder

    def _serialize_param_state(self, param, fqn, group, muon_optim, merged):
        merged[f"state.{fqn}.momentum_buffer"] = self._serialize_momentum_buffer(
            param, muon_optim
        )
        state = muon_optim.state.get(param, {})
        step = SwapOptimizersContainer.state_step_for_state_dict(
            muon_optim, param, state
        )
        if step is not None:
            merged[f"state.{fqn}.step"] = step
        SwapOptimizersContainer.add_param_group_to_state_dict(merged, group, fqn)

    def _ensure_swap_states_initialized(self):
        muon_optim = self.optimizers[0]
        for group in muon_optim.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in self._muon_swap_states:
                    continue
                state = muon_optim.state.setdefault(p, {})
                if "momentum_buffer" not in state or state["momentum_buffer"] is None:
                    local_p = unwrap_dtensor(p)
                    cpu_buf = torch.zeros_like(local_p, pin_memory=True, device="cpu")
                    state["momentum_buffer"] = None
                    swap_state = SwapMuonState(p, self._device_module)
                    swap_state.optim_state = state
                    swap_state.cpu_momentum = cpu_buf
                    swap_state.buf_shape = local_p.shape
                    swap_state.buf_dtype = local_p.dtype
                    swap_state.on_device = False
                else:
                    swap_state = SwapMuonState(p, self._device_module)
                    swap_state.optim_state = state
                self._muon_swap_states[pid] = swap_state

    def _load_momentum_from_state_dict(self, swap_state, value, muon_optim, param):
        swap_state.cpu_momentum = SwapOptimizersContainer.clone_to_cpu_cache(value)
        if swap_state.cpu_momentum is not None:
            swap_state.buf_shape = swap_state.cpu_momentum.shape
            swap_state.buf_dtype = swap_state.cpu_momentum.dtype
        state = muon_optim.state.get(param, {})
        buf = state.get("momentum_buffer")
        if buf is not None:
            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            if buf_local.untyped_storage().size() != 0:
                buf_local.zero_()
        state["momentum_buffer"] = None
        swap_state.on_device = False

    def _load_step_from_state_dict(self, state_dict, fqns, group):
        step = SwapOptimizersContainer.state_dict_value_for_fqns(
            state_dict, "state", fqns, "step"
        )
        if step is SwapOptimizersContainer.MISSING:
            step = SwapOptimizersContainer.state_dict_value_for_fqns(
                state_dict, "param_groups", fqns, "step"
            )
        if step is not SwapOptimizersContainer.MISSING:
            group["step"] = SwapOptimizersContainer.clone_loaded_value(step)

    def _load_param_state(self, param, fqns, group, muon_optim, state_dict):
        swap_state = self._muon_swap_states.get(id(param))
        if swap_state is not None:
            value = SwapOptimizersContainer.state_dict_value_for_fqns(
                state_dict, "state", fqns, "momentum_buffer"
            )
            if value is not SwapOptimizersContainer.MISSING:
                self._load_momentum_from_state_dict(
                    swap_state, value, muon_optim, param
                )
        self._load_step_from_state_dict(state_dict, fqns, group)


def build_swap_muon_hybrid_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> SwapMuonHybridOptimizersContainer:
    """Build SwapMuonHybridOptimizersContainer with swap support.

    Reuses build_muon_hybrid_optimizers to create the base Muon + AdamW
    optimizers, then wraps them in SwapMuonHybridOptimizersContainer.
    """
    # Create base optimizers via the non-swap factory (without virtual_allocator)
    base_container = build_muon_hybrid_optimizers(
        model_parts,
        optimizer_config,
        parallel_dims,
        ft_manager,
        virtual_allocator=False,
    )

    swap_optimizer_times = getattr(optimizer_config, "swap_optimizer_times", 16)
    swap_merge_buckets = getattr(optimizer_config, "swap_merge_buckets", 1)

    return SwapMuonHybridOptimizersContainer(
        model_parts,
        base_container.optimizers,
        muon_adjust_lr_fn=base_container.muon_adjust_lr_fn,  # pyrefly: ignore[missing-attribute]
        swap_optimizer_times=swap_optimizer_times,
        swap_merge_buckets=swap_merge_buckets,
    )
