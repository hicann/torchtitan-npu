# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/models/moe/moe.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
    StateDictUpdater,
)
from torchtitan_npu.converters.npu_registry import register_model_converter
from torchtitan_npu.tools.weight_utils import _split_w13_for_mapping, fuse_experts

logger = logging.getLogger(__name__)

# Calculate the number of experts and EP degree, which are used as parameters
# when invoking operators during Hifloat8 low-precision training.
group_size_params = {
    "num_experts": None,
    "expert_model_parallel_size": None,
    "g_size": None,
}


def npu_grouped_mm(x, weight, group_list):
    # This function is replaced at runtime by quantization converters
    # (e.g. HiF8 / MXFP8) that patch the reference to quantize inputs
    # before the grouped MM (see patches/quantization/quantize.py).
    return torch._grouped_mm(x, weight, group_list)


def _run_experts_grouped_mm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    # pyrefly: ignore [missing-attribute]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)

    h = npu_grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets)
    if swiglu_limit is not None:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)
    out = npu_grouped_mm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)

    return out


def npu_grouped_experts_forward(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    is_tp = False
    if isinstance(self.w2, DTensor):
        w2 = self.w2.to_local()
        w13 = self.w13.to_local() if self.w13 is not None else None
        from torch.distributed.tensor.placement_types import Shard as _Shard

        for p in self.w2.placements:
            if isinstance(p, _Shard) and p.dim == 2:
                is_tp = True
                break
        tp_group = self.w2.device_mesh.get_group() if is_tp else None
        logged_attr = "_logged"
        if not hasattr(npu_grouped_experts_forward, logged_attr):
            setattr(npu_grouped_experts_forward, logged_attr, True)
            logger.info(
                f"[GMM-TP] w2 placements={self.w2.placements}, is_tp={is_tp}, "
                f"w2 local shape={w2.shape}, w13 local shape={w13.shape if w13 is not None else None}"
            )
    else:
        w2 = self.w2
        w13 = self.w13
        tp_group = None

    # Refactor this, only DSv4 inject this attribute to its experts.
    swiglu_limit = getattr(self, "swiglu_limit", None)

    # pyrefly: ignore [bad-argument-type]
    out = _run_experts_grouped_mm(w13, w2, None, x, num_tokens_per_expert, swiglu_limit)

    if is_tp and tp_group is not None:
        import torch.distributed as dist

        pre_ar = out.mean().item()
        dist.all_reduce(out, group=tp_group)
        post_ar = out.mean().item()
        ar_logged_attr = "_ar_logged"
        if not hasattr(npu_grouped_experts_forward, ar_logged_attr):
            setattr(npu_grouped_experts_forward, ar_logged_attr, True)
            ratio = post_ar / pre_ar if pre_ar != 0 else float("inf")
            logger.info(
                "[GMM-TP] all-reduce: pre_mean=%.6f, post_mean=%.6f, ratio=%s",
                pre_ar,
                post_ar,
                ratio,
            )

    return out


def npu_grouped_experts_init_weights(self, init_std: float):
    for w in [self.w2, self.w13]:
        if w is not None:
            nn.init.normal_(w, mean=0.0, std=init_std)


class NpuGroupedExperts(GroupedExperts):
    def __init__(
        self,
        parent: GroupedExperts,
    ):
        self.__dict__.update(parent.__dict__)
        self.use_grouped_mm = True
        if self.w1 is not None and self.w3 is not None:
            # pyrefly: ignore [no-matching-overload]
            w13_data = torch.empty(
                self.num_experts,
                self.w2.shape[2] * 2,
                self.w2.shape[1],
                dtype=self.w1.dtype,
                device=self.w1.device,
            )
            self.w13 = nn.Parameter(w13_data)
            # Add w13 initializer to _param_init if it exists (new torchtitan config system)
            _param_init = getattr(parent, "_param_init", None)
            if _param_init is not None:
                # Use w1's initializer for w13 (combined w1+w3)
                w1_init = _param_init.get("w1")
                if w1_init is not None:
                    _param_init["w13"] = w1_init

            # pyrefly: ignore [bad-assignment]
            self.w1 = None
            # pyrefly: ignore [bad-assignment]
            self.w3 = None
            # pyrefly: ignore [bad-assignment]
            parent.w1 = None
            # pyrefly: ignore [bad-assignment]
            parent.w3 = None
            logger.info(f"  NpuGroupedExperts: Created w13 [{w13_data.shape}]")

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return npu_grouped_experts_forward(self, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        npu_grouped_experts_init_weights(self, init_std)


class NpuGroupedExpertConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, GroupedExperts):
                continue
            replace_module_with_name(model, name, NpuGroupedExperts(module))


class GMMStateDictUpdater(StateDictUpdater):
    @classmethod
    def to_hf(cls, state_dict):
        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            state_dict = _split_w13_for_mapping(state_dict)
        return state_dict

    @classmethod
    def from_hf(cls, state_dict):
        filtered = {
            k: v for k, v in state_dict.items() if not k.endswith(".weight_scale_inv")
        }

        return fuse_experts(filtered)


@register_model_converter("npu_gmm")
class GMMModelConfig(ModelCustomConfig):
    model_converter = NpuGroupedExpertConverter
    state_dict_updater = GMMStateDictUpdater
