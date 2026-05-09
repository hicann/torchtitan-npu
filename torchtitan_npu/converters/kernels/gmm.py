# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from torch import nn, Tensor
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe.moe import GroupedExperts

from torchtitan_npu.converters.kernels.permutation import NPUMoeTokenUnpermute
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
    ParallelizePlanUpdater,
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
        # Convert parameters from DTensors to plain Tensors, to work with
        # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
        is_dtensor = isinstance(self.w2, DTensor)
        # pyrefly: ignore [missing-attribute]
        w2 = self.w2.to_local() if is_dtensor else self.w2
        # pyrefly: ignore [missing-attribute]
        w13 = self.w13.to_local() if is_dtensor and self.w13 is not None else self.w13

        # NOTE: If EP is not used, we need to pad the indices
        #       to prepare for grouped_mm;
        #       otherwise, EP will handle the padding.
        if (
            not is_dtensor
            # pyrefly: ignore [not-iterable]
            or "ep"
            not in self.w2.device_mesh.mesh_dim_names  # pyrefly: ignore [missing-attribute]
        ):
            group_size_params["expert_model_parallel_size"] = 1
        else:
            # pyrefly: ignore [missing-attribute]
            ep_dim_index = self.w2.device_mesh.mesh_dim_names.index("ep")
            # pyrefly: ignore [missing-attribute]
            group_size_params["expert_model_parallel_size"] = self.w2.device_mesh.shape[
                ep_dim_index
            ]

        if group_size_params["g_size"] is None:
            group_size_params["num_experts"] = self.num_experts
            group_size_params["g_size"] = (
                # pyrefly: ignore [unsupported-operation]
                group_size_params["num_experts"]
                // group_size_params["expert_model_parallel_size"]
            )

        # Refactor this, only DSv4 inject this attribute to its experts.
        swiglu_limit = getattr(self, "swiglu_limit", None)

        # pyrefly: ignore [bad-argument-type]
        return _run_experts_grouped_mm(w13, w2, x, num_tokens_per_expert, swiglu_limit)

    def init_weights(self, init_std: float):
        for w in [self.w2, self.w13]:
            if w is not None:
                nn.init.normal_(w, mean=0.0, std=init_std)


class NpuGroupedExpertConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, GroupedExperts):
                continue
            splits = name.split(".")
            # parent module name
            parent_module_name = ".".join(splits[:-1])
            module_name = splits[-1]
            parent_module = model
            if parent_module_name:
                parent_module = model.get_submodule(parent_module_name)
            setattr(parent_module, module_name, NpuGroupedExperts(module))


class GMMExpertParallel(ExpertParallel):
    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout
        indices = (
            torch.arange(
                num_local_experts,
                dtype=torch.int64,
                device=routed_input.device,
            )
            .repeat(ep_degree)
            .repeat_interleave(
                num_tokens_per_expert_group.view(-1),
                output_size=sum(self.output_splits),
            )
        )

        routed_input, self.permuted_indices = torch_npu.npu_moe_token_permute(
            routed_input, indices
        )

        num_tokens_per_expert_group = num_tokens_per_expert_group.view(
            ep_degree, -1
        ).sum(0)

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        # Using NPUMoeTokenUnpermute.apply and npu_moe_token_unpermute is equivalent here,
        # and avoid storing tensor routed_output during backpropagation.
        routed_output = NPUMoeTokenUnpermute.apply(
            routed_output, self.permuted_indices, routed_output.shape
        )
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output


class GMMParallelizePlanUpdater(ParallelizePlanUpdater):
    @classmethod
    def update(
        cls, parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None
    ) -> ParallelStyle | dict[str, ParallelStyle] | None:
        """Update the layer plan"""
        if isinstance(parallelize_plan, ExpertParallel):
            return GMMExpertParallel()
        return parallelize_plan


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
    parallelize_plan_updater = GMMParallelizePlanUpdater
    state_dict_updater = GMMStateDictUpdater
