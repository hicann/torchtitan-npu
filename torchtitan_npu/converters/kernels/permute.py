# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Partial
from torchtitan.distributed.expert_parallel import ExpertParallel

from torchtitan.models.common.moe import MoE

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.permutation import NPUMoeTokenUnpermute
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
    ParallelizePlanUpdater,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.distributed.process_group import is_fake_process_group

logger = logging.getLogger(__name__)


def _npu_moe_forward(self, x):
    if isinstance(x, DTensor):
        x = x.to_local(grad_placements=(Partial(),))

    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    # Bypass self.router() entirely.  NoParallel / TP hooks on the gate
    # module convert inputs to DTensors, which then crash when mixed with
    # plain-Tensor expert_bias or NPU kernels.  Instead, compute gate
    # scores directly using local tensors.
    gate = self.router.gate
    gate_weight = gate.weight
    gate_bias = getattr(gate, "bias", None)
    if isinstance(gate_weight, DTensor):
        gate_weight = gate_weight.to_local()
    if gate_bias is not None and isinstance(gate_bias, DTensor):
        gate_bias = gate_bias.to_local()

    with torch.autocast(device_type=x.device.type, dtype=torch.float32):
        scores = torch.nn.functional.linear(x, gate_weight, gate_bias)

    score_func = self.router.score_func
    if score_func == "sigmoid":
        scores = torch.sigmoid(scores)
    elif score_func == "softmax":
        scores = torch.nn.functional.softmax(scores, dim=1)
    else:
        raise NotImplementedError(f"Unknown score function {score_func}")

    expert_bias = self.expert_bias
    scores_for_choice = scores if expert_bias is None else scores + expert_bias

    if self.router.num_expert_groups is not None:
        num_expert_groups = self.router.num_expert_groups
        num_limited_groups = self.router.num_limited_groups
        num_experts = self.router.num_experts
        experts_per_group = num_experts // num_expert_groups
        scores_grouped = scores_for_choice.view(
            -1, num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, num_experts)

    _, selected_experts_indices = torch.topk(
        scores_for_choice, k=self.router.top_k, dim=-1, sorted=False
    )
    top_scores = scores.gather(dim=1, index=selected_experts_indices)

    if self.router.route_norm:
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator
    top_scores = top_scores * self.router.route_scale

    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1),
        bins=self.router.num_experts,
        min=0,
        max=self.router.num_experts,
    )

    if self.shared_experts is not None:
        out = self.shared_experts(x)
        if isinstance(out, DTensor):
            out = out.to_local()
    else:
        out = torch.zeros_like(x)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        top_scores.to(x.dtype),
    )
    return (out + unpermuted).reshape(bs, slen, dim)


class NpuMoE(MoE):
    def __init__(
        self,
        parent: MoE,
    ):
        # Shallow copy of parent's __dict__ is intentional here:
        # - MoE attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on MoE.__init__ parameters (moe_args, dim, hidden_dim)
        # - Parent instance already has all attributes properly initialized
        # Note: If MoE had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(self, x):
        return _npu_moe_forward(self, x)


class NpuPermuteConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, MoE):
                continue
            replace_module_with_name(model, name, NpuMoE(module))


class NpuExpertParallel(ExpertParallel):
    def _compute_all_to_all_splits(
        self,
        num_tokens_per_expert: torch.Tensor,
        ep_degree: int,
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, list[int]]:
        # Generate the input/output splits for all-to-all and stash them on
        # self. Returns (num_tokens_per_expert_group, output_splits) for the
        # downstream local shuffle.
        group = device_mesh.get_group()
        is_fake = is_fake_process_group(group)
        with torch.no_grad():
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=not is_fake)
            )
            if is_fake:
                num_tokens_per_expert_group = num_tokens_per_expert
                output_splits = input_splits
            else:
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=group,
                )
                # NOTE: this would incur a device-to-host sync
                output_splits = (
                    num_tokens_per_expert_group.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()
        return num_tokens_per_expert_group, self.output_splits

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        num_tokens_per_expert_group, output_splits = self._compute_all_to_all_splits(
            num_tokens_per_expert, ep_degree, device_mesh
        )

        if not is_fake_process_group(device_mesh.get_group()):
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
                output_size=sum(output_splits),
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
        self, mod: nn.Module, routed_output: torch.Tensor, device_mesh: DeviceMesh
    ) -> torch.Tensor:
        # Using NPUMoeTokenUnpermute.apply and npu_moe_token_unpermute is equivalent here,
        # and avoid storing tensor routed_output during backpropagation.
        routed_output = NPUMoeTokenUnpermute.apply(
            routed_output, self.permuted_indices, routed_output.shape
        )
        if is_fake_process_group(device_mesh.get_group()):
            return routed_output

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output


class NpuParallelizePlanUpdater(ParallelizePlanUpdater):
    @classmethod
    def update(
        cls, parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None
    ) -> ParallelStyle | dict[str, ParallelStyle] | None:
        """Update the layer plan"""
        if isinstance(parallelize_plan, ExpertParallel):
            return NpuExpertParallel()
        return parallelize_plan


@register_model_converter("npu_permute")
class PermuteModelConfig(ModelCustomConfig):
    model_converter = NpuPermuteConverter
    parallelize_plan_updater = NpuParallelizePlanUpdater
