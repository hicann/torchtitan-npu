# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu

from torch import Tensor
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

from .permutation import NPUMoeTokenUnpermute

logger = logging.getLogger(__name__)


def _is_fake_process_group(group) -> bool:
    if not dist.is_initialized():
        return False
    try:
        return str(dist.get_backend(group)).lower() == "fake"
    except RuntimeError:
        return False


def _npu_moe_forward_for_dsv32(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    total_tokens = x.shape[0]

    (top_scores, selected_experts_indices, num_tokens_per_expert) = self.router(
        x, self.expert_bias
    )

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_scores, _ = torch_npu.npu_moe_token_permute(
        top_scores.reshape(-1).unsqueeze(-1), indices.reshape(-1, 1)
    )

    routed_output = self.experts(routed_input, num_tokens_per_expert, routed_scores)

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        None,
    )
    unpermuted = unpermuted.view(total_tokens, self.reorderer.top_k, dim).sum(dim=1)
    return (out + unpermuted).reshape(bs, slen, dim)


def _npu_moe_forward_for_dsv4(self, x, input_ids):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    total_tokens = x.shape[0]
    input_ids_flat = input_ids.flatten() if input_ids is not None else None

    (top_scores, selected_experts_indices, num_tokens_per_expert) = self.router(
        x, input_ids_flat, self.expert_bias
    )

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_scores, _ = torch_npu.npu_moe_token_permute(
        top_scores.reshape(-1).unsqueeze(-1), indices.reshape(-1, 1)
    )

    routed_output = self.experts(routed_input, num_tokens_per_expert, routed_scores)

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)
    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        None,
    )
    unpermuted = unpermuted.view(total_tokens, self.reorderer.top_k, dim).sum(dim=1)
    return (out + unpermuted).reshape(bs, slen, dim)


def _npu_moe_token_dispatch(
    self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
) -> tuple[Tensor, Tensor, Tensor]:
    routed_input, num_tokens_per_expert, routed_scores = inputs
    ep_degree = device_mesh.shape[0]
    num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

    ep_group = device_mesh.get_group()
    fake_pg = _is_fake_process_group(ep_group)

    # generate the input splits and output splits for all-to-all
    with torch.no_grad():
        if fake_pg:
            # FakeProcessGroup's all_to_all returns tensors with valid metadata
            # but arbitrary values, which would corrupt the split sizes derived
            # from the exchanged counts. Under fake PG no data actually crosses
            # ranks, so output == input.
            num_tokens_per_expert_group = num_tokens_per_expert
        else:
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=ep_group,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
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

    if not fake_pg:
        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

        routed_scores = all_to_all_single_autograd(
            routed_scores, self.output_splits, self.input_splits, ep_group
        )

    # NOTE: After this all-to-all, the routed input is put on proper EP rank.
    # However, the num_tokens_per_expert_group is not of the final target format
    # [#tokens for local expert 0, #tokens for local expert 1, ...]
    # Rather, it is of the format
    # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
    #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
    # We need to perform another shuffle to get the correct layout, via the _permute function
    # below, which also does padding to make sure the number of tokens each expert gets locally
    # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
    # Note that this will create side effects when wrapping the for-loop implementation
    # of GroupedExperts, as it does not need padding.
    indices = None
    with torch.no_grad():
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

    routed_scores, _ = torch_npu.npu_moe_token_permute(routed_scores, indices)

    num_tokens_per_expert_group = num_tokens_per_expert_group.view(ep_degree, -1).sum(0)
    return routed_input, num_tokens_per_expert_group, routed_scores


def _npu_moe_token_combine(
    self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
) -> Tensor:
    # Using NPUMoeTokenUnpermute.apply and npu_moe_token_unpermute is equivalent here,
    # and avoid storing tensor routed_output during backpropagation.
    routed_output = NPUMoeTokenUnpermute.apply(
        routed_output, self.permuted_indices, routed_output.shape
    )
    if _is_fake_process_group(device_mesh.get_group()):
        return routed_output
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,
        self.output_splits,
        device_mesh.get_group(),
    )
    return routed_output


@register_npu_converter("npu_permute")
class PermuteKernel(BaseConverter):
    """Replace `MoE.forward` with the NPU permute/unpermute implementation,
    and when expert parallel is enabled, replace
    `ExpertParallel._token_dispatch` / `_token_combine` with the matching
    NPU implementations.
    """

    DIST_PACKAGE = "torchtitan.distributed"

    # pyrefly: ignore [bad-assignment]
    MODEL_IMPL = {
        "deepseek_v3": ("torchtitan.models.moe", _npu_moe_forward_for_dsv32),
        "deepseek_v4": (
            "torchtitan_npu.models.deepseek_v4.model.moe",
            _npu_moe_forward_for_dsv4,
        ),
    }

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        count = 0

        impl = cls.get_impl_cls(model_name)
        if impl:
            # pyrefly: ignore [not-iterable]
            module_path, replace_func = impl
            count += replace_methods(
                class_name="MoE",
                method_name="forward",
                new_method=replace_func,
                package=module_path,
            )

        count += replace_methods(
            class_name="ExpertParallel",
            method_name="_token_dispatch",
            new_method=_npu_moe_token_dispatch,
            package=cls.DIST_PACKAGE,
        )
        count += replace_methods(
            class_name="ExpertParallel",
            method_name="_token_combine",
            new_method=_npu_moe_token_combine,
            package=cls.DIST_PACKAGE,
        )

        return count
