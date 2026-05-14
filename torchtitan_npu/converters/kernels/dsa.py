# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

import torch_npu

from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLossLoggingHelper

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter


logger = logging.getLogger(__name__)


class SparseLightningIndexerKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        query_indexer,
        key_indexer,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        *,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
        layer_number=None,
        num_layers=0,
    ):
        """NPU Sparse Lightning Indexer KL Divergence Loss Function"""
        bsz = query.shape[0]
        sq = query.shape[1]
        loss = LILossTrain.apply(
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            topk_indices,
            softmax_max,
            softmax_sum,
            layer_number,
            num_layers,
            scale_value,
            query_rope,
            key_rope,
            actual_seq_qlen,
            actual_seq_klen,
            layout,
            sparse_mode,
            pre_tokens,
            next_tokens,
        )
        return loss / (bsz * sq)


class LILossTrain(torch.autograd.Function):
    """
    A custom autograd function that computes kl loss in sparse lightning indexer.

    This interface implements the backward functionality of npu_lightning_indexer and integrates the loss computation.
    The npu_lightning_indexer selects the top-k pairs between queries and keys in attention that exhibit the strongest
    intrinsic correlations, storing them in sparse_indices. This reduces the computational cost of attention in
    long-sequence scenarios and improves training performance.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        key,
        query_indexer,
        key_indexer,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        layer_number=None,
        num_layers=0,
        scale_value=1,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    ):
        ctx.save_for_backward(
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
        )
        ctx.layer_number = layer_number
        ctx.num_layers = num_layers
        ctx.scale_value = scale_value
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_klen = actual_seq_klen

        # Return dummy loss during fwd, real operation is postponed to bwd
        # to avoid redundant computation when activation checkpointing is enabled.
        return torch.zeros(1, dtype=torch.float32, device=query.device)[0]

    @staticmethod
    def backward(ctx, *grad_output) -> tuple:
        (
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
        ) = ctx.saved_tensors

        (
            d_query_index,
            d_key_index,
            d_weights,
            loss,
        ) = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=ctx.scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_klen=ctx.actual_seq_klen,
            layout=ctx.layout,
            sparse_mode=ctx.sparse_mode,
            pre_tokens=ctx.pre_tokens,
            next_tokens=ctx.next_tokens,
        )
        if grad_output[0] != 1.0:
            d_query_index = d_query_index * grad_output[0]
            d_key_index = d_key_index * grad_output[0]
            d_weights = d_weights * grad_output[0]
        bsz, sq = query.shape[0], query.shape[1]
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss[0] / (bsz * sq), ctx.layer_number, ctx.num_layers
        )
        backward_grads = (
            None,
            None,
            d_query_index,
            d_key_index,
            d_weights,
            *([None] * 14),
        )
        return backward_grads


def dsa_forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    q_indexer: torch.Tensor | None = None,
    k_indexer: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    end_pos: torch.Tensor | None = None,
    index_topk: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward pass of the dsa module.
    """
    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError(
            "Only support num_head_kv == 1 in dsa forward under absorb mode."
        )

    # Fuse LILossTrain includes LIG
    # NOTE: set return_value=False to avoid torch.compile / DTensor meta path failure
    # ("when return_value is true, not support pytorch compile").
    ret = torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=index_topk,
        sparse_mode=3,
        return_value=False,
    )
    topk_indices = ret[0] if isinstance(ret, tuple) else ret

    # To BSND
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Split q_nope / q_pe
    q_nope, q_pe = torch.split(
        q, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1
    )
    k_nope, k_pe = torch.split(
        k, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1
    )

    bsz = q.shape[0]
    actual_seq_len = torch.full(
        (bsz,), q_nope.shape[1], dtype=torch.int32, device=q_nope.device
    )

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        q_nope,
        k_nope,
        v,
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len,
        query_rope=q_pe,
        key_rope=k_pe,
        scale_value=scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,  # 0: GQA/MHA, 1: MLA-naive, 2: MLA-absorb
        return_softmax_lse=True,  # it must be True in training mode
    )

    # The loss is actually computed by SparseLightningIndexerKLLoss.forward
    # If tp is enabled, inner_attention.compute_dsa_indexer_loss is patched in deepseek_v32_parallelize.py
    # Otherwise, inner_attention.compute_dsa_indexer_loss is patched in this file
    loss = self.compute_dsa_indexer_loss(
        q_nope,
        k_nope,
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=scale,
        query_rope=q_pe,
        key_rope=k_pe,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        layer_number=self.layer_number,
        num_layers=self.num_layers,
    )
    output = output.transpose(1, 2)
    # pyrefly: ignore [bad-return]
    return loss, output


@register_npu_converter("npu_dsa")
class DSAKernel(BaseConverter):

    MODEL_PACKAGE = "torchtitan_npu.models.deepseek_v32"
    SUPPORTED_MODELS = {"deepseek_v32"}
    SUPPORTED_ACTIVATION_CHECKPOINT = {"none", "full"}

    @classmethod
    def is_compatible(cls, job_config, model_name):
        mode = job_config.activation_checkpoint.mode
        if mode not in cls.SUPPORTED_ACTIVATION_CHECKPOINT:
            raise ValueError(
                f"Patch `npu_dsa` is NOT compatible with activation checkpoint mode '{mode}'\n"
                f"Supported activation checkpoint mode: {cls.SUPPORTED_ACTIVATION_CHECKPOINT}"
            )
        return super().is_compatible(job_config, model_name)

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        count = replace_methods(
            "DSV32_SDPA", "forward", dsa_forward, package=cls.MODEL_PACKAGE
        )
        logger.info(f"  [DSV32_SDPA forward] Applied {count} replacement(s)")
        logger.info(
            "  Only matrix absorb mode is supported, and LI Loss is enabled by default."
        )

        # If tp is not enabled, then the indexer_loss patch in deepseek_v32/parallelize.py won't be applied
        # The patch is applied here as a supplement
        # pyrefly: ignore [not-callable, bad-argument-type]
        num_layers = len(model.layers)
        # pyrefly: ignore [missing-attribute]
        for layer_id, transformer_block in model.layers.named_children():
            inner_attention = transformer_block.attention.inner_attention
            if not isinstance(
                inner_attention.compute_dsa_indexer_loss, SparseLightningIndexerKLLoss
            ):
                inner_attention.compute_dsa_indexer_loss = (
                    SparseLightningIndexerKLLoss()
                )
            inner_attention.layer_number = int(layer_id)
            inner_attention.num_layers = num_layers
        return count
