# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLossLoggingHelper
from torchtitan_npu.ops.aclnn.builder import build_op
from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)

TORCH_MAX_INT = 9223372036854775807

# Will be compiled lazily, only when converter hits.
_li_op, _kl_op, _sas_op = None, None, None


class SparseAttnSharedKV(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    ):
        ori_kv_stride = ori_kv.stride(0) if ori_kv is not None else 0
        cmp_kv_stride = cmp_kv.stride(0) if cmp_kv is not None else 0
        # pyrefly: ignore [missing-attribute]
        metadata = _sas_op.npu_sparse_attn_sharedkv_metadata(
            # pyrefly: ignore [missing-attribute]
            cu_seq_lens_q if cu_seq_lens_q is not None else torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            # pyrefly: ignore [missing-attribute]
            torch.tensor([]).npu(),
            num_heads_q,
            num_heads_kv,
            head_dim,
            batch_size,
            max_seq_len_q,
            max_seq_len_kv,
            0,  # oriTopk
            topk,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            ori_kv is not None,  # hasOriKv
            cmp_kv is not None,  # hasCmpKv
        )
        # pyrefly: ignore [missing-attribute]
        result, softmax_lse = _sas_op.npu_sparse_attn_sharedkv(
            query,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            None,  # oriBlockTable
            None,  # cmpBlockTable
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            None,  # sequsedQ
            None,  # sequsedKv
            sinks,
            metadata,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_kv_stride,
            cmp_kv_stride,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            True,  # returnSoftmaxLse
        )
        ctx.save_for_backward(
            query,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            sinks,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        return result

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        (
            query,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            sinks,
        ) = ctx.saved_tensors
        query_grad, ori_kv_grad, cmp_kv_grad, sinks_grad = (
            # pyrefly: ignore [missing-attribute]
            _sas_op.npu_sparse_attn_sharedkv_grad(
                query,
                ori_kv,
                cmp_kv,
                grad_output,
                result,
                softmax_lse,
                ori_sparse_indices,
                cmp_sparse_indices,
                None,  # cuSeqlensQ
                None,  # cuSeqlensOriKv
                None,  # cuSeqlensCmpKv
                sinks,
                ctx.softmax_scale,
                ctx.cmp_ratio,
                ctx.ori_mask_mode,
                ctx.cmp_mask_mode,
                ctx.ori_win_left,
                ctx.ori_win_right,
                ctx.layout_q,
            )
        )
        return (
            query_grad,
            ori_kv_grad,
            cmp_kv_grad,
            None,
            None,
            None,
            None,
            None,
            sinks_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def npu_sparse_attn_shared_kv(
    query,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    sinks,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
):
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # not support TND
    ori_sparse_indices = None  # ori kv use band mode
    batch_size, max_seq_len_q, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(1)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = "BSND"
    query = query.contiguous()  # [S, B, N, D] --> [B, S, N, D]
    ori_kv = ori_kv.unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_kv = (
        cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()
    )  # [S, B, D] --> [B, S, 1, D]
    if cmp_ratio != 4:
        cmp_sparse_indices = None
    else:
        cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    )
    return output.contiguous()


def sdpa_to_sfa_adapter(
    self, query_states, kv_states, attn_sink, kv_compress, compress_topk_idxs
):

    if compress_topk_idxs is not None:
        if compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)

    output = npu_sparse_attn_shared_kv(
        query=query_states,
        ori_kv=kv_states,
        cmp_kv=kv_compress,
        cmp_sparse_indices=compress_topk_idxs,
        sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
    )
    return output


def sdpa_to_li_adapter(
    self,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    seqlen: int,
    offset: int,
):

    q_indexer = q_indexer.to(torch.bfloat16)
    k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
    weights = weights.to(torch.bfloat16)

    # pyrefly: ignore [missing-attribute]
    compress_topk_idxs, index_score = _li_op.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        None,  # actual_seq_q
        None,  # actual_seq_k
        None,  # block_table
        "BSND",  # layout_q
        "BSND",  # layout_k
        self.index_topk,
        3,  # sparse_mode
        TORCH_MAX_INT,  # pre_tokens
        TORCH_MAX_INT,  # next_tokens
        self.ratio,
        True,  # return_values
    )

    compress_topk_idxs = compress_topk_idxs.squeeze(2)
    index_score = index_score.squeeze(2)
    if offset != 0:
        # pyrefly: ignore [no-matching-overload]
        compress_topk_idxs = torch.where(
            compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs + offset
        )

    return compress_topk_idxs, index_score


class SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        layer_number=None,
        num_layers=0,
    ):
        ctx.save_for_backward(
            query, key, query_index, key_index, weights, sparse_indices
        )
        ctx.scale_value = scale_value
        ctx.cmp_ratio = cmp_ratio
        ctx.layer_number = layer_number
        ctx.num_layers = num_layers
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_klen = actual_seq_klen
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens

        # Return dummy loss during fwd, real operation will be postponed
        # to bwd, to avoid redundant computation of the loss function in
        # case where activation checkpointing is enabled.
        return torch.zeros(1, dtype=torch.float32, device=query.device)[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        query, key, query_index, key_index, weights, sparse_indices = ctx.saved_tensors

        (
            d_query_index,
            d_key_index,
            d_weights,
            loss,
            # pyrefly: ignore [missing-attribute]
        ) = _kl_op.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            None,  # softmax_max
            None,  # softmax_sum
            None,  # query_rope
            None,  # key_rope
            ctx.actual_seq_qlen,
            ctx.actual_seq_klen,
            ctx.layout,
            ctx.sparse_mode,
            ctx.pre_tokens,
            ctx.next_tokens,
            ctx.cmp_ratio,
            ctx.scale_value,
            False,  # deterministic
        )

        bsz, slen, *_ = query.shape
        token_scale = 1 / (bsz * slen)
        grad_scale = grad * token_scale

        d_query_index = d_query_index * grad_scale
        d_key_index = d_key_index * grad_scale
        d_weights = d_weights * grad_scale
        loss = loss * token_scale

        if ctx.layer_number is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss[0], ctx.layer_number, ctx.num_layers
            )
        return None, None, d_query_index, d_key_index, d_weights, *([None] * 11)


# Wrapper for autograd.Function to support default/keyword argument
def npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    *,
    scale_value,
    cmp_ratio,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    layer_number=None,
    num_layers=0,
):
    return SparseLightningIndexerGradKLLossWrapper.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        layer_number,
        num_layers,
    )


def li_loss_adapter(
    self,
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    indexer_score,
    attention_masks,
    offset,
):
    return npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key.unsqueeze(2),
        query_index,
        key_index.unsqueeze(2),
        weights,
        sparse_indices.unsqueeze(2),
        scale_value=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
        layer_number=self.layer_id,
        num_layers=self.n_layers,
    )


@register_npu_converter("deepseek_v4_sfa")
class DeepSeekV4SFAKernel(BaseConverter):

    MODEL_PACKAGE = "torchtitan_npu.models.deepseek_v4"
    SUPPORTED_MODELS = {"deepseek_v4"}

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        pkg = cls.MODEL_PACKAGE
        total = 0

        count = replace_methods(
            "SparseAttention", "forward", sdpa_to_sfa_adapter, package=pkg
        )
        logger.info(f"  [SparseAttention forward] Applied {count} replacement(s)")
        total += count

        count = replace_methods("LiCompute", "forward", sdpa_to_li_adapter, package=pkg)
        logger.info(f"  [LiCompute forward] Applied {count} replacement(s)")
        total += count

        count = replace_methods("LiLoss", "forward", li_loss_adapter, package=pkg)
        logger.info(f"  [LiLoss forward] Applied {count} replacement(s)")
        total += count

        if total != 0:
            global _li_op, _kl_op, _sas_op

            _li_op = build_op("lightning_indexer", ["lightning_indexer/binding.cpp"])
            _kl_op = build_op(
                "sparse_lightning_indexer_grad_kl_loss",
                ["sparse_lightning_indexer_grad_kl_loss/binding.cpp"],
            )
            _sas_op = build_op(
                "sparse_attn_sharedkv", ["sparse_attn_sharedkv/binding.cpp"]
            )

        # pyrefly: ignore [bad-return]
        return total
