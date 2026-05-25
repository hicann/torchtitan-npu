# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.npu_registry import register_model_converter
from torchtitan_npu.models.deepseek_v4.model import LiCompute, LiLoss, SparseAttention
from torchtitan_npu.ops.aclnn.builder import build_op

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
        (
            query_grad,
            ori_kv_grad,
            cmp_kv_grad,
            sinks_grad,
            # pyrefly: ignore [missing-attribute]
        ) = _sas_op.npu_sparse_attn_sharedkv_grad(
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


def _c128a_cp_sfa_with_global_positions(
    self,
    query_states,
    kv_states,
    attn_sink,
    kv_compress,
):
    """Run C128A CP rank1 through SFA by restoring global token positions.


    The SFA kernel computes each query's effective position as (S2 - S1) + local_i,
    so padding only kv to length global_start * cp is sufficient: with S1=chunk_size
    and S2=global_seq_len, S2-S1=global_start shifts the window automatically.
    """
    cp_rank = getattr(self, "cp_rank", 0)
    n_boundary = self.window_size - 1
    bsz, seq_len, _n_heads, _head_dim = query_states.shape
    global_start = cp_rank * seq_len

    kv_prefix_len = global_start - n_boundary
    kv_states = kv_states.contiguous()
    kv_prefix = torch.zeros(
        bsz,
        kv_prefix_len,
        kv_states.size(-1),
        dtype=kv_states.dtype,
        device=kv_states.device,
    )
    kv_padded = torch.cat([kv_prefix, kv_states], dim=1)

    return npu_sparse_attn_shared_kv(
        query=query_states,
        ori_kv=kv_padded,
        cmp_kv=kv_compress,
        cmp_sparse_indices=None,
        sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
    )


class NpuSparseAttention(SparseAttention):
    def __init__(self, parent: SparseAttention) -> None:
        # Shallow copy of parent's __dict__ is intentional here:
        # - SparseAttention attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on SparseAttention.__init__ parameters (layer_id, window_size, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If SparseAttention had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attn_sink: torch.Tensor,
        kv_compress: torch.Tensor | None = None,
        compress_topk_idxs: torch.Tensor | None = None,
    ):
        if compress_topk_idxs is not None:
            if compress_topk_idxs.dtype != torch.int32:
                compress_topk_idxs = compress_topk_idxs.to(torch.int32)

        cp_rank = getattr(self, "cp_rank", 0)
        if cp_rank > 0:
            n_boundary = self.window_size - 1  # 127
            if self.compress_ratio == 128:
                # Restore C128A tensor positions to global token coordinates.
                return _c128a_cp_sfa_with_global_positions(
                    self, query_states, kv_states, attn_sink, kv_compress
                )

            if self.compress_ratio == 1:
                # kv_states layout for rank>0: [boundary_w-1 || local_chunk], so
                # SFA band mode naturally maps query i to kv_states[i:i + window_size].
                return npu_sparse_attn_shared_kv(
                    query=query_states,
                    ori_kv=kv_states,
                    cmp_kv=None,
                    cmp_sparse_indices=None,
                    sinks=attn_sink.float(),
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_win_left=n_boundary,
                )

        # Rank0/non-CP uses the kernel's native causal positions.
        output = npu_sparse_attn_shared_kv(
            query=query_states,
            ori_kv=kv_states,
            cmp_kv=kv_compress,
            cmp_sparse_indices=compress_topk_idxs if self.compress_ratio == 4 else None,
            sinks=attn_sink.float(),
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
        )
        return output


class NpuLiCompute(LiCompute):
    def __init__(self, parent: LiCompute) -> None:
        # Shallow copy of parent's __dict__ is intentional here:
        # - LiCompute attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on LiCompute.__init__ parameters (ratio, index_topk)
        # - Parent instance already has all attributes properly initialized
        # Note: If LiCompute had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
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
                compress_topk_idxs == -1,
                compress_topk_idxs,
                compress_topk_idxs + offset,
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
        loss_tracker,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    ):
        ctx.save_for_backward(
            query, key, query_index, key_index, weights, sparse_indices
        )
        ctx.loss_tracker = loss_tracker
        ctx.scale_value = scale_value
        ctx.cmp_ratio = cmp_ratio
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
        loss_scale = ctx.scale_value
        grad_scale = grad * token_scale * loss_scale

        d_query_index = d_query_index * grad_scale
        d_key_index = d_key_index * grad_scale
        d_weights = d_weights * grad_scale
        loss = loss * token_scale * loss_scale

        ctx.loss_tracker(loss[0])
        return None, None, d_query_index, d_key_index, d_weights, *([None] * 10)


# Wrapper for autograd.Function to support default/keyword argument
def npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    *,
    loss_tracker,
    scale_value,
    cmp_ratio,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
):
    return SparseLightningIndexerGradKLLossWrapper.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        loss_tracker,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )


class NpuLiLoss(LiLoss):
    def __init__(self, parent: LiLoss):
        # Shallow copy of parent's __dict__ is intentional here:
        # - LiLoss attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on LiLoss.__init__ parameters (n_heads, softmax_scale, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If LiLoss had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        q,
        k,
        q_indexer,
        k_indexer,
        weights,
        sparse_indices,
        indexer_score,
        attention_masks,
        offset,
    ):
        if sparse_indices.dtype != torch.int32:
            sparse_indices = sparse_indices.to(torch.int32)

        return npu_sparse_lightning_indexer_grad_kl_loss(
            q,
            k.unsqueeze(2),
            q_indexer,
            k_indexer.unsqueeze(2),
            weights,
            sparse_indices.unsqueeze(2),
            loss_tracker=self.save_loss,
            scale_value=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
        )


class DeepSeekV4SFAConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        global _li_op, _kl_op, _sas_op

        for name, module in model.named_modules():
            if isinstance(module, SparseAttention):
                _sas_op = build_op(
                    "sparse_attn_sharedkv", ["sparse_attn_sharedkv/binding.cpp"]
                )
                replace_module_with_name(model, name, NpuSparseAttention(module))
                logger.info(
                    "[DeepSeekV4SFAConverter] [SparseAttention forward] Applied."
                )

            if isinstance(module, LiCompute):
                _li_op = build_op(
                    "lightning_indexer", ["lightning_indexer/binding.cpp"]
                )
                replace_module_with_name(model, name, NpuLiCompute(module))
                logger.info("[DeepSeekV4SFAConverter] [LiCompute forward] Applied.")

            if isinstance(module, LiLoss):
                _kl_op = build_op(
                    "sparse_lightning_indexer_grad_kl_loss",
                    ["sparse_lightning_indexer_grad_kl_loss/binding.cpp"],
                )
                replace_module_with_name(model, name, NpuLiLoss(module))
                logger.info("[DeepSeekV4SFAConverter] [LiLoss forward] Applied.")


@register_model_converter("deepseek_v4_sfa")
class DeepSeekV4SFAModelConfig(ModelCustomConfig):
    model_converter = DeepSeekV4SFAConverter
