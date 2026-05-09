# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Context Parallel implementation for DeepSeek-V4.

Handles two attention layer types:
  - ratio=1  (Window Attention / C1A): P2P BoundaryExchange only
  - ratio=128 (C128A): BoundaryExchange + AllGather (overlap=False compressor)

Public entry point: patch_deepseek_v4_for_context_parallel(model, cp_mesh)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup

logger = logging.getLogger(__name__)


# Typing helpers


class PreAttentionProtocol(Protocol):
    n_heads: int

    def __call__(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, attention_mask: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any, Any, Any]:
        ...


class InnerAttentionProtocol(Protocol):
    attn_sink: torch.Tensor

    def sparse_attn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...


class PostAttentionProtocol(Protocol):
    n_groups: int

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...


# Alignment validation


def validate_cp_alignment(seq_len: int, cp_size: int) -> None:
    """
    seq_len % (cp_size * 128) == 0 is the tightest constraint; it implies
    chunk % 128 == 0 (C128A) and chunk % 4 == 0 (C4A) simultaneously.
    Even for C1A-only, we validate the full constraint so that adding
    C4A/C128A later requires no constraint changes.
    """
    if seq_len % (cp_size * 128) != 0:
        raise NotImplementedError(
            f"DeepSeek-V4 CP requires seq_len % (cp_size * 128) == 0. "
            f"Got seq_len={seq_len}, cp_size={cp_size}, "
            f"remainder={seq_len % (cp_size * 128)}."
        )


# BoundaryExchange — P2P with correct gradient back-propagation


@dataclass(frozen=True)
class BoundaryExchangeInfo:
    rank: int
    cp_size: int
    group: ProcessGroup | None
    init_value: float = 0.0


@dataclass(frozen=True)
class CPAttentionModules:
    pre_attn: PreAttentionProtocol
    inner_attn: InnerAttentionProtocol
    post_attn: PostAttentionProtocol


@dataclass(frozen=True)
class CPForwardContext:
    rank: int
    size: int
    group: ProcessGroup | None
    chunk_size: int
    compress_ratio: int
    window_size: int = 128


class AttentionForwardOutput(NamedTuple):
    x: torch.Tensor
    compress_topk_idxs: torch.Tensor | None
    offset: int | None
    q: torch.Tensor
    kv_compress: torch.Tensor | None
    attention_masks: Any
    index_score: Any
    q_indexer: Any
    k_indexer: Any
    weights: Any


class BoundaryExchange(torch.autograd.Function):
    """
    Forward: rank r-1 → rank r  (send_tensor → recv_buf)
    Backward: rank r   → rank r-1 (grad_recv  → grad_send)

    Uses group_dst / group_src (group-local rank indices) so the communication
    is correct under any multi-dimensional mesh, not just contiguous global ranks.

    init_value controls rank-0's recv_buf default:
      - kv data: 0.0   (matches non-CP overlap_transform value=0)
      - score: -inf  (reserved for C4A, not used in C1A)
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, send_tensor, info: BoundaryExchangeInfo):
        ctx.info = info

        recv_buf = (
            torch.zeros_like(send_tensor)
            if math.isclose(info.init_value, 0.0)
            else torch.full_like(send_tensor, info.init_value)
        )

        reqs = []
        if info.rank < info.cp_size - 1:
            reqs.append(
                dist.isend(
                    send_tensor.contiguous(),
                    group=info.group,
                    group_dst=info.rank + 1,
                )
            )
        if info.rank > 0:
            reqs.append(dist.irecv(recv_buf, group=info.group, group_src=info.rank - 1))
        for req in reqs:
            req.wait()  # pyrefly: ignore [missing-attribute]
        return recv_buf

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_recv):
        info = ctx.info
        grad_send = torch.zeros_like(grad_recv)
        reqs = []
        if info.rank > 0:
            reqs.append(
                dist.isend(
                    grad_recv.contiguous(), group=info.group, group_dst=info.rank - 1
                )
            )
        if info.rank < info.cp_size - 1:
            reqs.append(
                dist.irecv(grad_send, group=info.group, group_src=info.rank + 1)
            )
        for req in reqs:
            req.wait()  # pyrefly: ignore [missing-attribute]
        # send_tensor, info
        return grad_send, None


# AllGatherCompressedKV — forward AllGather, backward ReduceScatter


class AllGatherCompressedKV(torch.autograd.Function):
    """
    Forward: AllGather  local_kv  [B, chunk//r, D] → global_kv [B, seq//r, D]
    Backward: ReduceScatter grad_global             → grad_local
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, local_kv, group):
        ctx.group = group
        ctx.cp_size = dist.get_world_size(group)

        bsz, local_len, head_dim = local_kv.shape
        global_kv = torch.empty(
            bsz,
            local_len * ctx.cp_size,
            head_dim,
            dtype=local_kv.dtype,
            device=local_kv.device,
        )
        dist.all_gather_into_tensor(global_kv, local_kv.contiguous(), group=group)
        return global_kv

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_global):
        bsz, global_len, head_dim = grad_global.shape
        local_len = global_len // ctx.cp_size
        grad_local = torch.empty(
            bsz, local_len, head_dim, dtype=grad_global.dtype, device=grad_global.device
        )
        dist.reduce_scatter_tensor(
            grad_local, grad_global.contiguous(), group=ctx.group
        )
        return grad_local, None  # group has no gradient


# Shared boundary-exchange helper


def _exchange_and_concat_boundary_kv(kv, context: CPForwardContext):
    """Exchange boundary KV with the predecessor rank and prepend to local KV.

    Rank 0 receives zeros; the sum-times-zero trick keeps BoundaryExchange in
    the autograd graph so gradients flow correctly on all ranks.
    """
    n_boundary = context.window_size - 1
    boundary_kv = BoundaryExchange.apply(
        kv[:, -n_boundary:, :].contiguous(),
        BoundaryExchangeInfo(
            rank=context.rank,
            cp_size=context.size,
            group=context.group,
            init_value=0.0,
        ),
    )
    if boundary_kv is None:
        raise RuntimeError("BoundaryExchange.apply returned None.")
    if context.rank == 0:
        kv = kv + boundary_kv.sum(dim=(1, 2), keepdim=True) * 0.0
    return torch.cat([boundary_kv, kv], dim=1) if context.rank > 0 else kv


# CP forward (C1A and C128A)


def cp_forward(
    modules: CPAttentionModules,
    x: torch.Tensor,
    freqs_cis_global: torch.Tensor,
    attention_masks,
    context: CPForwardContext,
):
    """
    Unified CP forward for C1A (ratio=1) and C128A (ratio=128).

    Steps:
      1. Project Q/KV using the chunk's global RoPE positions.
      2. P2P BoundaryExchange: prepend boundary KV from the predecessor rank.
      3. C128A only: AllGather compressed KV and crop to causally visible tokens.
      4. Run sparse attention.
      5. Project output and return the standard 10-tuple.
    """
    bsz, chunk, _ = x.shape
    global_start = context.rank * context.chunk_size
    local_end = global_start + chunk
    freqs_local = freqs_cis_global[global_start:local_end].to(x.device)

    # Keep FSDP hooks on the pre_attention module boundary.
    q, local_window_kv, local_kv_compress, _, _, _, _ = modules.pre_attn(
        x, freqs_local, None
    )

    kv_full = _exchange_and_concat_boundary_kv(local_window_kv, context)
    offset = kv_full.size(1)

    # C1A (compress_ratio=1) pre_attn returns None for local_kv_compress because
    # window attention has no compressed KV; C128A returns a compressed KV tensor.
    # The None check distinguishes the two layer types without an explicit ratio branch.
    causal_kv_compress = None
    if local_kv_compress is not None:
        # AllGather collects each rank's local compressed KV into a global sequence.
        global_kv_compress = AllGatherCompressedKV.apply(
            local_kv_compress, context.group
        )
        if global_kv_compress is None:
            raise RuntimeError("AllGatherCompressedKV.apply returned None.")
        # Rank r can causally attend to tokens 0..(r+1)*chunk-1, which correspond
        # to (r+1)*(chunk//compress_ratio) compressed tokens.
        valid_compress_len = (context.rank + 1) * (chunk // context.compress_ratio)
        causal_kv_compress = global_kv_compress[:, :valid_compress_len, :]

    o = modules.inner_attn.sparse_attn(
        q,
        kv_full,
        modules.inner_attn.attn_sink,
        causal_kv_compress,
    )

    n_local_groups = modules.post_attn.n_groups // (
        modules.pre_attn.n_heads // q.shape[2]
    )
    x_out = modules.post_attn(o, freqs_local, bsz, chunk, n_local_groups)
    return AttentionForwardOutput(
        x=x_out,
        compress_topk_idxs=None,
        offset=offset if causal_kv_compress is not None else 0,
        q=q,
        kv_compress=causal_kv_compress,
        attention_masks=attention_masks,
        index_score=None,
        q_indexer=None,
        k_indexer=None,
        weights=None,
    )


# Patched Attention.forward dispatcher


def attention_forward_with_cp(self, x, freqs_cis, hadamard_mat, attention_masks):
    """
    CP-aware replacement for Attention.forward().

    freqs_cis received here is the GLOBAL table (self.freqs_cis or
    self.freqs_cis_wo_compressor from DeepSeekV4Model.forward).
    We do NOT do freqs_cis[:seqlen]; instead _cp_forward slices
    freqs_cis_global[global_start:global_start + chunk].
    """
    if self.compress_ratio not in (1, 128):
        raise NotImplementedError(
            f"DeepSeek-V4 CP for compress_ratio=4 (C4A) is not yet implemented. "
            f"Layer {self.layer_id} has compress_ratio={self.compress_ratio}."
        )
    _, chunk, _ = x.shape
    modules = CPAttentionModules(
        pre_attn=self.pre_attention,
        inner_attn=self.inner_attention,
        post_attn=self.post_attention,
    )
    context = CPForwardContext(
        rank=self.cp_rank,
        size=self.cp_size,
        group=self.cp_group,
        chunk_size=chunk,
        compress_ratio=self.compress_ratio,
        window_size=self.args.window_size,
    )
    return cp_forward(modules, x, freqs_cis, attention_masks, context)


# Entry point


def patch_deepseek_v4_for_context_parallel(
    model,
    cp_mesh: DeviceMesh,
) -> None:
    """
    Monkey-patch Attention.forward with the CP-aware implementation.
    Called from parallelize_deepseek_v4 when context_parallel_degree > 1.

    C1A (compress_ratio=1) and C128A (compress_ratio=128) are supported.
    C4A (compress_ratio=4) will raise NotImplementedError at runtime.
    """
    from torchtitan_npu.models.deepseek_v4.model.model import Attention, SparseAttention

    cp_group = cp_mesh.get_group()
    cp_rank = dist.get_rank(group=cp_group)
    cp_size = dist.get_world_size(group=cp_group)
    seq_len = model.model_args.max_seq_len

    validate_cp_alignment(seq_len, cp_size)

    # Attach CP context as attributes shared by all Attention instances.
    Attention.cp_rank = cp_rank
    Attention.cp_size = cp_size
    Attention.cp_group = cp_group
    Attention.cp_seq_len = seq_len
    Attention.forward = attention_forward_with_cp

    # SparseAttention may need cp_rank for future C128A/C4A index computations.
    SparseAttention.cp_rank = cp_rank
    SparseAttention.cp_size = cp_size
    SparseAttention.cp_seq_len = seq_len

    logger.info(
        f"[DeepSeek-V4 CP] Patched Attention.forward (C1A + C128A). "
        f"cp_rank={cp_rank}, cp_size={cp_size}, seq_len={seq_len}"
    )
