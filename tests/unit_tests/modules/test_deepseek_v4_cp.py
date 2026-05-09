# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan_npu.distributed.context_parallel import deepseek_v4_cp as cp


class TestDeepSeekV4CpAlignment:
    @staticmethod
    def test_validate_cp_alignment_accepts_128_aligned_sequence():
        cp.validate_cp_alignment(seq_len=4096, cp_size=4)

    @staticmethod
    def test_validate_cp_alignment_raises_with_remainder():
        with pytest.raises(
            NotImplementedError, match="seq_len % \\(cp_size \\* 128\\)"
        ):
            cp.validate_cp_alignment(seq_len=4097, cp_size=4)


class TestBoundaryExchange:
    @staticmethod
    def test_forward_middle_rank_posts_send_and_recv_with_group_local_ranks():
        send_tensor = torch.randn(2, 4, 3).transpose(1, 2)
        group = object()
        send_req = MagicMock()
        recv_req = MagicMock()

        with patch.object(
            cp.dist, "isend", return_value=send_req
        ) as mock_isend, patch.object(
            cp.dist, "irecv", return_value=recv_req
        ) as mock_irecv:
            ctx = MagicMock()
            recv_buf = cp.BoundaryExchange.forward(
                ctx,
                send_tensor,
                cp.BoundaryExchangeInfo(
                    rank=2, cp_size=4, group=group, init_value=-1.0
                ),
            )

        mock_isend.assert_called_once()
        isend_args, isend_kwargs = mock_isend.call_args
        assert isend_args[0].is_contiguous()
        assert isend_kwargs["group"] is group
        assert isend_kwargs["group_dst"] == 3

        mock_irecv.assert_called_once()
        irecv_args, irecv_kwargs = mock_irecv.call_args
        assert irecv_args[0] is recv_buf
        assert irecv_kwargs["group"] is group
        assert irecv_kwargs["group_src"] == 1

        send_req.wait.assert_called_once()
        recv_req.wait.assert_called_once()
        torch.testing.assert_close(recv_buf, torch.full_like(send_tensor, -1.0))

    @staticmethod
    def test_backward_middle_rank_sends_grad_left_and_receives_grad_right():
        grad_recv = torch.randn(2, 4, 3).transpose(1, 2)
        group = object()
        send_req = MagicMock()
        recv_req = MagicMock()
        ctx = MagicMock(
            info=cp.BoundaryExchangeInfo(rank=1, cp_size=4, group=group, init_value=0.0)
        )

        with patch.object(
            cp.dist, "isend", return_value=send_req
        ) as mock_isend, patch.object(
            cp.dist, "irecv", return_value=recv_req
        ) as mock_irecv:
            grad_send, info_grad = cp.BoundaryExchange.backward(ctx, grad_recv)

        mock_isend.assert_called_once()
        isend_args, isend_kwargs = mock_isend.call_args
        assert isend_args[0].is_contiguous()
        assert isend_kwargs["group"] is group
        assert isend_kwargs["group_dst"] == 0

        mock_irecv.assert_called_once()
        irecv_args, irecv_kwargs = mock_irecv.call_args
        assert irecv_args[0] is grad_send
        assert irecv_kwargs["group"] is group
        assert irecv_kwargs["group_src"] == 2

        send_req.wait.assert_called_once()
        recv_req.wait.assert_called_once()
        torch.testing.assert_close(grad_send, torch.zeros_like(grad_recv))
        assert info_grad is None


def _make_inner_post_attn_mocks(sparse_output, x_out):
    inner_attn = MagicMock()
    inner_attn.attn_sink = torch.randn(2)
    inner_attn.sparse_attn.return_value = sparse_output
    post_attn = MagicMock()
    post_attn.n_groups = 8
    post_attn.return_value = x_out
    return inner_attn, post_attn


@dataclass(frozen=True)
class _CPForwardSetup:
    inner_attn: object
    post_attn: object
    modules: cp.CPAttentionModules
    context: cp.CPForwardContext


@dataclass(frozen=True)
class _C1APreExchangeExpectation:
    kv: torch.Tensor
    freqs: torch.Tensor
    chunk: int
    context: cp.CPForwardContext


@dataclass(frozen=True)
class _C1APostExpectation:
    result: tuple
    sparse_output: torch.Tensor
    freqs_local: torch.Tensor
    batch_size: int
    chunk: int
    q: torch.Tensor
    x_out: torch.Tensor
    attention_masks: object


@dataclass(frozen=True)
class _C1AForwardCase:
    batch_size: int
    chunk: int
    window_size: int
    x: torch.Tensor
    freqs: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    boundary_kv: torch.Tensor
    sparse_output: torch.Tensor
    x_out: torch.Tensor
    attention_masks: object
    pre_attn: object
    setup: _CPForwardSetup


@dataclass(frozen=True)
class _C128AForwardCase:
    chunk: int
    window_size: int
    cp_group: object
    x: torch.Tensor
    freqs: torch.Tensor
    q: torch.Tensor
    local_window_kv: torch.Tensor
    local_kv_compress: torch.Tensor
    boundary_kv: torch.Tensor
    global_kv_compress: torch.Tensor
    attention_masks: object
    setup: _CPForwardSetup


def _make_cp_context(cp_rank, cp_size, cp_group, chunk, window_size):
    return cp.CPForwardContext(
        rank=cp_rank,
        size=cp_size,
        group=cp_group,
        chunk_size=chunk,
        compress_ratio=1,
        window_size=window_size,
    )


def _make_cp_forward_setup(pre_attn, sparse_output, x_out, context):
    inner_attn, post_attn = _make_inner_post_attn_mocks(sparse_output, x_out)
    modules = cp.CPAttentionModules(pre_attn, inner_attn, post_attn)
    return _CPForwardSetup(
        inner_attn=inner_attn,
        post_attn=post_attn,
        modules=modules,
        context=context,
    )


def _make_c1a_forward_case():
    batch_size = 1
    chunk = 4
    hidden_dim = 6
    kv_dim = 3
    window_size = 3
    cp_rank = 1
    cp_size = 4
    cp_group = object()
    x = torch.randn(batch_size, chunk, hidden_dim)
    freqs = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    q = torch.randn(batch_size, chunk, 2, kv_dim)
    kv = torch.randn(batch_size, chunk, kv_dim)
    boundary_kv = torch.randn(batch_size, window_size - 1, kv_dim)
    sparse_output = torch.randn(batch_size, chunk, 2, kv_dim)
    x_out = torch.randn(batch_size, chunk, hidden_dim)
    attention_masks = object()

    pre_attn = MagicMock()
    pre_attn.n_heads = 4
    pre_attn.return_value = (q, kv, None, None, None, None, None)
    context = _make_cp_context(cp_rank, cp_size, cp_group, chunk, window_size)
    setup = _make_cp_forward_setup(pre_attn, sparse_output, x_out, context)

    return _C1AForwardCase(
        batch_size=batch_size,
        chunk=chunk,
        window_size=window_size,
        x=x,
        freqs=freqs,
        q=q,
        kv=kv,
        boundary_kv=boundary_kv,
        sparse_output=sparse_output,
        x_out=x_out,
        attention_masks=attention_masks,
        pre_attn=pre_attn,
        setup=setup,
    )


def _call_c1a_forward_with_boundary(case):
    with patch.object(
        cp.BoundaryExchange, "apply", return_value=case.boundary_kv
    ) as mock_exchange:
        result = cp.cp_forward(
            case.setup.modules,
            case.x,
            case.freqs,
            case.attention_masks,
            case.setup.context,
        )
    return result, mock_exchange


def _assert_c1a_forward_case(case, result, mock_exchange):
    freqs_local = _assert_c1a_pre_attn_and_exchange(
        case.pre_attn,
        mock_exchange,
        _C1APreExchangeExpectation(
            kv=case.kv,
            freqs=case.freqs,
            chunk=case.chunk,
            context=case.setup.context,
        ),
    )
    _assert_c1a_sparse_attention(
        case.setup.inner_attn, case.q, case.boundary_kv, case.kv
    )
    _assert_c1a_post_attn_and_result(
        case.setup.post_attn,
        _C1APostExpectation(
            result=result,
            sparse_output=case.sparse_output,
            freqs_local=freqs_local,
            batch_size=case.batch_size,
            chunk=case.chunk,
            q=case.q,
            x_out=case.x_out,
            attention_masks=case.attention_masks,
        ),
    )


def _make_c128a_pre_attn(q, local_window_kv, local_kv_compress):
    pre_attn = MagicMock()
    pre_attn.n_heads = 4
    pre_attn.return_value = (
        q,
        local_window_kv,
        local_kv_compress,
        None,
        None,
        None,
        None,
    )
    return pre_attn


def _make_c128a_forward_case():
    batch_size = 1
    chunk = 256
    hidden_dim = 6
    kv_dim = 3
    window_size = 4
    cp_rank = 2
    cp_size = 4
    cp_group = object()
    x = torch.randn(batch_size, chunk, hidden_dim)
    freqs = torch.arange(2048, dtype=torch.float32).reshape(1024, 2)
    q = torch.randn(batch_size, chunk, 2, kv_dim)
    local_window_kv = torch.randn(batch_size, chunk, kv_dim)
    local_kv_compress = torch.randn(batch_size, chunk // 128, kv_dim)
    boundary_kv = torch.randn(batch_size, window_size - 1, kv_dim)
    global_kv_compress = torch.randn(batch_size, cp_size * (chunk // 128), kv_dim)
    sparse_output = torch.randn(batch_size, chunk, 2, kv_dim)
    x_out = torch.randn(batch_size, chunk, hidden_dim)
    attention_masks = object()

    pre_attn = _make_c128a_pre_attn(q, local_window_kv, local_kv_compress)
    context = cp.CPForwardContext(
        rank=cp_rank,
        size=cp_size,
        group=cp_group,
        chunk_size=chunk,
        compress_ratio=128,
        window_size=window_size,
    )
    setup = _make_cp_forward_setup(pre_attn, sparse_output, x_out, context)

    return _C128AForwardCase(
        chunk=chunk,
        window_size=window_size,
        cp_group=cp_group,
        x=x,
        freqs=freqs,
        q=q,
        local_window_kv=local_window_kv,
        local_kv_compress=local_kv_compress,
        boundary_kv=boundary_kv,
        global_kv_compress=global_kv_compress,
        attention_masks=attention_masks,
        setup=setup,
    )


def _call_c128a_forward_with_collectives(case):
    with patch.object(
        cp.BoundaryExchange, "apply", return_value=case.boundary_kv
    ) as mock_exchange, patch.object(
        cp.AllGatherCompressedKV, "apply", return_value=case.global_kv_compress
    ) as mock_allgather:
        result = cp.cp_forward(
            case.setup.modules,
            case.x,
            case.freqs,
            case.attention_masks,
            case.setup.context,
        )
    return result, mock_exchange, mock_allgather


def _assert_c128a_collectives(case, mock_exchange, mock_allgather):
    torch.testing.assert_close(
        mock_exchange.call_args.args[0], case.local_window_kv[:, -3:, :]
    )
    _assert_boundary_exchange_info(mock_exchange.call_args.args[1], case.setup.context)
    assert mock_allgather.call_args.args[0] is case.local_kv_compress
    assert mock_allgather.call_args.args[1] is case.cp_group


def _assert_c128a_forward_case(case, result, mock_exchange, mock_allgather):
    _assert_c128a_collectives(case, mock_exchange, mock_allgather)
    sparse_args, sparse_kwargs = case.setup.inner_attn.sparse_attn.call_args
    torch.testing.assert_close(
        sparse_args[1], torch.cat([case.boundary_kv, case.local_window_kv], dim=1)
    )
    torch.testing.assert_close(sparse_args[3], case.global_kv_compress[:, :6, :])
    assert sparse_kwargs == {}
    offset = case.window_size - 1 + case.chunk
    assert result[0] is case.setup.post_attn.return_value
    assert result[2] == offset
    assert result[4] is sparse_args[3]
    assert result[5] is case.attention_masks


def _assert_boundary_exchange_info(info, context):
    assert info.rank == context.rank
    assert info.cp_size == context.size
    assert info.group is context.group
    assert info.init_value == 0.0


def _assert_c1a_pre_attn_and_exchange(
    pre_attn, mock_exchange, expected: _C1APreExchangeExpectation
):
    freqs_end = 2 * expected.chunk
    freqs_local = expected.freqs.narrow(0, expected.chunk, freqs_end - expected.chunk)
    torch.testing.assert_close(pre_attn.call_args.args[1], freqs_local)
    torch.testing.assert_close(mock_exchange.call_args.args[0], expected.kv[:, -2:, :])
    _assert_boundary_exchange_info(mock_exchange.call_args.args[1], expected.context)
    return freqs_local


def _assert_c1a_sparse_attention(inner_attn, q, boundary_kv, kv):
    sparse_args, sparse_kwargs = inner_attn.sparse_attn.call_args
    torch.testing.assert_close(sparse_args[0], q)
    torch.testing.assert_close(sparse_args[1], torch.cat([boundary_kv, kv], dim=1))
    assert sparse_args[2] is inner_attn.attn_sink
    assert sparse_kwargs == {}


def _assert_c1a_post_attn_and_result(post_attn, expected: _C1APostExpectation):
    post_args, _ = post_attn.call_args
    assert post_args[0] is expected.sparse_output
    torch.testing.assert_close(post_args[1], expected.freqs_local)
    assert post_args[2:] == (expected.batch_size, expected.chunk, 4)
    assert expected.result[0] is expected.x_out
    assert expected.result[1] is None
    assert expected.result[2] == 0
    assert expected.result[3] is expected.q
    assert expected.result[4] is None
    assert expected.result[5] is expected.attention_masks
    assert expected.result[6:] == (None, None, None, None)


class TestC1AForwardWithCp:
    @staticmethod
    def test_rank_gt_zero_prepends_boundary_kv_and_calls_sparse_attention():
        case = _make_c1a_forward_case()
        result, mock_exchange = _call_c1a_forward_with_boundary(case)
        _assert_c1a_forward_case(case, result, mock_exchange)


class TestC128AForwardWithCp:
    @staticmethod
    def test_rank_gt_zero_crops_compress_kv_and_passes_global_topk():
        case = _make_c128a_forward_case()
        result, mock_exchange, mock_allgather = _call_c128a_forward_with_collectives(
            case
        )
        _assert_c128a_forward_case(case, result, mock_exchange, mock_allgather)


class TestPatchDeepSeekV4ForContextParallel:
    @staticmethod
    def test_sets_attention_and_sparse_attention_cp_context():
        from torchtitan_npu.models.deepseek_v4.model.model import (
            Attention,
            SparseAttention,
        )

        snapshot = TestPatchDeepSeekV4ForContextParallel._snapshot_class_state(
            Attention, SparseAttention
        )
        model = MagicMock()
        model.model_args.max_seq_len = 512
        cp_group = object()
        cp_mesh = MagicMock()
        cp_mesh.get_group.return_value = cp_group

        try:
            with patch.object(
                cp.dist, "get_rank", return_value=1
            ) as mock_get_rank, patch.object(
                cp.dist, "get_world_size", return_value=4
            ) as mock_get_world_size:
                cp.patch_deepseek_v4_for_context_parallel(model, cp_mesh)

            mock_get_rank.assert_called_once_with(group=cp_group)
            mock_get_world_size.assert_called_once_with(group=cp_group)
            assert Attention.cp_rank == 1  # pyrefly: ignore [missing-attribute]
            assert Attention.cp_size == 4  # pyrefly: ignore [missing-attribute]
            assert Attention.cp_group is cp_group  # pyrefly: ignore [missing-attribute]
            assert Attention.cp_seq_len == 512  # pyrefly: ignore [missing-attribute]
            assert (
                Attention.forward is cp.attention_forward_with_cp
            )  # pyrefly: ignore [missing-attribute]
            assert SparseAttention.cp_rank == 1  # pyrefly: ignore [missing-attribute]
            assert SparseAttention.cp_size == 4  # pyrefly: ignore [missing-attribute]
            assert SparseAttention.cp_seq_len == 512
        finally:
            TestPatchDeepSeekV4ForContextParallel._restore_class_state(
                Attention, SparseAttention, snapshot
            )

    @staticmethod
    def _snapshot_class_state(attention_cls, sparse_attention_cls):
        attrs = {}
        for cls, prefix in [
            (attention_cls, "attention"),
            (sparse_attention_cls, "sparse_attention"),
        ]:
            for attr in ["cp_rank", "cp_size", "cp_group", "cp_seq_len", "forward"]:
                key = (prefix, attr)
                attrs[key] = (hasattr(cls, attr), getattr(cls, attr, None))
        return attrs

    @staticmethod
    def _restore_class_state(attention_cls, sparse_attention_cls, snapshot):
        for cls, prefix in [
            (attention_cls, "attention"),
            (sparse_attention_cls, "sparse_attention"),
        ]:
            for attr in ["cp_rank", "cp_size", "cp_group", "cp_seq_len", "forward"]:
                had_attr, value = snapshot[(prefix, attr)]
                if had_attr:
                    setattr(cls, attr, value)
                elif hasattr(cls, attr):
                    delattr(cls, attr)
