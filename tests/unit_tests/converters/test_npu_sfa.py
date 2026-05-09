# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from torchtitan_npu.converters.kernels.deepseek_v4_sfa import sdpa_to_sfa_adapter

_mock_fused_fn = MagicMock()


class MockSparseAttn:
    apply = _mock_fused_fn


_mock_ops_mod = types.ModuleType("mindspeed.ops.npu_sparse_attn_shared_kv")

_mock_ops_mod.SparseAttnSharedKV = MockSparseAttn
_mock_ops_mod.npu_sparse_attn_shared_kv = _mock_fused_fn

sys.modules.setdefault("mindspeed", types.ModuleType("mindspeed"))
sys.modules.setdefault("mindspeed.ops", types.ModuleType("mindspeed.ops"))
sys.modules.setdefault("mindspeed.ops.npu_sparse_attn_shared_kv", _mock_ops_mod)


class TestSFANPUKernel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 1024
        self.seq_len_kv = 1024
        self.num_heads_q = 16
        self.head_dim = 128
        self.compress_ratio = 4
        self.topk = 256
        self.softmax_scale = 0.125

        self.query = torch.randn(
            self.batch_size, self.seq_len_q, self.num_heads_q, self.head_dim
        )
        self.kv_states = torch.randn(self.seq_len_kv, self.batch_size, self.head_dim)
        self.kv_compress = torch.randn(
            self.seq_len_kv // self.compress_ratio, self.batch_size, self.head_dim
        )
        self.indices = torch.randint(
            0, self.seq_len_kv, (self.batch_size, self.seq_len_q, self.topk)
        ).to(torch.int64)
        self.sinks = torch.randn(1, 128)

    def test_sdpa_to_sfa_adapter_flow(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()

        mock_fused_op.return_value = torch.randn(
            self.batch_size, self.seq_len_q, self.num_heads_q, self.head_dim
        )

        mock_self = MagicMock()
        mock_self.softmax_scale = self.softmax_scale
        mock_self.compress_ratio = self.compress_ratio
        mock_self.cp_rank = 0

        output = sdpa_to_sfa_adapter(
            mock_self,
            self.query,
            self.kv_states,
            self.sinks,
            self.kv_compress,
            self.indices,
        )

        args, _ = mock_fused_op.call_args

        cmp_sparse_indices = args[7]
        self.assertEqual(cmp_sparse_indices.dtype, torch.int32)
        ori_kv_processed = args[1]
        self.assertEqual(
            ori_kv_processed.shape, (self.seq_len_kv, self.batch_size, 1, self.head_dim)
        )

        self.assertTrue(output.is_contiguous())

    def test_cp_rank_gt_zero_c128a_pads_only_kv_to_global_positions(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()

        batch_size = 2
        chunk = 256
        num_heads = 3
        head_dim = 8
        cp_rank = 2
        window_size = 128
        n_boundary = window_size - 1
        query = torch.randn(batch_size, chunk, num_heads, head_dim)
        kv_states = torch.randn(batch_size, n_boundary + chunk, head_dim)
        kv_compress = torch.randn(batch_size, 6, head_dim)
        attn_sink = torch.randn(num_heads, dtype=torch.float16)
        fused_output = torch.randn(batch_size, num_heads, chunk, head_dim).transpose(
            1, 2
        )
        mock_fused_op.return_value = fused_output

        mock_self = self._make_sparse_attention(
            compress_ratio=128, cp_rank=cp_rank, window_size=window_size
        )
        output = sdpa_to_sfa_adapter(
            mock_self, query, kv_states, attn_sink, kv_compress
        )

        args, _ = mock_fused_op.call_args
        query_arg = args[0]
        ori_kv_arg = args[1]
        cmp_sparse_indices = args[7]
        sinks = args[8]

        kv_prefix_len = cp_rank * chunk - n_boundary
        self.assertEqual(query_arg.shape, query.shape)
        torch.testing.assert_close(query_arg, query)
        self.assertEqual(
            ori_kv_arg.shape,
            (batch_size, kv_prefix_len + n_boundary + chunk, 1, head_dim),
        )
        expected_prefix = torch.zeros(
            batch_size,
            kv_prefix_len,
            head_dim,
            dtype=kv_states.dtype,
            device=kv_states.device,
        )
        torch.testing.assert_close(ori_kv_arg[:, :kv_prefix_len, 0, :], expected_prefix)
        torch.testing.assert_close(ori_kv_arg[:, kv_prefix_len:, 0, :], kv_states)
        self.assertIsNone(cmp_sparse_indices)
        self.assertEqual(sinks.dtype, torch.float32)
        self.assertTrue(output.is_contiguous())

    def test_rank_zero_c128a_ignores_explicit_sparse_indices(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()
        mock_fused_op.return_value = torch.randn(
            self.batch_size, self.seq_len_q, self.num_heads_q, self.head_dim
        )

        mock_self = self._make_sparse_attention(compress_ratio=128, cp_rank=0)
        compress_topk_idxs = torch.randint(
            0, 16, (self.batch_size, self.seq_len_q, 8), dtype=torch.int64
        )

        sdpa_to_sfa_adapter(
            mock_self,
            self.query,
            torch.randn(self.batch_size, self.seq_len_kv, self.head_dim),
            self.sinks,
            torch.randn(self.batch_size, self.seq_len_kv // 128, self.head_dim),
            compress_topk_idxs,
        )

        args, _ = mock_fused_op.call_args
        self.assertIsNone(args[7])

    def test_cp_rank_gt_zero_c1a_uses_sfa_with_boundary_window(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()

        batch_size = 2
        chunk = 5
        num_heads = 3
        head_dim = 8
        window_size = 4
        query = torch.randn(batch_size, chunk, num_heads, head_dim)
        kv_states = torch.randn(batch_size, window_size - 1 + chunk, head_dim)
        attn_sink = torch.randn(num_heads)
        fused_output = torch.randn(batch_size, chunk, num_heads, head_dim)
        mock_fused_op.return_value = fused_output
        mock_self = self._make_sparse_attention(
            compress_ratio=1, cp_rank=3, window_size=window_size
        )

        output = sdpa_to_sfa_adapter(mock_self, query, kv_states, attn_sink)

        args, _ = mock_fused_op.call_args
        torch.testing.assert_close(args[0], query)
        torch.testing.assert_close(args[1], kv_states.unsqueeze(2))
        self.assertIsNone(args[2])
        self.assertIsNone(args[7])
        self.assertEqual(args[10], 1)
        self.assertEqual(args[13], window_size - 1)
        self.assertEqual(args[14], 0)
        self.assertEqual(args[8].dtype, torch.float32)
        torch.testing.assert_close(output, fused_output.contiguous())

    def _make_sparse_attention(self, compress_ratio, cp_rank=0, window_size=128):
        mock_self = MagicMock()
        mock_self.softmax_scale = self.softmax_scale
        mock_self.compress_ratio = compress_ratio
        mock_self.cp_rank = cp_rank
        mock_self.window_size = window_size
        return mock_self


if __name__ == "__main__":
    unittest.main()
