# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

import torchtitan_npu.converters.kernels.deepseek_v4_sfa as sfa_mod
from torchtitan_npu.converters.kernels.deepseek_v4_sfa import NpuLiCompute

_mock_fused_fn = MagicMock()
_mock_li_op = MagicMock()
_mock_li_op.npu_lightning_indexer = _mock_fused_fn
sfa_mod._li_op = _mock_li_op


class TestLIKernel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 512
        self.seq_len_k = 1024
        self.dim = 64
        self.topk = 128
        self.ratio = 4
        self.q_indexer = torch.randn(
            self.batch_size, self.seq_len_q, self.dim, dtype=torch.float32
        )
        self.k_indexer = torch.randn(
            self.batch_size, self.seq_len_k, self.dim, dtype=torch.float32
        )
        self.weights = torch.randn(
            self.batch_size, self.seq_len_q, self.dim, dtype=torch.float32
        )

        self.mock_parent = MagicMock()
        self.mock_parent.index_topk = self.topk
        self.mock_parent.ratio = self.ratio

    def test_npu_li_compute_forward_logic(self):
        """Test NpuLiCompute.forward calls lightning indexer correctly."""
        mock_li_op = _mock_fused_fn
        mock_li_op.reset_mock()

        mock_indices = torch.full(
            (self.batch_size, self.seq_len_q, 1, self.topk), 10, dtype=torch.int32
        )
        mock_indices[0, 0, 0, 0] = -1
        mock_scores = torch.randn(
            self.batch_size, self.seq_len_q, 1, self.topk, dtype=torch.bfloat16
        )

        mock_li_op.return_value = (mock_indices, mock_scores)

        offset_val = 100

        wrapper = NpuLiCompute(self.mock_parent)

        res_indices, res_scores = wrapper.forward(
            self.q_indexer,
            self.k_indexer,
            self.weights,
            seqlen=self.seq_len_q,
            offset=offset_val,
        )

        args, kwargs = mock_li_op.call_args

        self.assertEqual(args[0].dtype, torch.bfloat16)  # q
        self.assertEqual(args[1].dtype, torch.bfloat16)  # k
        self.assertEqual(args[2].dtype, torch.bfloat16)  # weights

        self.assertEqual(args[1].ndim, 4)

        self.assertEqual(
            res_indices.shape, (self.batch_size, self.seq_len_q, self.topk)
        )
        self.assertEqual(res_scores.shape, (self.batch_size, self.seq_len_q, self.topk))

        self.assertEqual(res_indices[0, 0, 0].item(), -1)

        self.assertEqual(res_indices[0, 0, 1].item(), 110)


if __name__ == "__main__":
    unittest.main()
