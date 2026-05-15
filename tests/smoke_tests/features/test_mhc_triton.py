# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import assert_tensor_finite, stable_randn
from torchtitan_npu.ops.triton.add import add_fwd
from torchtitan_npu.ops.triton.mhc_triton import (
    MHCPostTriton,
    MHCPreOnlyTriton,
    MHCPreTriton,
)

pytestmark = pytest.mark.smoke

_RTOL = 1e-2
_ATOL = 1e-2


@dataclass
class HcSplitConfig:
    num_stream: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    mhc_use_gamma: bool = True


@dataclass
class HcWeights:
    phi_weight: torch.Tensor
    branch_alpha: torch.Tensor
    branch_beta: torch.Tensor
    norm_gamma: torch.Tensor


def sinkhorn_knopps(h_res, sinkhorn_iters, eps):
    h_res = h_res.softmax(-1) + eps
    col_sum = h_res.sum(-2, keepdim=True)
    h_res = h_res / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = h_res.sum(-1, keepdim=True)
        h_res = h_res / (row_sum + eps)
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
    return h_res


def hc_split_sinkhorn_torch(weight, branch_alpha, branch_beta, cfg):
    ns = cfg.num_stream
    h_pre, h_post, h_res = weight.split([ns, ns, ns * ns], dim=-1)
    h_res = h_res.unflatten(-1, (ns, ns))

    h_pre = (
        F.sigmoid(h_pre * branch_alpha[0] + branch_beta[:ns].unsqueeze(0).unsqueeze(0))
        + cfg.eps
    )
    h_post = 2 * F.sigmoid(
        h_post * branch_alpha[1] + branch_beta[ns : 2 * ns].unsqueeze(0).unsqueeze(0)
    )
    h_res = h_res * branch_alpha[2] + branch_beta[2 * ns :].view(ns, ns).unsqueeze(
        0
    ).unsqueeze(0)

    h_res = sinkhorn_knopps(h_res, cfg.sinkhorn_iters, cfg.eps)
    return h_pre, h_post, h_res


class MhcModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = HcSplitConfig()
        self.norm_eps = 1e-6

    def hc_pre(self, x, w):
        dtype = x.dtype
        x = x.float()
        phi_weight = w.phi_weight.float().t()
        branch_alpha = w.branch_alpha.float()
        branch_beta = w.branch_beta.float()
        norm_gamma = w.norm_gamma.float()

        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        x_normed = x * rsqrt * norm_gamma if self.cfg.mhc_use_gamma else x * rsqrt

        weight = torch.matmul(x_normed, phi_weight)
        h_pre, h_post, h_res = hc_split_sinkhorn_torch(
            weight, branch_alpha, branch_beta, self.cfg
        )
        y = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.cfg.num_stream, -1)),
            dim=2,
        )
        return y.to(dtype), h_post, h_res

    def hc_post(self, x, residual, h_post, h_res):
        y = (
            h_post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(
                h_res.unsqueeze(-1)
                * residual.unflatten(dim=-1, sizes=(self.cfg.num_stream, -1)).unsqueeze(
                    -2
                ),
                dim=2,
            )
        ).flatten(2)
        return y.type_as(x)

    def hc_pre_only(self, x, w, cfg=None):
        if cfg is None:
            cfg = self.cfg
        dtype = x.dtype
        x = x.float()
        phi_weight = w.phi_weight.float().t()
        branch_alpha = w.branch_alpha.float()
        branch_beta = w.branch_beta.float()
        norm_gamma = w.norm_gamma.float()

        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.eps)
        if cfg.mhc_use_gamma:
            weight = torch.matmul(x * rsqrt * norm_gamma, phi_weight)
        else:
            weight = torch.matmul(x, phi_weight) * rsqrt
        h_pre = (
            F.sigmoid(weight * branch_alpha + branch_beta.unsqueeze(0).unsqueeze(0))
            + cfg.eps
        )
        y = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(cfg.num_stream, -1)), dim=2
        )
        return y.to(dtype)


def _assert_grads_close(pairs, rtol=_RTOL, atol=_ATOL):
    for torch_grad, triton_grad in pairs:
        torch.testing.assert_close(
            torch_grad, triton_grad, rtol=rtol, atol=atol, equal_nan=True
        )


def _backward_and_assert(out_torch, out_triton, grad_pairs):
    out_torch.sum().backward()
    out_triton.sum().backward()
    _assert_grads_close(grad_pairs)


def _clone_pair(tensor):
    a = tensor.clone().detach().requires_grad_(True)
    b = tensor.clone().detach().requires_grad_(True)
    return a, b


def _make_pre_only_inputs(device, n, d):
    hidden = n * d
    x = torch.rand(
        1, 1024, hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.rand(
        n, hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    branch_alpha = torch.rand(1, device=device, dtype=torch.float32, requires_grad=True)
    branch_beta = torch.rand(n, device=device, dtype=torch.float32, requires_grad=True)
    norm_gamma = torch.rand(
        hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    x_t, x_i = _clone_pair(x)
    w_t, w_i = _clone_pair(weight)
    a_t, a_i = _clone_pair(branch_alpha)
    b_t, b_i = _clone_pair(branch_beta)
    g_t, g_i = _clone_pair(norm_gamma)
    return x_t, x_i, HcWeights(w_t, a_t, b_t, g_t), HcWeights(w_i, a_i, b_i, g_i)


def _make_pre_inputs(device, n, d):
    hidden = n * d
    x = torch.rand(
        1, 1024, hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    weight = torch.rand(
        n * n + 2 * n, hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    branch_alpha = torch.rand(3, device=device, dtype=torch.float32, requires_grad=True)
    branch_beta = torch.rand(
        2 * n + n * n, device=device, dtype=torch.float32, requires_grad=True
    )
    norm_gamma = torch.rand(
        hidden, device=device, dtype=torch.float32, requires_grad=True
    )
    x_t, x_i = _clone_pair(x)
    w_t, w_i = _clone_pair(weight)
    a_t, a_i = _clone_pair(branch_alpha)
    b_t, b_i = _clone_pair(branch_beta)
    g_t, g_i = _clone_pair(norm_gamma)
    return x_t, x_i, HcWeights(w_t, a_t, b_t, g_t), HcWeights(w_i, a_i, b_i, g_i)


def test_add_triton(npu_device):
    x = stable_randn(1024, 32768, device=npu_device, dtype=torch.float32)
    y = stable_randn(1024, 32768, device=npu_device, dtype=torch.float32)

    out_triton = add_fwd(x, y)
    out_torch = x + y

    torch.testing.assert_close(
        out_triton, out_torch, rtol=1e-3, atol=1e-3, equal_nan=True
    )
    assert_tensor_finite(out_triton, "add_fwd output should be finite")


def test_mhc_pre_only_triton(npu_device):
    torch.manual_seed(42)
    n, d = 4, 8192
    x_torch, x_triton, w_torch, w_triton = _make_pre_only_inputs(npu_device, n, d)

    cfg = HcSplitConfig(num_stream=n)
    mhc_torch = MhcModule()
    y_torch = mhc_torch.hc_pre_only(x_torch, w_torch, cfg)
    y_triton = MHCPreOnlyTriton.apply(
        x_triton,
        w_triton.phi_weight,
        w_triton.branch_alpha,
        w_triton.branch_beta,
        w_triton.norm_gamma,
        True,
        cfg.eps,
        n,
    )

    torch.testing.assert_close(
        y_torch, y_triton, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    assert_tensor_finite(y_triton, "MHCPreOnlyTriton output should be finite")

    _backward_and_assert(
        y_torch,
        y_triton,
        [
            (x_torch.grad, x_triton.grad),
            (w_torch.phi_weight.grad, w_triton.phi_weight.grad),
            (w_torch.branch_alpha.grad, w_triton.branch_alpha.grad),
            (w_torch.branch_beta.grad, w_triton.branch_beta.grad),
            (w_torch.norm_gamma.grad, w_triton.norm_gamma.grad),
        ],
    )


def test_mhc_post_triton(npu_device):
    torch.manual_seed(42)
    b, s, n, d = 1, 1024, 4, 8192

    x = torch.rand(b, s, d, device=npu_device, dtype=torch.float32, requires_grad=True)
    x_torch = x.clone().detach().requires_grad_(True)
    x_triton = x.clone().detach().requires_grad_(True)
    residual = torch.rand(
        b, s, n * d, device=npu_device, dtype=torch.float32, requires_grad=True
    )
    residual_torch = residual.clone().detach().requires_grad_(True)
    residual_triton = residual.clone().detach().requires_grad_(True)
    h_post = torch.rand(
        b, s, n, device=npu_device, dtype=torch.float32, requires_grad=True
    )
    h_post_torch = h_post.clone().detach().requires_grad_(True)
    h_post_triton = h_post.clone().detach().requires_grad_(True)
    h_res = torch.rand(
        b, s, n, n, device=npu_device, dtype=torch.float32, requires_grad=True
    )
    h_res_torch = h_res.clone().detach().requires_grad_(True)
    h_res_triton = h_res.clone().detach().requires_grad_(True)

    mhc_torch = MhcModule()
    result_torch = mhc_torch.hc_post(x_torch, residual_torch, h_post_torch, h_res_torch)
    result_triton = MHCPostTriton.apply(
        x_triton, residual_triton, h_post_triton, h_res_triton
    )

    torch.testing.assert_close(
        result_torch, result_triton, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    assert_tensor_finite(result_triton, "MHCPostTriton output should be finite")

    result_torch.sum().backward()
    result_triton.sum().backward()
    _assert_grads_close(
        [
            (x_torch.grad, x_triton.grad),
            (residual_torch.grad, residual_triton.grad),
            (h_post_torch.grad, h_post_triton.grad),
            (h_res_torch.grad, h_res_triton.grad),
        ]
    )


def test_mhc_pre_triton(npu_device):
    torch.manual_seed(42)
    n, d = 4, 8192
    x_torch, x_triton, w_torch, w_triton = _make_pre_inputs(npu_device, n, d)

    mhc_torch = MhcModule()
    y_torch, h_post_torch, h_res_torch = mhc_torch.hc_pre(x_torch, w_torch)
    y_triton, h_post_triton, h_res_triton = MHCPreTriton.apply(
        x_triton,
        w_triton.phi_weight,
        w_triton.branch_alpha,
        w_triton.branch_beta,
        w_triton.norm_gamma,
        True,
        n,
        20,
        1e-6,
    )

    torch.testing.assert_close(
        y_torch, y_triton, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    torch.testing.assert_close(
        h_post_torch, h_post_triton, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    torch.testing.assert_close(
        h_res_torch, h_res_triton, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    assert_tensor_finite(y_triton, "MHCPreTriton output should be finite")

    _backward_and_assert(
        y_torch,
        y_triton,
        [
            (x_torch.grad, x_triton.grad),
            (w_torch.phi_weight.grad, w_triton.phi_weight.grad),
            (w_torch.branch_alpha.grad, w_triton.branch_alpha.grad),
            (w_torch.branch_beta.grad, w_triton.branch_beta.grad),
            (w_torch.norm_gamma.grad, w_triton.norm_gamma.grad),
        ],
    )
