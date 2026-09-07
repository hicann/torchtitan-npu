# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC head-compute-mix with the TileLang fused kernel.

This is the override-refactor counterpart of master's
``converters/kernels/mhc_head_compute_mix_tilelang.py`` converter: instead of
swapping the ``HcHead`` module at converter time, we subclass it and replace the
``forward`` with a call to the TileLang fused op
``mhc_head_compute_mix_tilelang`` (see ``ops/tilelang/head_compute_mix.py``).
"""

from dataclasses import dataclass

from torch import Tensor
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.deepseek_v4.mhc import HcHead
from torchtitan_npu.ops.tilelang import mhc_head_compute_mix_tilelang


def _to_local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TilelangHcHead(HcHead):
    """HcHead backed by the TileLang fused ``mhc_head_compute_mix`` kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcHead.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # ``HcHead`` only uses hc_mult to size its parameters and drops it; the
        # TileLang kernel needs it as ``num_stream``.
        self.hc_mult = config.hc_mult

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "TilelangHcHead expects local tensor input; apply HcHeadParallelStyle with local TP input."
            )
        hc_fn = _to_local_tensor(self.hc_fn)
        hc_base = _to_local_tensor(self.hc_base)
        hc_scale = _to_local_tensor(self.hc_scale)

        is_tnd = x.dim() == 3

        if is_tnd:
            x = x.flatten(1).unsqueeze(1)  # [T, N, D] -> [T, 1, N*D]
        elif x.dim() == 4:
            x = x.flatten(2)  # [B, S, N, D] -> [B, S, N*D]
        else:
            raise ValueError(
                f"TilelangHcHead expects 3D [T, N, D] or 4D [B, S, N, D] tensor, but got input with shape {x.shape}"
            )

        y = mhc_head_compute_mix_tilelang(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            None,
            False,
            self.norm_eps,
            self.eps,
            self.hc_mult,
        )

        if is_tnd:
            y = y.squeeze(1)  # [T, 1, out_features] -> [T, out_features]

        return y
