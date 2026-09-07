# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TileLang fused DeepSeek-V4 kernels exposed as torch custom operators.

The head-compute-mix kernel is registered via ``torch.library.custom_op``
(with ``register_fake`` and ``register_autograd``); the differentiable
composition over it lives in
``torchtitan_npu/override/deepseek_v4/mhc/tilelang.py``.
"""

__all__ = ["mhc_head_compute_mix_tilelang"]

from .head_compute_mix import mhc_head_compute_mix_tilelang
