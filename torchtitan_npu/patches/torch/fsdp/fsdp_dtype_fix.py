# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/pull/183805
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Patch for FSDP2: sync DTensor spec.tensor_meta.dtype after mixed precision cast.

When FSDP2 + TP + MixedPrecisionPolicy(param_dtype=bf16) are combined,
init_unsharded_param() reconstructs a DTensor with bf16 _local_tensor
but fp32 _spec.tensor_meta.dtype (frozen from _tp_spec at init time).

This causes gen_fake_args() in sharding propagation to create fp32 fake
tensors while the real local tensors are bf16, leading to inconsistencies
in both eager and torch.compile paths.

The upstream fix adds _get_unsharded_dtensor_spec() which checks if
tensor_meta.dtype matches the actual unsharded param dtype and creates
a corrected DTensorSpec when they mismatch (due to mixed precision cast).
"""
from dataclasses import replace

from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor._dtensor_spec import TensorMeta


_orig_init_unsharded_param = FSDPParam.init_unsharded_param


def _get_unsharded_dtensor_spec(tp_spec, unsharded_param):
    """
    If the spec's tensor_meta.dtype differs from the unsharded param's dtype
    (due to mixed precision cast), create a new spec with the correct dtype.
    """
    tensor_meta = tp_spec.tensor_meta
    if tensor_meta is None or tensor_meta.dtype == unsharded_param.dtype:
        return tp_spec
    return replace(
        tp_spec,
        tensor_meta=TensorMeta(
            tensor_meta.shape,
            tensor_meta.stride,
            unsharded_param.dtype,
        ),
    )


def _patched_init_unsharded_param(self):
    # Call the original first - this handles all the actual logic
    _orig_init_unsharded_param(self)

    # Now fix the DTensor's _spec if dtype is mismatched
    if not self.is_dtensor or not hasattr(self, "_unsharded_param"):
        return

    up = self._unsharded_param
    if not hasattr(up, "_local_tensor") or not hasattr(up, "_spec"):
        return

    local_dtype = up._local_tensor.dtype
    spec_dtype = up._spec.tensor_meta.dtype if up._spec.tensor_meta else None

    if local_dtype != spec_dtype:
        corrected_spec = _get_unsharded_dtensor_spec(up._spec, up._local_tensor)
        up._spec = corrected_spec


def apply_patch():
    """Apply the FSDP2 dtype sync patch if the upstream fix is not present.

    TODO: remove this entire file and its import in __init__.py when bumping
    torch to a version that includes pytorch#183805. The upstream fix adds
    _get_unsharded_dtensor_spec on FSDPParam; once present, this patch logs
    a warning and becomes dead code.
    """
    import logging

    logger = logging.getLogger(__name__)

    if hasattr(FSDPParam, "_get_unsharded_dtensor_spec"):
        logger.warning(
            "FSDP2 dtype sync patch skipped: installed PyTorch already "
            "includes the upstream fix (_get_unsharded_dtensor_spec). "
            "The patch file (patches/torch/fsdp/fsdp_dtype_fix.py) can be removed."
        )
        return

    FSDPParam.init_unsharded_param = _patched_init_unsharded_param
    logger.info("FSDP2 dtype sync patch applied (backport from pytorch#183805).")


apply_patch()
