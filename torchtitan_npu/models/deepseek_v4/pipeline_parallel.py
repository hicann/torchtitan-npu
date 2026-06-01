# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def _is_deepseek_v4_pp_target(trainer) -> bool:
    parallel_dims = getattr(trainer, "parallel_dims", None)
    if not getattr(parallel_dims, "pp_enabled", False):
        return False

    model_spec = getattr(trainer.config, "model_spec", None)
    return getattr(model_spec, "name", None) == "deepseek_v4"


def _with_deepseek_v4_pp_input_ids(trainer, result):
    inputs, labels, extra_inputs, extra_kwargs = result
    extra_inputs = extra_inputs or {}
    extra_kwargs = extra_kwargs or {}

    input_ids = extra_kwargs.get("input_ids")
    if "input_ids" in extra_inputs:
        extra_input_ids = extra_inputs.pop("input_ids")
        if input_ids is None:
            input_ids = extra_input_ids
    if input_ids is None:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise RuntimeError(
                "DeepSeekV4 PP input_ids injection expects inputs with shape [B, S]."
            )
        input_ids = inputs

    extra_kwargs["input_ids"] = input_ids.detach().long()
    return inputs, labels, extra_inputs, extra_kwargs
