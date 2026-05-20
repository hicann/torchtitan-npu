# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize PP loss before backward to match the non-PP grad scale.

**Why this patch exists**

Upstream torchtitan's non-PP path divides ``loss_sum / global_valid_tokens``
before ``.backward()`` (see ``Trainer.forward_backward_step``). The PP path
delegates backward to ``pp_schedule.step``, which calls the raw ``loss_fn``
(``reduction='sum'``) and ``.backward()`` on the unnormalized result. So PP
grads end up ~``global_valid_tokens`` times larger than non-PP grads
(global_batch × seq_len ≈ 4096 on a 16×256 step).

``clip_grad_norm_(max_norm=1.0)`` cancels the difference at the parameter
update step, so loss curves stay bit-aligned between PP=1 and PP>1 — but the
logged ``grad_norm`` differs by ~``token_count`` because it's reported before
clipping. This makes PP vs non-PP runs look like they have wildly different
training dynamics when they don't.

**Fix**

1. Wrap ``pp_schedule._loss_fn`` so each microbatch's loss is divided by
   ``global_valid_tokens`` *before* PP backward — this normalizes grads.
2. Skip the upstream PP-path final-loss reduction (which would divide again
   by ``global_valid_tokens``) and replace it with ``sum(losses)`` over the
   microbatch list, matching the non-PP "loss = loss_sum / global_tokens"
   semantics exactly.
"""

from __future__ import annotations

import functools

import torch
import torchtitan.trainer as titan_trainer


_orig_forward_backward_step = titan_trainer.Trainer.forward_backward_step


def _as_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


@functools.wraps(_orig_forward_backward_step)
def _patched_forward_backward_step(
    self,
    *,
    input_dict,
    labels,
    global_valid_tokens,
    **kwargs,
):
    parallel_dims = self.parallel_dims
    if not parallel_dims.pp_enabled:
        return _orig_forward_backward_step(
            self,
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
            **kwargs,
        )

    schedule = self.pp_schedule
    orig_loss_fn = getattr(schedule, "_loss_fn", None)
    tokens = _as_float(global_valid_tokens) if global_valid_tokens is not None else 0.0
    if orig_loss_fn is None or tokens <= 0.0:
        return _orig_forward_backward_step(
            self,
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
            **kwargs,
        )

    inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
        input_dict, labels
    )

    def _normalized_loss_fn(pred, target, _orig=orig_loss_fn, _tokens=tokens):
        return _orig(pred, target) / _tokens

    schedule._loss_fn = _normalized_loss_fn
    try:
        with self.train_context():
            targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
            if self.pp_has_first_stage:
                schedule.step(
                    inputs,
                    **extra_inputs,
                    **extra_kwargs,
                    target=targets,
                    losses=losses,
                    return_outputs=False,
                )
            else:
                schedule.step(
                    **extra_kwargs,
                    target=targets,
                    losses=losses,
                    return_outputs=False,
                )
    finally:
        schedule._loss_fn = orig_loss_fn

    # Each entry in ``losses`` is already divided by ``global_valid_tokens``
    # (via _normalized_loss_fn), so summing reconstructs the same value the
    # non-PP path stores as ``loss = loss_sum / global_valid_tokens``.
    if self.pp_has_last_stage:
        loss = torch.sum(torch.stack(losses)).to(self.device)
    else:
        loss = torch.tensor([-1.0], device=self.device)

    return loss


titan_trainer.Trainer.forward_backward_step = _patched_forward_backward_step
