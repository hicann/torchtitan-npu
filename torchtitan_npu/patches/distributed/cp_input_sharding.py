# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import wraps

import torchtitan.train as titan_train
from torchtitan.tools.logging import init_logger, logger

init_logger()


def _is_custom_cp_target(trainer) -> bool:
    if not getattr(trainer.parallel_dims, "cp_enabled", False):
        return False

    model_name = getattr(getattr(trainer, "job_config", None), "model", None)
    model_name = str(getattr(model_name, "name", ""))

    # DeepSeek-V4 always requires sequential sharding (window attention + compressor)
    if "deepseek_v4" in model_name:
        return True

    # DeepSeek-V32 with DSA requires sequential sharding (AllGather causal slice)
    if "deepseek_v32" not in model_name:
        return False

    if not getattr(trainer.model_args, "enable_indexer_loss", False):
        return False

    attn_type = getattr(trainer.model_args, "attn_type", "sdpa")
    return attn_type in ("sdpa", "dsa")


def _patch_post_dataloading_process_for_dsa_cp() -> None:
    original = titan_train.Trainer.post_dataloading_process

    @wraps(original)
    def wrapper(self, input_dict, labels):
        if not _is_custom_cp_target(self):
            return original(self, input_dict, labels)

        parallelism_cfg = self.job_config.parallelism
        old_lb = parallelism_cfg.context_parallel_load_balancer

        # Force sequential sharding: HeadTail is for ring-attention only.
        parallelism_cfg.context_parallel_load_balancer = None
        try:
            return original(self, input_dict, labels)
        finally:
            parallelism_cfg.context_parallel_load_balancer = old_lb

    titan_train.Trainer.post_dataloading_process = wrapper
    logger.info(
        "[Patch] Registered post_dataloading hook for CP sequential sharding "
        "(forces context_parallel_load_balancer=None for deepseek_v4 and deepseek_v32+DSA)."
    )


_patch_post_dataloading_process_for_dsa_cp()
