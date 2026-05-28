# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import wraps

import torchtitan.train as titan_train
from torchtitan.tools.logging import init_logger, logger

init_logger()


def _is_dsa_cp_target(trainer) -> bool:
    if not getattr(trainer.parallel_dims, "cp_enabled", False):
        return False

    model_name = str(getattr(getattr(trainer.config, "model_spec", None), "name", ""))

    # Qwen3 only supports Ulysses CP (not upstream HeadTail ring attention),
    # so cp_enabled=True always means Ulysses CP → sequential sharding required
    if "qwen3" in model_name:
        return True

    # DeepSeek-V32 with DSA requires sequential sharding (AllGather causal slice)
    if "deepseek_v32" not in model_name:
        return False

    model_config = getattr(trainer, "model_config", None)
    if model_config is None:
        return False
    if not getattr(model_config, "enable_indexer_loss", False):
        return False

    attn_type = getattr(model_config, "attn_type", "sdpa")
    return attn_type in ("sdpa", "dsa")


def _patch_post_dataloading_process_for_dsa_cp() -> None:
    original = titan_train.Trainer.post_dataloading_process

    @wraps(original)
    def wrapper(self, input_dict, labels):
        if not _is_dsa_cp_target(self):
            return original(self, input_dict, labels)

        parallelism_cfg = self.config.parallelism
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
        "(forces context_parallel_load_balancer=None for qwen3, and deepseek_v32+DSA)."
    )


_patch_post_dataloading_process_for_dsa_cp()
