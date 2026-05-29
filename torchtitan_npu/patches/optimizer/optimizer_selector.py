# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import fields, replace
from typing import Any

from torchtitan.components.optimizer import OptimizersContainer

logger = logging.getLogger(__name__)


class NpuOptimizerDispatcher:
    _virtual_patched = False
    _swap_patched = False

    @staticmethod
    def dispatch_build(self: Any, **kwargs) -> OptimizersContainer:
        config_fields = {f.name for f in fields(self)}
        overlap = config_fields & kwargs.keys()
        if overlap:
            raise ValueError(f"build() kwargs {overlap} overlap with config fields.")

        is_virtual = getattr(self, "virtual_optimizer", False)
        is_swap = getattr(self, "swap_optimizer", False)

        if is_virtual and is_swap:
            raise ValueError(
                "Cannot enable both virtual_optimizer and swap_optimizer at the same time."
            )

        if is_virtual:
            logger.info("[OptimizerDispatcher] Using VirtualOptimizer")

            from .virtual_optimizer import VirtualOptimizersContainer

            NpuOptimizerDispatcher._apply_virtual_patch()

            return VirtualOptimizersContainer(config=replace(self), **kwargs)

        elif is_swap:
            logger.info("[OptimizerDispatcher] Using SwapOptimizer")

            from .swap_optimizer import SwapOptimizersContainer

            NpuOptimizerDispatcher._apply_swap_patch()

            return SwapOptimizersContainer(config=replace(self), **kwargs)

        else:
            logger.info("[OptimizerDispatcher] Using standard Optimizer")
            base_config = OptimizersContainer.Config(
                name=self.name,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                implementation=self.implementation,
            )
            return OptimizersContainer(config=base_config, **kwargs)

    @classmethod
    def _apply_virtual_patch(cls):
        if cls._virtual_patched:
            return

        import torch

        from .virtual_optimizer import (
            _make_patched_load,
            patched_state_dict,
            swap_tensor_copy_wrapper,
            swap_tensor_func_wrapper,
            virtual_optimizer_step,
        )

        torch.Tensor.copy_ = swap_tensor_copy_wrapper(torch.Tensor.copy_)
        torch.Tensor.cpu = swap_tensor_func_wrapper(torch.Tensor.cpu, "cpu")
        torch.Tensor.clone = swap_tensor_func_wrapper(torch.Tensor.clone, "clone")
        torch.Tensor.detach = swap_tensor_func_wrapper(torch.Tensor.detach, "detach")

        for cls_opt in [torch.optim.AdamW, torch.optim.Adam]:
            if not hasattr(cls_opt, "_original_state_dict"):
                cls_opt._original_state_dict = cls_opt.state_dict
                cls_opt.state_dict = patched_state_dict
                cls_opt.step = virtual_optimizer_step
                cls_opt._original_load_state_dict = cls_opt.load_state_dict
                cls_opt.load_state_dict = _make_patched_load(cls_opt.load_state_dict)

        cls._virtual_patched = True
        logger.info("[VirtualOptimizer] Patched Adam/AdamW successfully")

    @classmethod
    def _apply_swap_patch(cls):
        if cls._swap_patched:
            return

        import torch

        from .swap_optimizer import swap_optimizer_step

        torch.optim.AdamW.step = swap_optimizer_step
        torch.optim.Adam.step = swap_optimizer_step
        cls._swap_patched = True
        logger.info("[SwapOptimizer] Patched Adam/AdamW successfully")


def patch_npu_optimizer_framework():
    try:
        from torchtitan_npu.config.configs import OptimizerConfig

        OptimizerConfig.build = NpuOptimizerDispatcher.dispatch_build
        logger.info("[OptimizerFramework] Patch successful")
    except Exception as e:
        logger.error(f"Patch failed: {e}")
        raise
