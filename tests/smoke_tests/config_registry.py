# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters.npu_registry import get_model_converter_config
from torchtitan_npu.models.deepseek_v32 import model_registry
from torchtitan_npu.patches.optimizer.swap_optimizer import SwapOptimizersContainer


def deepseek_v32_smoketest() -> TrainerConfig:
    # Minimal end-to-end smoke config. ``npu_gmm`` is intentionally omitted
    # because dsv32 parallelize.py asserts gmm requires EP when TP is on.
    # ``npu_dsa`` is also left out of daily smoke because the fused sparse
    # attention path is covered separately and can be unstable on CI kernels;
    # this config uses the SDPA fallback for a stable end-to-end check.
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv32_tokenizer",
        model_spec=model_registry("smoketest"),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                get_model_converter_config("npu_rms_norm"),
                get_model_converter_config("npu_rope"),
                get_model_converter_config("npu_permute"),
            ],
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=SwapOptimizersContainer.Config(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        profiling=ProfilingConfig(enable_profiling=False),
    )
