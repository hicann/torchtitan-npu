# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
from types import SimpleNamespace

from torchtitan_npu.models.deepseek_v32 import deepseekv32_configs


def test_update_from_config_keeps_grouped_mm_enabled():
    model_config = deepcopy(deepseekv32_configs["smoketest"]())
    moe_layers = [layer for layer in model_config.layers if layer.moe is not None]
    assert moe_layers
    for layer in moe_layers:
        layer.moe.experts.use_grouped_mm = True

    trainer_config = SimpleNamespace(
        training=SimpleNamespace(seq_len=4096),
        parallelism=SimpleNamespace(
            context_parallel_degree=1,
            expert_parallel_comm_backend="standard",
        ),
        debug=SimpleNamespace(moe_force_load_balance=False),
    )

    model_config.update_from_config(trainer_config=trainer_config)

    for layer in model_config.layers:
        if layer.moe is not None:
            assert layer.moe.experts.use_grouped_mm is True
