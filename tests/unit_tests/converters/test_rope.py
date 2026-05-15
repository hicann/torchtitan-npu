# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

from torchtitan_npu.converters.kernels.rope import (
    npu_apply_rotary_emb_deepseek,
    npu_apply_rotary_emb_deepseek_v4,
    npu_apply_rotary_emb_llama,
    npu_apply_rotary_emb_qwen,
    NpuRoPEConverter,
    RoPEKernel,
)


def test_model_impl_mapping():
    assert NpuRoPEConverter.MODEL_IMPL["deepseek_v3"] == npu_apply_rotary_emb_deepseek
    assert NpuRoPEConverter.MODEL_IMPL["deepseek_v32"] == npu_apply_rotary_emb_deepseek
    assert (
        NpuRoPEConverter.MODEL_IMPL["deepseek_v4"] == npu_apply_rotary_emb_deepseek_v4
    )
    assert NpuRoPEConverter.MODEL_IMPL["qwen3"] == npu_apply_rotary_emb_qwen
    assert NpuRoPEConverter.MODEL_IMPL["_default"] == npu_apply_rotary_emb_llama

    module_path, func_name, impl = RoPEKernel.get_impl_cls("deepseek_v3")

    assert module_path == "torchtitan.models.common.rope"
    assert func_name == "apply_rotary_emb_single_complex"
    assert impl is npu_apply_rotary_emb_deepseek


def test_deepseek_v3_impl_selection():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "deepseek_v3"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_deepseek

    # NOTE: "deepseek_v3" key matches "deepseek_v32" first due to substring
    # matching; v32-specific entry is currently unreachable. This is a known
    # issue to be fixed separately.
    module_path, func_name, impl = RoPEKernel.get_impl_cls("deepseek_v32")

    assert impl is npu_apply_rotary_emb_deepseek


def test_deepseek_v32_impl_selection():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "deepseek_v32"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_deepseek

    module_path, func_name, impl = RoPEKernel.get_impl_cls("qwen3")

    assert module_path == "torchtitan.models.common.rope"
    assert func_name == "apply_rotary_emb_cos_sin"
    assert impl is npu_apply_rotary_emb_qwen


def test_deepseek_v4_impl_selection():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "deepseek_v4"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_deepseek_v4


def test_qwen3_impl_selection():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "qwen3"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_qwen


def test_llama_impl_selection():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "llama3"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_llama


def test_unknown_model_fallback():
    mock_job_config = MagicMock()
    mock_job_config.model.name = "unknown_model"

    converter = NpuRoPEConverter(mock_job_config, None)
    impl = converter._get_impl_cls()

    assert impl == npu_apply_rotary_emb_llama
