# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest

from torchtitan_npu.converters.kernels.rope import (
    _ROPE_REPLACEMENTS,
    npu_apply_rotary_emb_complex,
    npu_apply_rotary_emb_cos_sin,
    npu_apply_rotary_emb_single_complex,
    NpuRoPEConverter,
)


def test_rope_replacement_mapping_tracks_current_upstream_api():
    assert _ROPE_REPLACEMENTS == {
        "apply_rotary_emb_complex": npu_apply_rotary_emb_complex,
        "apply_rotary_emb_single_complex": npu_apply_rotary_emb_single_complex,
        "apply_rotary_emb_cos_sin": npu_apply_rotary_emb_cos_sin,
    }


@pytest.mark.parametrize(
    "func_name,impl", list(_ROPE_REPLACEMENTS.items()), ids=list(_ROPE_REPLACEMENTS)
)
def test_replace_one_invokes_replace_functions_for_each_entry(func_name, impl):
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(func_name, impl, fake_model)

    assert mock_replace.call_count >= 1
    for call in mock_replace.call_args_list:
        assert call.args[0] == func_name
        assert call.args[1] is impl


def test_convert_iterates_all_replacements():
    converter = NpuRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()

    with patch.object(NpuRoPEConverter, "_replace_one") as mock_replace_one:
        converter.convert(fake_model)

    assert mock_replace_one.call_count == len(_ROPE_REPLACEMENTS)
    called_pairs = {
        (call.args[0], call.args[1].__name__)
        for call in mock_replace_one.call_args_list
    }
    expected_pairs = {
        (name, impl.__name__) for name, impl in _ROPE_REPLACEMENTS.items()
    }
    assert called_pairs == expected_pairs
    for call in mock_replace_one.call_args_list:
        assert call.args[2] is fake_model


def test_replace_one_walks_three_packages_for_npu_model():
    """
    torchtitan_npu.* model → walk three locations:
    (1) the model's own module tree,
    (2) the upstream-rewritten package (torchtitan_npu→torchtitan),
    (3) the shared torchtitan.models.common package.
    """
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 3
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {
        "package": "torchtitan.models.deepseek_v4.model"
    }
    assert mock_replace.call_args_list[2].kwargs == {
        "package": "torchtitan.models.common"
    }


def test_replace_one_walks_two_packages_when_model_is_already_upstream():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 2
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {
        "package": "torchtitan.models.common"
    }


def test_replace_one_walks_only_model_when_already_in_common_pkg():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.common.rope"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 1
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
