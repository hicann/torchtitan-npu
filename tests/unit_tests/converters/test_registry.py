# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.converters.npu_registry import registry


class DummyModelCustomConfig:
    name: str = "dummy"


def _run_register_case(register_name):
    calls = []
    test_registry = registry()

    original_register = test_registry._register_as_model_converter

    def mock_register(name, config):
        calls.append((name, config))

    test_registry._register_as_model_converter = mock_register

    decorated_config = test_registry.register(register_name)(DummyModelCustomConfig)
    config = test_registry.get(register_name)

    test_registry._register_as_model_converter = original_register

    return decorated_config, config, calls


def test_registry_is_singleton():
    registry1 = registry()
    registry2 = registry()

    assert registry1 is registry2


def test_register_sets_name_and_stores_config():
    decorated_config, config, calls = _run_register_case("unit_dummy")

    assert decorated_config is DummyModelCustomConfig
    assert config is not None
    assert config.name == "unit_dummy"
    assert len(calls) == 1
    assert calls[0][0] == "unit_dummy"
    assert calls[0][1] is DummyModelCustomConfig


def test_get_returns_none_for_unknown_config():
    assert registry.get("definitely_missing_config") is None


def test_core_converter_registrations_exist():
    expected_names = ["npu_dsa", "npu_rms_norm", "npu_rope", "npu_permute"]

    for name in expected_names:
        config = registry.get(name)
        assert config is not None, f"{name} should be registered"
        assert config.name == name
