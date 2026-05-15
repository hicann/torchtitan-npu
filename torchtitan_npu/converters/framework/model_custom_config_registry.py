# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import Configurable

from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig


class _ConverterRegistry:
    _instance = None
    _model_configs: dict[str, ModelCustomConfig]
    _converter_classes: dict[str, type]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_configs = {}
            cls._instance._converter_classes = {}

        return cls._instance

    def register(
        self,
        name: str,
    ):
        def decorator(config: ModelCustomConfig):
            config.name = name
            self._model_configs[name] = config
            self._register_as_model_converter(name, config)
            return config

        return decorator

    def get_config(self, name: str) -> Configurable.Config | None:
        converter_cls = self._converter_classes.get(name)
        if converter_cls is None:
            return None
        return converter_cls.Config()

    def get(self, name: str) -> ModelCustomConfig | None:
        return self._model_configs.get(name)

    def _register_as_model_converter(
        self,
        name: str,
        config: ModelCustomConfig,
    ):
        from .model_custom_config_converter import ModelCustomConfigConverter

        converter_cls = type(
            f"{name}ModelCustomConfigConverter",
            (ModelCustomConfigConverter,),
            {
                "_model_config": config,
            },
        )

        # Create a unique Config class for this converter
        config_cls = type(
            f"{converter_cls.__name__}Config",
            (Configurable.Config,),
            {"__annotations__": {}},
        )
        owner_attr = "_owner"
        setattr(config_cls, owner_attr, converter_cls)
        converter_cls.Config = config_cls

        self._converter_classes[name] = converter_cls


registry = _ConverterRegistry()
