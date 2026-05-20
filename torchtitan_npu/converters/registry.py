# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from .base_converter import BaseConverter


@dataclass
class PatchInfo:
    name: str
    patch_cls: type["BaseConverter"]
    supported_models: set[str] = field(default_factory=lambda: {"*"})


class ConverterRegistry:
    _instance = None
    _patches: dict[str, PatchInfo]
    _converter_classes: dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._patches = {}
            cls._instance._converter_classes = {}
        return cls._instance

    def register(self, name: str, supported_models: set[str] | None = None):
        def decorator(patch_cls: type["BaseConverter"]):
            from .npu_converter import NPUConverter

            models = supported_models
            if models is None:
                models = getattr(patch_cls, "SUPPORTED_MODELS", {"*"})

            # Create a unique converter class with its own Config
            converter_cls = type(
                f"{patch_cls.__name__}Converter",
                (NPUConverter,),
                {
                    "_patch_cls": patch_cls,
                    "_patch_name": name,
                    "_supported_models": models,
                },
            )

            # Create a unique Config class for this converter
            config_cls = type(
                f"{patch_cls.__name__}Config",
                (Configurable.Config,),
                {"__annotations__": {}},
            )
            owner_attr = "_owner"
            setattr(config_cls, owner_attr, converter_cls)
            converter_cls.Config = config_cls

            self._converter_classes[name] = converter_cls
            self._patches[name] = PatchInfo(
                name=name,
                patch_cls=patch_cls,
                supported_models=models,
            )

            return patch_cls

        return decorator

    def get_config(self, name: str) -> Configurable.Config | None:
        converter_cls = self._converter_classes.get(name)
        if converter_cls is None:
            return None
        return converter_cls.Config()

    def get(self, name: str) -> PatchInfo | None:
        return self._patches.get(name)


registry = ConverterRegistry()


def register_npu_converter(name: str, supported_models: set[str] | None = None):
    return registry.register(name, supported_models)


def get_npu_converter_config(name: str) -> Configurable.Config | None:
    return registry.get_config(name)


def has_npu_converter(converters: list, name: str) -> bool:
    """Return True if ``converters`` contains an NPU converter Config registered under ``name``.

    Recognises both registries:
      * Legacy ``ConverterRegistry``: dynamically-generated converter class
        carries ``_patch_name``; its ``Config`` carries ``_owner`` pointing
        back at the converter class.
      * New ``ModelCustomConfig`` registry (``npu_registry.py``): the
        dynamically-generated converter class carries ``_model_config``
        whose ``.name`` is the registered patch name; the ``Config`` again
        carries ``_owner`` pointing at that converter class.

    ``converters`` contains Config instances; hop through ``_owner`` to read
    the name. Also accept the raw converter class on the off chance the list
    ever contains them.
    """
    for c in converters:
        owner = getattr(c, "_owner", None) or c
        # Legacy registry
        if getattr(owner, "_patch_name", None) == name:
            return True
        # New ModelCustomConfig registry
        model_config = getattr(owner, "_model_config", None)
        if model_config is not None and getattr(model_config, "name", None) == name:
            return True
    return False
