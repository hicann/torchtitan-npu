# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import torch.nn as nn

from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.multimodal import DenseMaskSDPA
from torchtitan_npu.models.vlm.model import Llama3Siglip2TransformerNpu


class NpuVLMConverter(ModelCustomConverter):
    """Validate the VLM NPU config result without mutating the model.

    NPU-specific attention modules are selected by ``to_npu_vlm_config`` during
    model config conversion. This converter remains as the explicit ``npu_vlm``
    feature gate and only verifies that the built model matches that contract.
    ``self.model_name`` is set by ``ModelCustomConverter`` from the Trainer
    ``ModelSpec`` passed through ``ModelCustomConfigConverter`` before
    ``convert`` is called.
    """

    def convert(self, model: nn.Module) -> None:
        if self.model_name != "vlm":
            raise ValueError(
                f"npu_vlm converter only supports model 'vlm', got {self.model_name!r}."
            )
        if not isinstance(model, Llama3Siglip2TransformerNpu):
            raise TypeError(
                "npu_vlm converter requires Llama3Siglip2TransformerNpu. "
                f"Got {type(model).__name__}."
            )

        decoder_layers = cast(Any, model.layers)
        if not all(
            isinstance(layer.attention.inner_attention, DenseMaskSDPA)
            for layer in decoder_layers.values()
        ):
            raise TypeError("npu_vlm requires decoder DenseMaskSDPA attention.")

        encoder_layers = cast(Any, model.encoder.layers)
        if not all(
            isinstance(layer.self_attn.inner_attention, DenseMaskSDPA)
            for layer in encoder_layers.values()
        ):
            raise TypeError("npu_vlm requires encoder DenseMaskSDPA attention.")


@register_model_converter("npu_vlm")
class VLMModelConfig(ModelCustomConfig):
    model_converter = NpuVLMConverter
