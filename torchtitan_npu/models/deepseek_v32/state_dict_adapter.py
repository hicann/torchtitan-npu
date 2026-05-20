# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader

from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter

from torchtitan_npu.tools.weight_utils import detect_input_format_by_path

logger = logging.getLogger(__name__)


def _first_moe_layer(model_config):
    """Return the first ``layer.moe`` config, or ``None`` if all dense."""
    return next(
        (layer.moe for layer in model_config.layers if layer.moe is not None),
        None,
    )


def _first_moe_index(model_config) -> int:
    """Index of the first MoE layer (== count of leading dense layers).

    Mirrors the legacy ``model_args.moe_args.first_k_dense`` semantics from
    the pre-Config era. Returns the total layer count if every layer is dense.
    """
    return next(
        (i for i, layer in enumerate(model_config.layers) if layer.moe is not None),
        len(model_config.layers),
    )


class DeepSeekV32StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)

        # key mapping
        self._setup_v32_mappings(model_config)

        # MoE config knobs derived from the Config tree.
        moe = _first_moe_layer(model_config)
        if moe is not None:
            self.use_gmm = getattr(moe.experts, "use_grouped_mm", False)
            self.n_experts = moe.experts.num_experts
        else:
            self.use_gmm = False
            self.n_experts = 0
        self.first_k_dense = _first_moe_index(model_config)
        self._input_format = "hf"
        self._input_expert_format = "standard"

        # apply checkpoint patch
        self._setup_checkpoint_patch(model_config)

    # pyrefly: ignore [bad-override]
    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        self._input_format = detect_input_format_by_path(path)

        if self._input_format == "hf":
            return HuggingFaceStorageReader(path)
        else:
            from torch.distributed.checkpoint import FileSystemReader

            return FileSystemReader(path)

    def _setup_checkpoint_patch(self, model_config):
        """setup checkpoint save patch"""
        try:
            from ...tools import checkpoint_patch

            checkpoint_patch.configure_from_model_args(model_config, adapter=self)

            if checkpoint_patch.is_enabled():
                success = checkpoint_patch.apply_patch()
                if success:
                    logger.info(
                        "Checkpoint save patch initialized from StateDict Adaptor"
                    )

        except Exception as e:
            logger.error(
                f"Failed to setup checkpoint patch, training will continue with original "
                f"saving configs: {e}"
            )

    def _setup_v32_mappings(self, model_config):
        """
        Setup key maps for DeepSeek V3.2 with attention split to
        pre-attention, inner_attention, and post_attention.
        """
        self._setup_attention_q_mappings()
        self._setup_indexer_mappings()
        self._setup_attention_kvo_mappings()

        # MTP
        if model_config.num_mtp_modules > 0:
            self.from_hf_map.update(
                {
                    "model.layers.{}.enorm.weight": "layers.{}.enorm.weight",
                    "model.layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
                    "model.layers.{}.eh_proj.weight": "layers.{}.eh_proj.weight",
                }
            )

    def _setup_attention_q_mappings(self):
        self.from_hf_map.pop("model.layers.{}.self_attn.q_proj.weight", None)
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.q_a_proj.weight": (
                    "layers.{}.attention.pre_attention.wq_a.weight"
                ),
                "model.layers.{}.self_attn.q_a_layernorm.weight": (
                    "layers.{}.attention.pre_attention.q_norm.weight"
                ),
                "model.layers.{}.self_attn.q_b_proj.weight": (
                    "layers.{}.attention.pre_attention.wq_b.weight"
                ),
            }
        )

    def _setup_indexer_mappings(self):
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.indexer.wq_b.weight": (
                    "layers.{}.attention.pre_attention.indexer.wq_b.weight"
                ),
                "model.layers.{}.self_attn.indexer.wk.weight": (
                    "layers.{}.attention.pre_attention.indexer.wk.weight"
                ),
                "model.layers.{}.self_attn.indexer.k_norm.weight": (
                    "layers.{}.attention.pre_attention.indexer.k_norm.weight"
                ),
                "model.layers.{}.self_attn.indexer.k_norm.bias": (
                    "layers.{}.attention.pre_attention.indexer.k_norm.bias"
                ),
                "model.layers.{}.self_attn.indexer.weights_proj.weight": (
                    "layers.{}.attention.pre_attention.indexer.weights_proj.weight"
                ),
            }
        )

    def _setup_attention_kvo_mappings(self):
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": (
                    "layers.{}.attention.pre_attention.wkv_a.weight"
                ),
                "model.layers.{}.self_attn.kv_a_layernorm.weight": (
                    "layers.{}.attention.pre_attention.kv_norm.weight"
                ),
                "model.layers.{}.self_attn.kv_b_proj.weight": (
                    "layers.{}.attention.pre_attention.wkv_b.weight"
                ),
                "model.layers.{}.self_attn.o_proj.weight": (
                    "layers.{}.attention.post_attention.wo.weight"
                ),
            }
        )
