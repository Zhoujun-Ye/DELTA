"""DELTA model configuration."""

import os
from typing import Any
from typing import Optional, Union

from transformers import CLIPConfig
from transformers.configuration_utils import PretrainedConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.common import registry

__all__ = ["DELTAConfig"]


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_python(v) for v in value)
    return value


@registry.register_model_config("DELTAConfig")
class DELTAConfig(PretrainedConfig):
    model_type = "delta"

    _TEXT_BINDING_KEYS = {
        "enabled",
        "layer_indices",
        "num_buckets",
        "max_distance",
        "fused_readout_enabled",
        "fused_readout_gate_bias_init",
        "loss_enabled",
        "loss_detach_image",
        "loss_weight",
        "loss_temperature",
    }
    _VISION_BINDING_KEYS = {
        "enabled",
        "loss_weight",
        "loss_temperature",
        "loss_detach_text",
        "object_attention_temperature",
        "attribute_attention_temperature",
        "object_prior_gamma",
    }

    def __init__(
            self,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            text_binding: Optional[dict] = None,
            vision_binding: Optional[dict] = None,
            **kwargs
    ):
        backbone_config = kwargs.pop("backbone_config", None)
        kwargs.pop("gradient_checkpointing_scope", None)
        backbone_pretrained_model_name_or_path = kwargs.pop(
            "backbone_pretrained_model_name_or_path",
            pretrained_model_name_or_path,
        )
        kwargs = _to_plain_python(kwargs)
        text_binding = _to_plain_python(text_binding)
        vision_binding = _to_plain_python(vision_binding)
        backbone_config = _to_plain_python(backbone_config)
        if text_binding is None:
            text_binding = {}
        if vision_binding is None:
            vision_binding = {}
        text_binding = {
            key: value
            for key, value in text_binding.items()
            if key in self._TEXT_BINDING_KEYS
        }
        vision_binding = {
            key: value
            for key, value in vision_binding.items()
            if key in self._VISION_BINDING_KEYS
        }

        super().__init__(**kwargs)

        if backbone_config is None:
            backbone_config_source = backbone_pretrained_model_name_or_path or pretrained_model_name_or_path
            if backbone_config_source is None:
                backbone_config = CLIPConfig().to_dict()
            else:
                backbone_config = CLIPConfig.from_pretrained(backbone_config_source).to_dict()
        if backbone_pretrained_model_name_or_path is None:
            backbone_pretrained_model_name_or_path = pretrained_model_name_or_path

        self.backbone_config = _to_plain_python(backbone_config)
        self.backbone_pretrained_model_name_or_path = backbone_pretrained_model_name_or_path
        self.text_binding = {
            "enabled": False,
            "layer_indices": [4, 5, 6, 7],
            "num_buckets": 32,
            "max_distance": 77,
            "fused_readout_enabled": True,
            "fused_readout_gate_bias_init": -2.0,
            "loss_weight": 1.0,
            "loss_temperature": 5.0,
            "loss_detach_image": True,
            "loss_enabled": True,
            **text_binding,
        }
        self.vision_binding = {
            "enabled": False,
            "loss_weight": 1.0,
            "loss_temperature": 10.0,
            "loss_detach_text": True,
            "object_attention_temperature": 0.07,
            "attribute_attention_temperature": 0.07,
            "object_prior_gamma": 1.0,
            **vision_binding,
        }

        self.initializer_factor = 1.0

    def to_dict(self):
        output = super().to_dict()
        output["backbone_config"] = _to_plain_python(self.backbone_config)
        output["backbone_pretrained_model_name_or_path"] = self.backbone_pretrained_model_name_or_path
        output["text_binding"] = _to_plain_python(self.text_binding)
        output["vision_binding"] = _to_plain_python(self.vision_binding)
        return _to_plain_python(output)

    def to_diff_dict(self):
        output = super().to_diff_dict()
        output["backbone_config"] = _to_plain_python(self.backbone_config)
        output["backbone_pretrained_model_name_or_path"] = self.backbone_pretrained_model_name_or_path
        output["text_binding"] = _to_plain_python(self.text_binding)
        output["vision_binding"] = _to_plain_python(self.vision_binding)
        return _to_plain_python(output)
