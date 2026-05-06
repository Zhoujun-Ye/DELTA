from transformers import CLIPModel, CLIPConfig, CLIPProcessor

from src.common.registry import registry
from .configuration_delta import DELTAConfig
from .modeling_delta import DELTAModel

__all__ = [
    "DELTAConfig",
    "DELTAModel",
    "CLIPModel",
    "CLIPConfig",
    "CLIPProcessor",
]

registry.register_model("CLIPModel")(CLIPModel)
registry.register_model_config("CLIPConfig")(CLIPConfig)
registry.register_processor("CLIPProcessor")(CLIPProcessor)
