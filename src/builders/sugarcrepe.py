from datasets import Dataset, Features, Image, Sequence, Value

from src.common import registry
from .base import BaseBuilder
from .dataset_sources import load_caption_pairs


@registry.register_builder("SugarCrepeBuilder")
class SugarCrepeBuilder(BaseBuilder):
    """SugarCrepe's seven caption categories paired with COCO val2017."""

    split: str = "test"
    name: str = "SugarCrepe"

    def build_dataset(self) -> Dataset:
        features = Features({
            "images": Sequence(Image()),
            "image_paths": Sequence(Value("string")),
            "positive_caption": Sequence(Value("string")),
            "negative_caption": Sequence(Value("string")),
            "original_file_name": Value("string"),
            "category": Value("string"),
        })
        return load_caption_pairs("sugarcrepe", features, self.split)
