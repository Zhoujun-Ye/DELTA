from datasets import Dataset, Features, Image, Sequence, Value

from src.common import registry
from .base import BaseBuilder
from .dataset_sources import load_caption_pairs


@registry.register_builder("CrepeBuilder")
class CrepeBuilder(BaseBuilder):
    """CREPE productivity captions and Visual Genome region images."""

    split: str = "test"
    name: str = "Crepe"

    def build_dataset(self) -> Dataset:
        features = Features({
            "images": Sequence(Image()),
            "image_paths": Sequence(Value("string")),
            "positive_caption": Sequence(Value("string")),
            "negative_caption": Sequence(Value("string")),
            "original_file_name": Value("string"),
            **{key: Value("float") for key in ("x", "y", "width", "height")},
        })
        return load_caption_pairs("crepe", features, self.split)
