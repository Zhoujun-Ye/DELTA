from datasets import Dataset, Features, Image, Sequence, Value

from src.common import registry
from .base import BaseBuilder
from .dataset_sources import load_caption_pairs


@registry.register_builder("ValseBuilder")
class ValseBuilder(BaseBuilder):
    """VALSE caption/foil pairs validated by the annotators."""

    split: str = "test"
    name: str = "Valse"

    def build_dataset(self) -> Dataset:
        features = Features({
            "images": Sequence(Image()),
            "image_paths": Sequence(Value("string")),
            "positive_caption": Sequence(Value("string")),
            "negative_caption": Sequence(Value("string")),
            **{key: Value("string") for key in (
                "original_file_name", "dataset", "key",
                "linguistic_phenomena", "original_split",
            )},
        })
        return load_caption_pairs("valse", features, self.split)
