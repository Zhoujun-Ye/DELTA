import json
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset, Features, Image, Sequence, Value

from src.common import registry
from .base import (
    BaseBuilder
)
from .dataset_sources import load_coco_metadata, prepare_dataset

__all__ = [
    "COCOCaptionsDatasetBuilder",
    "COCOCaptionsWithImageDatasetBuilder",
]


def _image_examples(examples_file):
    with open(examples_file, encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            image_path = sample.pop("filename")
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
            yield {**sample, "images": image_path}


@registry.register_builder('COCOCaptionsDatasetBuilder')
class COCOCaptionsDatasetBuilder(BaseBuilder):
    """
    A builder class for creating a dataset for COCO captions with non-iterable format.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: Union[str, List[str]] = ['train', 'restval']
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset.

        Returns:
            Dataset: The COCO captions dataset.
        """
        splits = self.split if isinstance(self.split, list) else [self.split]
        if not set(splits) <= {"train", "restval", "val", "test"}:
            raise ValueError(f"Unsupported Karpathy split: {self.split}")
        return Dataset.from_list([
            {
                "images": f"http://images.cocodataset.org/{sample['filepath']}/{sample['filename']}",
                "text": [sentence["raw"] for sentence in sample["sentences"]],
            }
            for sample in load_coco_metadata()["images"]
            if sample["split"] in splits
        ])


@registry.register_builder('COCOCaptionsWithImageDatasetBuilder')
class COCOCaptionsWithImageDatasetBuilder(BaseBuilder):
    """
    A builder class for creating a dataset for COCO captions with non-iterable format.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: str = 'train'
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset.

        Returns:
            Dataset: The COCO captions dataset.
        """
        if self.split != "train":
            raise ValueError("COCO image training data only supports split='train'")
        files = prepare_dataset("coco")
        return Dataset.from_generator(
            _image_examples,
            gen_kwargs={"examples_file": files["examples"]},
            features=Features({
                "images": Image(),
                "sentences": Sequence(Value("string")),
                "cocoid": Value("string"),
            }),
            split=self.split,
        )
