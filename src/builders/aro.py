from typing import Optional

from datasets import Dataset, load_dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder("AROCOCOOrderBuilder")
class AROCOCOOrderBuilder(BaseBuilder):
    split: Optional[str] = "test"
    name: Optional[str] = "AROCOCOOrder"

    def build_dataset(self) -> Dataset:
        return load_dataset(
            path="gowitheflow/ARO-COCO-order",
            split=self.split,
            trust_remote_code=True,
        )


@registry.register_builder("AROFlickrOrderBuilder")
class AROFlickrOrderBuilder(BaseBuilder):
    split: Optional[str] = "test"
    name: Optional[str] = "AROFlickrOrder"

    def build_dataset(self) -> Dataset:
        return load_dataset(
            path="gowitheflow/ARO-Flickr-Order",
            split=self.split,
            trust_remote_code=True,
        )
