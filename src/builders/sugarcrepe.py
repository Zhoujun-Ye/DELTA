import os
from typing import Optional

from datasets import load_dataset, Dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder('SugarCrepeBuilder')
class SugarCrepeBuilder(BaseBuilder):
    """
    A builder class for creating the SugarCrepe dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'SugarCrepe').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'SugarCrepe'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SUGARCREPE dataset.

        Returns:
            Dataset: The SUGARCREPE dataset with sequences of text and images.
        """
        import os
        script_path = os.path.join(os.path.dirname(__file__), "sugarcrepe_vlms.py")
        dataset = load_dataset(
            path=script_path,
            trust_remote_code=True,
            split=self.split,
        )

        return dataset


@registry.register_builder('SugarCrepePPBuilder')
class SugarCrepePPBuilder(BaseBuilder):
    """
    A builder class for creating the SugarCrepe dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'SugarCrepe').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'SugarCrepe++'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SUGARCREPE dataset.

        Returns:
            Dataset: The SUGARCREPE dataset with sequences of text and images.
        """
        script_path = os.path.join(os.path.dirname(__file__), "sugarcrepepp_vlms.py")
        dataset = load_dataset(
            path=script_path,
            trust_remote_code=True,
            split=self.split,
        )

        return dataset