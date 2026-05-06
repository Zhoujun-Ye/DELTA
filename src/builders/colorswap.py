import glob
import os
from typing import Optional

from datasets import Dataset, Features, Image, Value, load_dataset

from src.builders.base import BaseBuilder
from src.common import registry


_COLORSWAP_FEATURES = Features(
    {
        "id": Value("int32"),
        "image_1": Image(),
        "image_2": Image(),
        "caption_1": Value("string"),
        "caption_2": Value("string"),
        "image_source": Value("string"),
        "caption_source": Value("string"),
    }
)


@registry.register_builder("ColorSwapBuilder")
class ColorSwapBuilder(BaseBuilder):
    split: Optional[str] = "test"
    name: Optional[str] = "ColorSwap"

    @staticmethod
    def _get_local_data_files(split: str) -> list[str]:
        dataset_base_dir = os.environ.get("DATASET_BASE_DIR", "/app/dataset")
        pattern = os.path.join(dataset_base_dir, "colorswap", "data", f"{split}-*.parquet")
        return sorted(glob.glob(pattern))

    def build_dataset(self) -> Dataset:
        local_files = self._get_local_data_files(self.split)
        if local_files:
            return load_dataset(
                path="parquet",
                data_files={self.split: local_files},
                features=_COLORSWAP_FEATURES,
                split=self.split,
            )

        try:
            return load_dataset(
                path="stanfordnlp/colorswap",
                split=self.split,
                token=os.environ.get("HF_TOKEN", True),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ColorSwap. Either place the downloaded dataset under "
                "`dataset/colorswap/data/` or provide Hugging Face access via `HF_TOKEN`."
            ) from exc
