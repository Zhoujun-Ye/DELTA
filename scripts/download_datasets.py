#!/usr/bin/env python3
"""
Pre-download all datasets used by this project into the local `dataset/` directory.

This script runs in the server's Python environment. It pins all Hugging Face
caches and the custom dataset download root to
the repository-local dataset directory.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"

# Pin every dataset-related cache to the repo-local dataset directory.
os.environ["DATASET_BASE_DIR"] = str(DATASET_DIR)
os.environ["HF_HOME"] = str(DATASET_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASET_DIR)

# Avoid the Xet/CAS download path, which is often the source of timeout issues.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Make `src` imports available when the script is run from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_datasets")


def _materialize_dataset(name: str, dataset) -> None:
    num_rows = getattr(dataset, "num_rows", None)
    if num_rows is None and hasattr(dataset, "__len__"):
        try:
            num_rows = len(dataset)
        except Exception:
            num_rows = "unknown"
    logger.info("Ready: %s (rows=%s)", name, num_rows)


def download_coco() -> None:
    logger.info("Downloading COCO + Karpathy annotations into %s", DATASET_DIR / "raw")
    from src.builders.coco import COCOCaptionsWithImageDatasetBuilder

    dataset = COCOCaptionsWithImageDatasetBuilder().build_dataset()
    _materialize_dataset("coco", dataset)


def download_eval_datasets() -> None:
    from src.builders.crepe import CrepeBuilder
    from src.builders.valse import ValseBuilder
    from src.builders.sugarcrepe import SugarCrepeBuilder

    for name, builder in (
        ("crepe", CrepeBuilder),
        ("valse", ValseBuilder),
        ("sugarcrepe", SugarCrepeBuilder),
    ):
        logger.info("Downloading %s into %s", name, DATASET_DIR / "raw" / name)
        dataset = builder().build_dataset()
        _materialize_dataset(name, dataset)

    from huggingface_hub import snapshot_download
    from src.builders.colorswap import ColorSwapBuilder

    snapshot_download(
        repo_id="stanfordnlp/colorswap",
        repo_type="dataset",
        allow_patterns=["data/test-*.parquet"],
        local_dir=str(DATASET_DIR / "colorswap"),
        max_workers=4,
    )
    _materialize_dataset("colorswap", ColorSwapBuilder().build_dataset())


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Repository root: %s", REPO_ROOT)
    logger.info("Dataset directory: %s", DATASET_DIR)
    logger.info("HF_HUB_DISABLE_XET=%s", os.environ.get("HF_HUB_DISABLE_XET"))
    logger.info("HF_HUB_ENABLE_HF_TRANSFER=%s", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))

    download_coco()
    download_eval_datasets()

    logger.info("All dataset downloads completed")


if __name__ == "__main__":
    main()
