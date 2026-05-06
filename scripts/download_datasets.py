#!/usr/bin/env python3
"""
Pre-download all datasets used by this project into the local `dataset/` directory.

This script is intended for non-Docker environments such as a conda setup on a
server. It pins all Hugging Face caches and the custom dataset download root to
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

from datasets import concatenate_datasets, load_dataset  # noqa: E402
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
    logger.info("Downloading COCO raw images + examples into %s", DATASET_DIR / "coco_raw_data")
    script_path = REPO_ROOT / "src" / "builders" / "coco-karpathy-with-image.py"
    dataset = load_dataset(
        str(script_path),
        trust_remote_code=True,
        split="train",
    )
    _materialize_dataset("coco_raw_data", dataset)


def download_coco_metadata() -> None:
    logger.info("Downloading optional COCO captions metadata from Hugging Face")
    dataset = concatenate_datasets(
        load_dataset(
            "yerevann/coco-karpathy",
            trust_remote_code=True,
            split=["train", "restval"],
        )
    )
    _materialize_dataset("coco-karpathy train+restval", dataset)


def download_eval_datasets() -> None:
    custom_scripts = [
        ("crepe_vlms", "test", REPO_ROOT / "src" / "builders" / "crepe_vlms.py"),
        ("valse_vlms", "test", REPO_ROOT / "src" / "builders" / "valse_vlms.py"),
        ("sugarcrepe_vlms", "test", REPO_ROOT / "src" / "builders" / "sugarcrepe_vlms.py"),
        ("sugarcrepepp_vlms", "test", REPO_ROOT / "src" / "builders" / "sugarcrepepp_vlms.py"),
    ]
    for name, split, script_path in custom_scripts:
        logger.info("Downloading %s into %s", name, DATASET_DIR / name)
        dataset = load_dataset(
            str(script_path),
            trust_remote_code=True,
            split=split,
        )
        _materialize_dataset(name, dataset)

    hf_datasets = [
        ("aro_visual_attribution", "gowitheflow/ARO-Visual-Attribution", "test"),
        ("aro_coco_order", "gowitheflow/ARO-COCO-order", "test"),
        ("aro_flickr_order", "gowitheflow/ARO-Flickr-Order", "test"),
    ]
    for name, path, split in hf_datasets:
        logger.info("Downloading %s via Hugging Face cache into %s", name, DATASET_DIR)
        dataset = load_dataset(
            path=path,
            split=split,
            trust_remote_code=True,
        )
        _materialize_dataset(name, dataset)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("Downloading winoground via Hugging Face cache into %s", DATASET_DIR)
        dataset = load_dataset(
            path="facebook/winoground",
            split="test",
            trust_remote_code=True,
            token=hf_token,
        )
        _materialize_dataset("winoground", dataset)
    else:
        logger.info("Skipping winoground predownload because HF_TOKEN is not set")


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
