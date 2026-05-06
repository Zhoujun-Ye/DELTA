"""
Shared download utilities for dataset builders.

Provides:
- Local-file-first checking to avoid re-downloading
- Resumable downloads
- Single-threaded download with resume support
"""

import os
import logging
import zipfile
from typing import Dict, Optional
import requests
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args, **kwargs): pass
        @staticmethod
        def write(s, *args, **kwargs): print(s)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 8 * 1024 * 1024       # 8 MB per read chunk
CONNECT_TIMEOUT = 30               # seconds
READ_TIMEOUT = 300                 # seconds

# Base directory where pre-downloaded datasets live
DATASET_BASE_DIR = os.environ.get("DATASET_BASE_DIR", "/app/dataset")


def _dataset_local_dir(dataset_name: str) -> str:
    """Return the conventional local directory for a dataset."""
    return os.path.join(DATASET_BASE_DIR, dataset_name)


def _head_content_length(url: str) -> Optional[int]:
    """Get the Content-Length from a HEAD request (returns None if unavailable)."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=CONNECT_TIMEOUT)
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def _download_single_thread_resume(url: str, dest_path: str):
    """
    Single-threaded download with resume support.
    If *dest_path* already exists and is partially written, the download
    resumes from where it left off.
    """
    existing_size = 0
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info(f"  Resuming download from byte {existing_size}: {url}")

    with requests.get(url, headers=headers, stream=True,
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
        # 416 = Range Not Satisfiable: file already completely downloaded.
        if r.status_code == 416:
            logger.info(f"  File already fully downloaded: {dest_path}")
            return
        r.raise_for_status()

        mode = "ab" if existing_size > 0 and r.status_code == 206 else "wb"
        total = r.headers.get("Content-Length")
        total_len = int(total) if total else None
        
        desc = f"Downloading {os.path.basename(dest_path)}"
        with tqdm(total=total_len, unit='B', unit_scale=True, desc=desc, leave=False) as pbar:
            with open(dest_path, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    logger.info(f"  Download completed: {os.path.basename(dest_path)}")

def fast_download(url: str, dest_path: str):
    """
    Download *url* to *dest_path* as fast as possible.

    Strategy:
    1. If *dest_path* already exists and looks complete, skip.
    2. Use single-threaded download with resume support.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check if already fully downloaded
    total_size = _head_content_length(url)
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if total_size and local_size >= total_size:
            logger.info(f"  Already downloaded, skipping: {os.path.basename(dest_path)}")
            return
        elif local_size > 0 and total_size is None:
            # Cannot verify size; assume complete.
            logger.info(f"  File exists (size unknown from server), assuming complete: "
                        f"{os.path.basename(dest_path)}")
            return

    _download_single_thread_resume(url, dest_path)


def _extract_zip(zip_path: str, extract_dir: str):
    """Extract a zip archive if not already extracted."""
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        logger.info(f"  Already extracted, skipping: {extract_dir}")
        return extract_dir
    logger.info(f"  Extracting: {os.path.basename(zip_path)} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        desc = f"Extracting {os.path.basename(zip_path)}"
        with tqdm(iterable=members, total=len(members), desc=desc, leave=False) as pbar:
            for member in pbar:
                zf.extract(member, extract_dir)
    logger.info(f"  Extraction completed: {extract_dir}")
    return extract_dir


def resolve_dataset_files(
    dataset_name: str,
    files: Dict[str, str],
    dl_manager=None,
    extract_zips: bool = True,
) -> Dict[str, str]:
    """
    Resolve dataset files: prefer local copies, download if missing, extract zips.

    Parameters
    ----------
    dataset_name : str
        Subfolder name under DATASET_BASE_DIR (e.g. "crepe_vlms").
    files : dict[str, str]
        Mapping of logical name to remote URL.
        Example: {"images": "https://...images.zip", "examples": "https://...examples.jsonl"}
    dl_manager : optional
        HuggingFace DownloadManager (unused now, kept for API compat).
    extract_zips : bool
        If True, automatically extract .zip files after download.

    Returns
    -------
    dict[str, str]
        Mapping of logical name to local file/directory path.
    """
    local_dir = _dataset_local_dir(dataset_name)
    os.makedirs(local_dir, exist_ok=True)

    result = {}
    for key, url in files.items():
        filename = os.path.basename(url.split("?")[0])  # strip query params
        local_path = os.path.join(local_dir, filename)

        # Download if missing / incomplete
        fast_download(url, local_path)

        # Extract zip files
        if extract_zips and filename.endswith(".zip"):
            extract_dir = os.path.join(local_dir, filename.replace(".zip", ""))
            _extract_zip(local_path, extract_dir)
            result[key] = extract_dir
        else:
            result[key] = local_path

    return result
