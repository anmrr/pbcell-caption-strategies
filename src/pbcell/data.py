"""Data utilities for resolving image directories from local paths or HF Hub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_image_dir(
    image_dir: str,
    image_dataset: Optional[str] = None,
) -> Path:
    """Resolve the image directory, downloading from HF Hub if needed.

    Args:
        image_dir: Local path to the image directory. Used directly if it
            exists or if ``image_dataset`` is not set.
        image_dataset: HF Hub dataset repo ID (e.g.
            ``"example/pbcell_images"``). When set, the dataset is
            downloaded via ``huggingface_hub.snapshot_download`` and the
            cached path is returned.

    Returns:
        Path to the local image directory (either the original or the
        downloaded cache path).
    """
    local_path = Path(image_dir)

    if not image_dataset:
        if not local_path.exists():
            raise FileNotFoundError(
                f"image_dir '{local_path}' does not exist and no "
                f"image_dataset was specified. Set image_dataset in your "
                f"YAML config to download images from HF Hub."
            )
        logger.info(f"Using local image directory: {local_path}")
        return local_path

    if local_path.exists():
        logger.info(
            f"Local image directory '{local_path}' exists, using it "
            f"(skipping HF download of '{image_dataset}')"
        )
        return local_path

    # Download from HF Hub
    logger.info(f"Downloading image dataset '{image_dataset}' from HF Hub...")

    from huggingface_hub import snapshot_download

    cached_path = snapshot_download(
        repo_id=image_dataset,
        repo_type="dataset",
    )

    logger.info(f"Dataset cached at: {cached_path}")
    return Path(cached_path)
