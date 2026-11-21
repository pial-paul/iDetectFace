"""Bounded decoding for local enrollment and evaluation images."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 16_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_image(path: Path):
    if path.is_symlink() or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Expected a regular JPEG or PNG file.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Image exceeds 16 MiB.")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    return decode_image(raw)


def decode_image(raw: bytes, *, max_side: int | None = None):
    """Decode one bounded JPEG/PNG and optionally downscale before inference."""
    if len(raw) > MAX_BYTES:
        raise ValueError("Image exceeds 16 MiB.")
    try:
        with Image.open(BytesIO(raw), formats=("JPEG", "PNG")) as image:
            if min(image.size) < 2 or image.width * image.height > MAX_PIXELS:
                raise ValueError("Expected an image at least 2x2 and at most 16 megapixels.")
            if max_side is not None:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Image.DecompressionBombError as error:
        raise ValueError("Image exceeds the pixel limit.") from error
    return rgb, hashlib.sha256(raw).digest()
