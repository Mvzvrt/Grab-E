# Filename: io_utils.py
# -*- coding: utf-8 -*-
"""
I/O Utilities for GrabCut Batch Processing

This module provides utility functions for loading, saving, and processing
image and annotation files used in the GrabCut segmentation pipeline.

Functions:
    voc_palette: Generate the standard PASCAL VOC color palette.
    save_indexed_png: Save a 2D mask as an indexed PNG with VOC palette.
    load_img: Load an RGB image as a numpy array.
    load_anns: Load annotation masks from various file formats.
    find_image: Locate an image file by base name across supported extensions.
    base_from_ann_name: Extract the base image name from an annotation filename.

Constants:
    NUM_VOC_CLASSES: Number of classes in PASCAL VOC (21).
    IMG_EXTS: Supported image file extensions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------- Constants ----------

NUM_VOC_CLASSES: int = 21
"""Number of classes in the PASCAL VOC segmentation dataset (including background)."""

IMG_EXTS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
"""Supported image file extensions for input images."""


# ---------- Palette Functions ----------

def voc_palette() -> np.ndarray:
    """
    Generate the standard PASCAL VOC color palette.
    
    The VOC palette uses a bit-interleaving scheme to generate visually
    distinct colors for each class index. This ensures adjacent class
    indices have contrasting colors for better visualization.
    
    Returns:
        np.ndarray: A (256, 3) uint8 array containing RGB values for each
            palette index.
    
    Example:
        >>> palette = voc_palette()
        >>> palette[1]  # Class 1 color (typically red-ish)
        array([128,   0,   0], dtype=uint8)
    """
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lab = i
        for j in range(8):
            pal[i, 0] |= (((lab >> 0) & 1) << (7 - j))
            pal[i, 1] |= (((lab >> 1) & 1) << (7 - j))
            pal[i, 2] |= (((lab >> 2) & 1) << (7 - j))
            lab >>= 3
    return pal


def save_indexed_png(mask_2d: np.ndarray, path: str) -> None:
    """
    Save a 2D segmentation mask as an indexed PNG with VOC palette.
    
    Converts the input mask to an 8-bit indexed image and applies the
    standard PASCAL VOC color palette for visualization.
    
    Args:
        mask_2d: A 2D numpy array containing class indices (0-255).
        path: Output file path for the PNG image.
    
    Note:
        The mask values are cast to uint8, so values outside [0, 255]
        will be truncated.
    """
    img = Image.fromarray(mask_2d.astype(np.uint8))
    img = img.convert("P")
    img.putpalette(voc_palette().ravel())
    img.save(path)


# ---------- Image Loading Functions ----------

def load_img(p: Path) -> np.ndarray:
    """
    Load an image file as an RGB uint8 numpy array.
    
    Uses PIL for loading and automatic conversion to RGB format.
    
    Args:
        p: Path to the image file.
    
    Returns:
        np.ndarray: A (H, W, 3) uint8 array in RGB channel order.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.
    """
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def load_anns(p: Path) -> np.ndarray:
    """
    Load an annotation mask from various file formats.
    
    Supports .npy (numpy), .png, .bmp, .tif, and .tiff formats.
    For .npy files, uses memory-mapped loading for efficiency.
    Includes bounds checking for expected VOC class range [0, 21].
    
    Args:
        p: Path to the annotation file.
    
    Returns:
        np.ndarray: A 2D int32 array containing class indices.
    
    Raises:
        ValueError: If the file format is not supported.
    
    Warns:
        UserWarning: If annotation values are outside [0, NUM_VOC_CLASSES].
    """
    ext = p.suffix.lower()
    if ext == ".npy":
        # Memory-mapped loading for large arrays
        a = np.load(p, mmap_mode="r")
        a = np.asarray(a, dtype=np.int32)
    elif ext in (".png", ".bmp", ".tif", ".tiff"):
        a = np.asarray(Image.open(p).convert("P"), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported annotation format: {ext}")

    a_min = int(a.min()) if a.size else 0
    a_max = int(a.max()) if a.size else 0
    if a_min < 0 or a_max > NUM_VOC_CLASSES:
        warnings.warn(
            f"Annotation values out of expected range [0, {NUM_VOC_CLASSES}]: [{a_min}, {a_max}]"
        )
    return a


# ---------- File Discovery Functions ----------

def find_image(base: str, images_dir: Path) -> Optional[Path]:
    """
    Find an image file by base name across supported extensions.
    
    Searches the specified directory for an image file matching the
    base name with any of the supported extensions.
    
    Args:
        base: The base filename without extension.
        images_dir: Directory to search for the image.
    
    Returns:
        Optional[Path]: Path to the found image, or None if not found.
    
    Example:
        >>> find_image("2007_000032", Path("./images"))
        PosixPath('./images/2007_000032.jpg')
    """
    for e in IMG_EXTS:
        q = images_dir / f"{base}{e}"
        if q.exists():
            return q
    return None


def base_from_ann_name(name: str) -> str:
    """
    Extract the base image name from an annotation filename.
    
    Removes common annotation suffixes to recover the original image
    base name for matching with source images.
    
    Args:
        name: The annotation filename (without extension).
    
    Returns:
        str: The base image name with annotation suffixes removed.
    
    Example:
        >>> base_from_ann_name("2007_000032_anns_scribbleids")
        '2007_000032'
        >>> base_from_ann_name("image_001_anns")
        'image_001'
    """
    for sfx in ("_anns_scribbleids", "_scribbleids", "_anns"):
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name
