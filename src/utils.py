# Filename: utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for the Interactive GrabCut application
"""

import numpy as np
from PIL import Image


def voc_palette() -> np.ndarray:
    """
    Generate VOC-style color palette for 256 classes.
    
    Returns:
        256x3 uint8 array of RGB colors
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


def save_indexed_png(mask: np.ndarray, path: str) -> None:
    """
    Save a segmentation mask as indexed PNG with VOC palette.
    
    Args:
        mask: HxW uint8 array with class labels 0..20
        path: Output file path
    """
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(voc_palette().ravel().tolist())
    img.save(path)


def load_indexed_png(path: str) -> np.ndarray:
    """
    Load an indexed PNG segmentation mask.
    
    Args:
        path: Input file path
    
    Returns:
        HxW uint8 array with class labels
    """
    img = Image.open(path).convert("P")
    return np.array(img, dtype=np.uint8)
