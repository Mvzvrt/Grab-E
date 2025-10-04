
# clahe_preprocess.py
# -*- coding: utf-8 -*-
"""
CLAHE preprocessing utility for GrabCut pipelines.

Exposes a single function:
    apply_clahe(img_rgb_u8, method="lab_l", clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray

Inputs:
  img_rgb_u8: H x W x 3, dtype uint8, RGB layout
  method: which channel to enhance with CLAHE
          "lab_l"     [default], convert RGB to LAB, apply CLAHE to L, convert back to RGB
          "ycbcr_y"   convert RGB to YCrCb, apply CLAHE to Y, convert back to RGB
          "hsv_v"     convert RGB to HSV, apply CLAHE to V, convert back to RGB
          "rgb"       apply CLAHE per channel directly in RGB space
  clip_limit: CLAHE clipLimit parameter
  tile_grid_size: tuple of ints [tile width, tile height]

Output:
  Enhanced RGB image, same shape and dtype.

Note:
  Keep preprocessing here, do not touch OpenCV's GrabCut internals.
  Call this right after loading the image, before any color space conversion
  or calls to cv2.grabCut. This way both the single-space and ensemble paths
  receive the same enhanced input.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2 as cv

def _ensure_u8_rgb(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise TypeError("apply_clahe expects a numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_clahe(img_rgb_u8: np.ndarray,
                method: str = "lab_l",
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to an RGB uint8 image using the chosen method.
    Returns an RGB uint8 image of the same shape.
    """
    img = _ensure_u8_rgb(img_rgb_u8)
    clahe = cv.createCLAHE(clipLimit=float(clip_limit),
                           tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])))

    m = str(method).lower()
    if m == "lab_l":
        lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        L, a, b = cv.split(lab)
        Lc = clahe.apply(L)
        out = cv.cvtColor(cv.merge([Lc, a, b]), cv.COLOR_LAB2RGB)
        return out
    elif m == "ycbcr_y":
        ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv.split(ycrcb)
        Yc = clahe.apply(Y)
        out = cv.cvtColor(cv.merge([Yc, Cr, Cb]), cv.COLOR_YCrCb2RGB)
        return out
    elif m == "hsv_v":
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        H, S, V = cv.split(hsv)
        Vc = clahe.apply(V)
        out = cv.cvtColor(cv.merge([H, S, Vc]), cv.COLOR_HSV2RGB)
        return out
    elif m == "rgb":
        r, g, b = cv.split(img)
        rc = clahe.apply(r)
        gc = clahe.apply(g)
        bc = clahe.apply(b)
        out = cv.merge([rc, gc, bc])
        return out
    else:
        raise ValueError(f"Unknown CLAHE method '{method}', expected one of ['lab_l', 'ycbcr_y', 'hsv_v', 'rgb']")
