
# ms_features_preprocess.py
# -*- coding: utf-8 -*-
"""
Multiscale feature builders paired with the pipeline's color spaces.

Default behavior expected by the main pipeline:
  1) Convert the RGB image to the selected color space.
  2) Build a three-channel feature image [luminance_like, grad_sigma1, grad_sigma2].
  3) Pass this feature image to cv2.grabCut, unchanged.

Public API
  ms_feature_from_space(img_rgb_u8, mode, sigmas=(1.0, 2.0, 4.0), scale_each=True) -> np.ndarray
      Converts using color_space.convert_color_space, then derives luminance from that space and
      builds multiscale gradient features.
  ms_feature_from_features(feats_u8, mode, sigmas=(1.0, 2.0, 4.0), scale_each=True) -> np.ndarray
      Same as above but takes the already-converted HxWx3 uint8 features for 'mode'.
  ms_feature_from_luminance(luma_u8, sigmas=(1.0, 2.0, 4.0), scale_each=True) -> np.ndarray
      Builds features from a single-channel luminance input directly.

Channel policy for luminance_like, based on color_space.py:
  rgb            -> compute gray from RGB  [OpenCV gray]
  hsv_conic      -> channel 0  [V as C0]
  cielab         -> channel 0  [L]
  c02_scd        -> channel 0  [J]
  c16_scd        -> channel 0  [J']
  oklab          -> channel 0  [L]
  oklch          -> channel 0  [L]
  jzazbz         -> channel 0  [Jz]
  jzczhz         -> channel 0  [Jz]
  ictcp_pq       -> channel 0  [I]
  xyz            -> channel 1  [Y]
  ycbcr_bt709    -> channel 0  [Y']
  srgb_linear    -> compute Y = 0.2126 R + 0.7152 G + 0.0722 B on linear 8-bit

All functions return HxWx3 uint8 ready for cv2.grabCut.
"""
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import cv2 as cv

# import the project's color space converter
from color_space import convert_color_space

# ------------------------ helpers ------------------------

def _ensure_u8_hwc3(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError("expected a numpy array")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _ensure_u8_gray(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError("expected a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"expected HxW array for luminance, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _normalize_to_u8(arr: np.ndarray, clip_low_high: Tuple[float, float] | None = None) -> np.ndarray:
    a = arr.astype(np.float32)
    if clip_low_high is not None:
        lo, hi = clip_low_high
        a = np.clip(a, float(lo), float(hi))
    mn = float(a.min())
    mx = float(a.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) * (255.0 / (mx - mn))
    return a.astype(np.uint8)

def _gradmag_at_sigma(base_u8: np.ndarray, sigma: float) -> np.ndarray:
    # Gaussian smoothing then Sobel magnitude
    if sigma <= 0:
        blur = base_u8
    else:
        k = int(round(6.0 * float(sigma) + 1))
        if k % 2 == 0:
            k += 1
        blur = cv.GaussianBlur(base_u8, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    return mag  # float32

# ------------------------ luminance rules ------------------------

def _luminance_from_features(feats_u8: np.ndarray, mode: str) -> np.ndarray:
    mode_l = str(mode).lower()
    H, W, _ = feats_u8.shape

    if mode_l == "rgb":
        # convert RGB 8-bit to gray via OpenCV
        return cv.cvtColor(feats_u8, cv.COLOR_RGB2GRAY)

    if mode_l == "hsv_conic":
        # channel 0 holds V scaled to 0..255 in our HSV conic mapping
        return feats_u8[:, :, 0]

    if mode_l in {"cielab", "c02_scd", "c16_scd", "oklab", "oklch",
                  "jzazbz", "jzczhz", "ictcp_pq", "ycbcr_bt709"}:
        # These modes place a luminance-like value in channel 0
        return feats_u8[:, :, 0]

    if mode_l == "xyz":
        # Y is channel 1 in XYZ
        return feats_u8[:, :, 1]

    if mode_l == "srgb_linear":
        # Compute Y from linear 8-bit channels using BT.709 coefficients
        r = feats_u8[:, :, 0].astype(np.float32) / 255.0
        g = feats_u8[:, :, 1].astype(np.float32) / 255.0
        b = feats_u8[:, :, 2].astype(np.float32) / 255.0
        Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.clip(Y * 255.0, 0, 255).astype(np.uint8)

    # Fallback, be conservative: gray from the triplet
    return cv.cvtColor(feats_u8, cv.COLOR_RGB2GRAY)

# ------------------------ public builders ------------------------

def ms_feature_from_luminance(luma_u8: np.ndarray,
                              sigmas: Iterable[float] = (1.0, 2.0, 4.0),
                              scale_each: bool = True) -> np.ndarray:
    """
    Build [luma, grad_sigma1, grad_sigma2] from a single-channel luminance image.
    Returns HxWx3 uint8.
    """
    base = _ensure_u8_gray(luma_u8)

    # choose two smallest non-negative sigmas
    sig = sorted([float(s) for s in sigmas if float(s) >= 0.0])
    if len(sig) == 0:
        sig = [1.0, 2.0]
    if len(sig) == 1:
        sig = [sig[0], max(sig[0] * 2.0, 1.0)]
    s1, s2 = sig[0], sig[1]

    m1 = _gradmag_at_sigma(base, s1)
    m2 = _gradmag_at_sigma(base, s2)

    if scale_each:
        ch0 = base.copy()
        ch1 = _normalize_to_u8(m1)
        ch2 = _normalize_to_u8(m2)
    else:
        both = np.stack([m1, m2], axis=2)
        lo = float(np.percentile(both, 0.5))
        hi = float(np.percentile(both, 99.5))
        ch0 = base.copy()
        ch1 = _normalize_to_u8(m1, (lo, hi))
        ch2 = _normalize_to_u8(m2, (lo, hi))

    return cv.merge([ch0, ch1, ch2])

def ms_feature_from_features(feats_u8: np.ndarray,
                             mode: str,
                             sigmas: Iterable[float] = (1.0, 2.0, 4.0),
                             scale_each: bool = True) -> np.ndarray:
    """
    Build [luma_like, grad_sigma1, grad_sigma2] from already converted HxWx3 uint8 features for 'mode'.
    """
    feats = _ensure_u8_hwc3(feats_u8)
    luma = _luminance_from_features(feats, mode)
    return ms_feature_from_luminance(luma, sigmas=sigmas, scale_each=scale_each)

def ms_feature_from_space(img_rgb_u8: np.ndarray,
                          mode: str,
                          sigmas: Iterable[float] = (1.0, 2.0, 4.0),
                          scale_each: bool = True) -> np.ndarray:
    """
    Convert RGB to 'mode' using the project's converter, then build the multiscale feature image.
    """
    feats = convert_color_space(img_rgb_u8, mode)
    return ms_feature_from_features(feats, mode, sigmas=sigmas, scale_each=scale_each)
