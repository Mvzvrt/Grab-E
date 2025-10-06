# doe_grabcut.py
# -*- coding: utf-8 -*-
"""
Stronger DOE for GrabCut, color-space aware and seed-adaptive.

This module keeps the same public API, but upgrades the internals:
  - compute_doe_map(...): can use RGB or a provided 3-channel feature image.
  - select_roi_from_doe(...): adds optional expansion to guarantee seed coverage
    and a minimum ROI area fraction.
  - doe_limit_seeds(...): plugs the above and adapts thresholds when ROI is too small.

Design [Unverified]:
  Based on the paper's idea of density of edges as a gate before GrabCut,
  we replace plain grayscale edges with multi-channel Scharr magnitude, Laplacian-of-Gaussian,
  and multi-scale pooling, with per-channel max fusion and robust normalization.
"""

from __future__ import annotations
from typing import Tuple, Optional, Literal
import numpy as np
import cv2 as cv

EdgeMethod = Literal["auto", "canny", "scharr", "sobel", "log"]


def _ensure_u8_3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8, got {x.shape}")
    return x


def _scharr_mag(img_u8: np.ndarray) -> np.ndarray:
    # Per channel Scharr gradient magnitude, max-reduce over channels
    gx0 = cv.Scharr(img_u8[:,:,0], cv.CV_32F, 1, 0); gy0 = cv.Scharr(img_u8[:,:,0], cv.CV_32F, 0, 1)
    gx1 = cv.Scharr(img_u8[:,:,1], cv.CV_32F, 1, 0); gy1 = cv.Scharr(img_u8[:,:,1], cv.CV_32F, 0, 1)
    gx2 = cv.Scharr(img_u8[:,:,2], cv.CV_32F, 1, 0); gy2 = cv.Scharr(img_u8[:,:,2], cv.CV_32F, 0, 1)
    m0 = cv.magnitude(gx0, gy0)
    m1 = cv.magnitude(gx1, gy1)
    m2 = cv.magnitude(gx2, gy2)
    return np.maximum(np.maximum(m0, m1), m2)


def _sobel_mag(img_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(int(ksize), 3)
    gx = [cv.Sobel(img_u8[:,:,i], cv.CV_32F, 1, 0, ksize=k) for i in range(3)]
    gy = [cv.Sobel(img_u8[:,:,i], cv.CV_32F, 0, 1, ksize=k) for i in range(3)]
    mags = [cv.magnitude(gx[i], gy[i]) for i in range(3)]
    return np.maximum(np.maximum(mags[0], mags[1]), mags[2])


def _log_edge(img_u8: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    # Laplacian of Gaussian per channel then max reduce
    k = int(round(6 * sigma + 1)) | 1
    g = cv.GaussianBlur(img_u8, (k, k), sigmaX=sigma, sigmaY=sigma)
    laps = [cv.Laplacian(g[:,:,i], cv.CV_32F, ksize=3) for i in range(3)]
    return np.maximum(np.maximum(np.abs(laps[0]), np.abs(laps[1])), np.abs(laps[2]))


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        y = (x - mn) / (mx - mn)
    else:
        y = np.zeros_like(x, dtype=np.float32)
    # light contrast stretch
    p1, p99 = np.percentile(y, [1, 99])
    if p99 > p1:
        y = np.clip((y - p1) / (p99 - p1), 0.0, 1.0)
    return y.astype(np.float32)


def _local_density01(edge01: np.ndarray, window: int) -> np.ndarray:
    w = max(int(window) | 1, 3)
    dsum = cv.boxFilter(edge01.astype(np.float32), ddepth=-1, ksize=(w, w), normalize=False, borderType=cv.BORDER_REPLICATE)
    return np.clip(dsum / float(w * w), 0.0, 1.0).astype(np.float32)


def compute_doe_map(
    img_rgb_u8: np.ndarray,
    *,
    feats_u8: Optional[np.ndarray] = None,
    median_ksize: int = 5,
    edge: EdgeMethod = "auto",
    canny_low: int = 60,
    canny_high: int = 180,
    sobel_ksize: int = 3,
    log_sigma: float = 1.2,
    window: int = 41,
    multiscale: Tuple[int, int] = (21, 61),
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a stronger DOE map.
    - If feats_u8 is given, use it [HxWx3 uint8] as the source for edges,
      else use the RGB image.
    - Edge method 'auto' computes channel Scharr magnitude, channel LoG, and
      takes a robust max of both. Other methods kept for compatibility.
    - Multi scale pooling, min scale and max scale windows, then take max.

    Returns float32 HxW in 0 to 1 when normalize is True.
    """
    src = _ensure_u8_3(feats_u8 if feats_u8 is not None else img_rgb_u8)
    k = max(int(median_ksize) | 1, 3)
    src_med = cv.medianBlur(src, k)

    if edge == "auto":
        m1 = _scharr_mag(src_med)
        m2 = _log_edge(src_med, sigma=float(log_sigma))
        mag = np.maximum(m1, m2)
        e01 = (_normalize01(mag) >= 0.25).astype(np.float32)  # soft fixed binarization
    elif edge == "canny":
        # Use luminance for canny, still from src to honor feats
        g = cv.cvtColor(src_med, cv.COLOR_RGB2GRAY)
        e = cv.Canny(g, int(canny_low), int(canny_high))
        e01 = (e > 0).astype(np.float32)
    elif edge == "scharr":
        e01 = (_normalize01(_scharr_mag(src_med)) >= 0.25).astype(np.float32)
    elif edge == "sobel":
        e01 = (_normalize01(_sobel_mag(src_med, ksize=int(sobel_ksize))) >= 0.25).astype(np.float32)
    elif edge == "log":
        e01 = (_normalize01(_log_edge(src_med, sigma=float(log_sigma))) >= 0.25).astype(np.float32)
    else:
        raise ValueError(f"Unsupported edge method: {edge}")

    # Multi scale edge density, take max over scales
    w_small, w_large = int(multiscale[0]), int(multiscale[1])
    d_small = _local_density01(e01, w_small)
    d_large = _local_density01(e01, w_large)
    doe = np.maximum(d_small, d_large)

    if normalize:
        doe = _normalize01(doe)
    return doe.astype(np.float32)


def select_roi_from_doe(
    doe: np.ndarray,
    *,
    thresh_percentile: float = 40.0,
    min_area_ratio: float = 0.01,
    close_ksize: int = 11,
    open_ksize: int = 7,
    ensure_min_frac: float = 0.10,
    ensure_seed_coverage: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Convert DOE to ROI with optional expansion heuristics.

    Parameters
    - thresh_percentile, percentile threshold on DOE.
    - ensure_min_frac, if the ROI area is below this fraction, lower the threshold until satisfied.
    - ensure_seed_coverage, optional boolean mask of FG seeds. The ROI will expand [by lowering the threshold]
      until it covers all True pixels in this mask.
    """
    H, W = doe.shape[:2]
    d = doe.astype(np.float32, copy=False)

    def _mask_from_percentile(pct: float) -> np.ndarray:
        t = float(np.percentile(d, np.clip(pct, 0.0, 100.0)))
        m = (d >= t).astype(np.uint8)
        ck = max(int(close_ksize) | 1, 3)
        ok = max(int(open_ksize) | 1, 3)
        k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ck, ck))
        k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok, ok))
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_close, iterations=1)
        m = cv.morphologyEx(m, cv.MORPH_OPEN, k_open, iterations=1)
        return m

    pct = float(thresh_percentile)
    m = _mask_from_percentile(pct)

    # Keep largest component above min area
    def _largest_component(msk: np.ndarray) -> np.ndarray:
        num, lbl, stats, _ = cv.connectedComponentsWithStats(msk, connectivity=8)
        min_area = max(int(min_area_ratio * H * W), 1)
        best = 0
        best_area = 0
        for i in range(1, num):
            area = int(stats[i, cv.CC_STAT_AREA])
            if area >= min_area and area > best_area:
                best = i
                best_area = area
        return (lbl == best).astype(np.uint8) if best != 0 else np.zeros_like(msk, dtype=np.uint8)

    mask_roi = _largest_component(m)

    # Expansion rules: minimum fraction and seed coverage
    area = float(mask_roi.sum())
    need_area = ensure_min_frac * H * W
    def _covers_seeds() -> bool:
        if ensure_seed_coverage is None:
            return True
        return bool((mask_roi.astype(bool) | (~ensure_seed_coverage.astype(bool) == True)).all())

    # Lower percentile in small steps to expand ROI
    while (area < need_area) or (ensure_seed_coverage is not None and not _covers_seeds()):
        pct = max(pct - 3.0, 0.0)
        m = _mask_from_percentile(pct)
        mask_roi = _largest_component(m)
        area = float(mask_roi.sum())
        if pct <= 0.0:
            break

    if mask_roi.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8), (0, 0, W - 1, H - 1)

    ys, xs = np.where(mask_roi > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = (x1, y1, x2, y2)
    return mask_roi.astype(np.uint8), bbox


def doe_limit_seeds(
    img_rgb_u8: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    *,
    feats_u8: Optional[np.ndarray] = None,
    median_ksize: int = 5,
    edge: EdgeMethod = "auto",
    canny_low: int = 60,
    canny_high: int = 180,
    sobel_ksize: int = 3,
    log_sigma: float = 1.2,
    window: int = 41,
    multiscale: Tuple[int, int] = (21, 61),
    thresh_percentile: float = 40.0,
    min_area_ratio: float = 0.01,
    ensure_min_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the DOE-based maximal range of analysis to seeds.
    Everything outside the ROI becomes firm background.

    New parameters:
      feats_u8, when provided, DOE uses this 3 channel feature image [for example JzAzBz or OKLab] instead of RGB.
      multiscale, two window sizes for density pooling, we take the max of both.
      ensure_min_frac, adaptive widening when the ROI is too small, relative to image area.
    """
    H, W = seeds_fg.shape
    doe = compute_doe_map(
        img_rgb_u8,
        feats_u8=feats_u8,
        median_ksize=median_ksize,
        edge=edge,
        canny_low=canny_low,
        canny_high=canny_high,
        sobel_ksize=sobel_ksize,
        log_sigma=log_sigma,
        window=window,
        multiscale=multiscale,
        normalize=True,
    )

    # Expand ROI so that it covers scribbles, if present
    seed_union = (seeds_fg | seeds_bg).astype(np.uint8)
    roi_mask, _ = select_roi_from_doe(
        doe,
        thresh_percentile=thresh_percentile,
        min_area_ratio=min_area_ratio,
        ensure_min_frac=float(ensure_min_frac),
        ensure_seed_coverage=seed_union if seed_union.any() else None,
    )

    inside = roi_mask.astype(bool)
    fg_limited = seeds_fg & inside
    bg_limited = seeds_bg | (~inside)  # outside becomes sure background
    return fg_limited.astype(bool), bg_limited.astype(bool), roi_mask


def crop_to_roi(arr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return arr[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1].copy()
