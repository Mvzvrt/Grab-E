# prgr_refine.py
# Brief, pRGR style refinement that grows high confidence labels into low confidence regions
# using simple Gaussian color models and multi scale neighborhood support.
# Works as a post step after OpenCV GrabCut, no OpenCV internals changed.

from __future__ import annotations
import numpy as np
import cv2 as cv
from typing import Iterable, Tuple

def _ridge_inv(cov: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Invert covariance with Tikhonov ridge for numerical stability."""
    d = cov.shape[0]
    return np.linalg.inv(cov + eps * np.eye(d, dtype=cov.dtype))

def _mah_dist_map(img_feats_f32: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Mahalanobis distance per pixel to a Gaussian."""
    X = img_feats_f32.reshape(-1, 3).astype(np.float32)
    dX = X - mean.reshape(1, 3)
    # (x - mu)^T Sigma^{-1} (x - mu)
    m = np.einsum("ni,ij,nj->n", dX, inv_cov, dX, optimize=True)
    return m.reshape(img_feats_f32.shape[:2])

def _band(mask01: np.ndarray, band: int = 2) -> np.ndarray:
    """Return a thin band around object boundary to focus updates."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    dil = cv.dilate(mask01, k, iterations=1)
    ero = cv.erode(mask01, k, iterations=1)
    return cv.absdiff(dil, ero).astype(bool)

def prgr_refine_mask(
    img_feats_u8: np.ndarray,
    bin_mask01: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    *,
    region_sizes: Iterable[int] = (3, 5, 9),
    boundary_band: int = 2
) -> np.ndarray:
    """
    Inputs
      img_feats_u8, H by W by 3 uint8 features, any color space you ran GrabCut in
      bin_mask01, H by W uint8 in {0,1}, the binary result from OpenCV GrabCut
      seeds_fg, H by W bool, firm foreground scribbles
      seeds_bg, H by W bool, firm background scribbles

    Returns
      refined H by W uint8 mask in {0,1}

    Method, concise
      1) Build simple Gaussian color models for high confidence FG and BG.
         High confidence comes from user scribbles plus current mask interior and exterior near those scribbles.
      2) Compute per pixel Mahalanobis distances to FG and BG, form a signed score s = d_bg - d_fg.
      3) Iterate over a few window sizes, spatially average s, update labels in a boundary band where
         score is confidently positive or negative, never overwrite user scribbles.
    """
    if img_feats_u8.dtype != np.uint8:
        img_u8 = np.clip(img_feats_u8, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_feats_u8
    H, W, C = img_u8.shape
    if C != 3:
        raise ValueError(f"Expected HxWx3, got {img_u8.shape}")
    if bin_mask01.shape[:2] != (H, W):
        raise ValueError("Mask shape must match image")

    seeds_fg = seeds_fg.astype(bool)
    seeds_bg = seeds_bg.astype(bool)
    m = (bin_mask01 > 0).astype(np.uint8)

    # High confidence sets, user scribbles plus nearby same label pixels
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    near_fg = cv.dilate(seeds_fg.astype(np.uint8), k, iterations=2).astype(bool)
    near_bg = cv.dilate(seeds_bg.astype(np.uint8), k, iterations=2).astype(bool)

    hi_fg = (m == 1) & near_fg
    hi_bg = (m == 0) & near_bg

    # Fallback if hi sets are too small
    if hi_fg.sum() < 32:
        hi_fg = seeds_fg.copy()
    if hi_bg.sum() < 32:
        hi_bg = seeds_bg.copy()

    X = img_u8.reshape(-1, 3).astype(np.float32)
    F = img_u8.astype(np.float32)

    # Gaussian stats with ridge
    def _stats(mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = F[mask_bool]
        if pts.shape[0] < 8:
            # Use global stats as weak fallback
            mu = F.mean(axis=(0, 1))
            cov = np.cov(F.reshape(-1, 3).T)
        else:
            mu = pts.mean(axis=0)
            cov = np.cov(pts.reshape(-1, 3).T)
        inv = _ridge_inv(cov.astype(np.float32))
        return mu.astype(np.float32), inv

    mu_fg, inv_fg = _stats(hi_fg)
    mu_bg, inv_bg = _stats(hi_bg)

    d_fg = _mah_dist_map(F, mu_fg, inv_fg)
    d_bg = _mah_dist_map(F, mu_bg, inv_bg)
    s = (d_bg - d_fg).astype(np.float32)  # positive favors FG

    # Robust thresholds from high confidence distributions
    s_fg = s[hi_fg] if np.any(hi_fg) else np.array([0.6], dtype=np.float32)
    s_bg = s[hi_bg] if np.any(hi_bg) else np.array([-0.6], dtype=np.float32)
    med_fg = float(np.median(s_fg))
    med_bg = float(np.median(s_bg))
    margin = max(0.1, 0.25 * abs(med_fg - med_bg))
    hi_thr = 0.0 + 0.5 * margin
    lo_thr = 0.0 - 0.5 * margin

    band = _band(m, band=int(boundary_band))
    upd_mask = m.copy()

    # Multi scale propagation
    for ksz in region_sizes:
        ksz = int(ksz)
        if ksz < 1:
            continue
        k = (ksz, ksz)
        s_blur = cv.blur(s, k)
        grow_fg = (s_blur >= hi_thr)
        grow_bg = (s_blur <= lo_thr)

        # Only update near boundary, never override firm seeds
        target_fg = grow_fg & band & (~seeds_bg)
        target_bg = grow_bg & band & (~seeds_fg)

        upd_mask[target_fg] = 1
        upd_mask[target_bg] = 0

        # Recompute band around the updated mask to focus next scale
        band = _band(upd_mask, band=int(boundary_band))

    return upd_mask.astype(np.uint8)
