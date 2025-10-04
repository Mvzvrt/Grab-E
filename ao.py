# ao_grabcut_prep.py
# Brief: AO style pre step for GrabCut, builds stronger firm seeds from appearance hist overlap, plus a light saliency prior.
# Depends: numpy, opencv-python

from __future__ import annotations
import numpy as np
import cv2 as cv

def _quantize3(img_u8: np.ndarray, bins: int = 16) -> np.ndarray:
    """Uniform 3 channel quantization to bins^3, returns 2D array of bin ids."""
    H, W, C = img_u8.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels, got {C}")
    step = 256 // bins
    q = img_u8 // max(step, 1)
    q = np.clip(q, 0, bins - 1)
    return (q[:, :, 0] * bins + q[:, :, 1]) * bins + q[:, :, 2]

def _build_hist(qbins: np.ndarray, mask: np.ndarray, K: int) -> np.ndarray:
    """Count histogram for masked pixels over K bins, returns float vector length K."""
    count = np.bincount(qbins[mask].ravel(), minlength=K).astype(np.float64)
    return count

def _global_saliency(gray_f32: np.ndarray) -> np.ndarray:
    """Cheap saliency proxy, magnitude of gradient on smoothed grayscale, normalized to 0..1."""
    g = cv.GaussianBlur(gray_f32, (0, 0), 2.0)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax > mmin:
        mag = (mag - mmin) / (mmax - mmin)
    else:
        mag = np.zeros_like(mag, dtype=np.float32)
    return mag

def ao_refine_seeds(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    bins: int = 16,
    smooth_kernel: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine firm BG and FG seeds before calling cv2.grabCut.

    Approach, real world friendly:
      1) Quantize appearance features to a compact 3D histogram, bins per channel.
      2) Build FG and BG histograms from current firm seeds.
      3) Compute per pixel likelihood difference using Laplace smoothed per bin probabilities.
      4) Combine with a light saliency prior from gradients to push confident pixels to firm seeds.
      5) Expand around user FG and BG seeds with morphology, resolve conflicts in favor of original seeds.

    Notes:
      This is an approximation of the paper’s appearance overlap idea, it does not alter OpenCV internals.
      It only strengthens firm seeds that you pass into GrabCut.
    """
    if img_feats_u8.dtype != np.uint8:
        img_feats_u8 = np.clip(img_feats_u8, 0, 255).astype(np.uint8)
    H, W, C = img_feats_u8.shape
    if C != 3:
        raise ValueError(f"Expected HxWx3 features, got {img_feats_u8.shape}")
    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError("Seed shapes must match image features height and width")

    # If no FG seeds, return as is, OpenCV path already handles this case.
    if not np.any(seeds_fg):
        return seeds_fg, seeds_bg

    # Build appearance quantization and hists
    K = bins * bins * bins
    q = _quantize3(img_feats_u8, bins=bins)

    # Require at least a small number of pixels for both histograms
    min_needed = 8
    if int(seeds_fg.sum()) < min_needed or int(seeds_bg.sum()) < min_needed:
        return seeds_fg, seeds_bg

    h_fg = _build_hist(q, seeds_fg, K)
    h_bg = _build_hist(q, seeds_bg, K)
    # Laplace smoothing
    alpha = 1.0
    p_fg = (h_fg + alpha) / (h_fg.sum() + alpha * K)
    p_bg = (h_bg + alpha) / (h_bg.sum() + alpha * K)

    # Score per pixel from appearance
    pf = p_fg[q]
    pb = p_bg[q]
    # log odds style score, safe log
    eps = 1e-12
    s_app = np.log((pf + eps) / (pb + eps)).astype(np.float32)

    # Saliency term built from simple gradients of mean channel
    gray = img_feats_u8.mean(axis=2).astype(np.float32)
    s_sal = _global_saliency(gray)

    # Combine, keep weights modest so we do not override user intent
    # Appearance dominates, saliency is a tie breaker
    s = 0.8 * s_app + 0.2 * (s_sal - 0.5)

    # Normalize scores to robust range
    p10, p90 = np.percentile(s, [10, 90])
    if p90 > p10:
        s_norm = (s - p10) / (p90 - p10)
        s_norm = np.clip(s_norm, 0.0, 1.0)
    else:
        s_norm = 0.5 * np.ones_like(s, dtype=np.float32)

    # Thresholds relative to current seeds
    # Encourage expansions near existing seeds to avoid bleeding
    ksize = max(int(smooth_kernel), 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
    fg_d = cv.dilate(seeds_fg.astype(np.uint8), kernel, iterations=1).astype(bool)
    bg_d = cv.dilate(seeds_bg.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Adaptive thresholds from distributions
    hi = float(np.percentile(s_norm[fg_d], 70)) if np.any(fg_d) else 0.7
    lo = float(np.percentile(s_norm[bg_d], 30)) if np.any(bg_d) else 0.3

    cand_fg = (s_norm >= hi) & (~seeds_bg)
    cand_bg = (s_norm <= lo) & (~seeds_fg)

    # Only trust candidates near the dilated scribbles, plus a small safety margin
    near_fg = cv.dilate(seeds_fg.astype(np.uint8), kernel, iterations=2).astype(bool)
    near_bg = cv.dilate(seeds_bg.astype(np.uint8), kernel, iterations=2).astype(bool)

    add_fg = cand_fg & near_fg
    add_bg = cand_bg & near_bg

    # Resolve conflicts, user seeds win
    add_fg[seeds_bg] = False
    add_bg[seeds_fg] = False
    both = add_fg & add_bg
    add_fg[both] = False
    add_bg[both] = False

    seeds_fg_ref = seeds_fg | add_fg
    seeds_bg_ref = seeds_bg | add_bg

    return seeds_fg_ref, seeds_bg_ref
