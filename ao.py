# ao.py
# Brief, AO style preprocessing for GrabCut that strengthens firm seeds using appearance histograms,
# adds an adaptive aggressiveness based on inside versus outside histogram overlap,
# mixes a light saliency prior, then exposes an optional post step to smooth small artifacts.
# This approximates ideas from the paper while keeping OpenCV GrabCut unchanged.

from __future__ import annotations
import numpy as np
import cv2 as cv
from typing import Optional, Tuple

# ---------- helpers ----------

def _quantize3(img_u8: np.ndarray, bins: int = 16) -> np.ndarray:
    """Uniform 3 channel quantization to bins^3, returns 2D array of bin ids."""
    H, W, C = img_u8.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels, got {C}")
    step = max(256 // int(bins), 1)
    q = img_u8 // step
    q = np.clip(q, 0, bins - 1)
    return (q[:, :, 0] * bins + q[:, :, 1]) * bins + q[:, :, 2]

def _build_hist(qbins: np.ndarray, mask: np.ndarray, K: int) -> np.ndarray:
    """Count histogram for masked pixels over K bins, returns float vector length K."""
    count = np.bincount(qbins[mask].ravel(), minlength=K).astype(np.float64)
    return count

def _global_saliency(gray_f32: np.ndarray) -> np.ndarray:
    """Simple saliency proxy, gradient magnitude on a smoothed grayscale, normalized 0 to 1."""
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

def _bbox_from_fg_seeds(seeds_fg: np.ndarray, margin_frac: float = 0.05) -> Optional[Tuple[int, int, int, int]]:
    """Tight rectangle around FG seeds with a small margin, returns (x1, y1, x2, y2) or None if no seeds."""
    ys, xs = np.where(seeds_fg)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    H, W = seeds_fg.shape
    pad = int(margin_frac * min(H, W))
    return max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, W - 1), min(y2 + pad, H - 1)

def _hist_overlap_l1(p: np.ndarray, q: np.ndarray) -> float:
    """L1 overlap between two probability histograms, sum over min(p, q), range 0 to 1."""
    return float(np.minimum(p, q).sum())

def _remove_small_regions(mask01: np.ndarray, min_area: int, fg_value: int = 1) -> np.ndarray:
    """Remove small connected components of the given value, 8 connectivity."""
    m = (mask01 == fg_value).astype(np.uint8)
    num, lbl, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area >= min_area:
            keep[lbl == i] = 1
    out = mask01.copy()
    out[(out == fg_value) & (keep == 0)] = 1 - fg_value
    return out

# ---------- main, seed refinement ----------

def ao_refine_seeds(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    bins: int = 16,
    smooth_kernel: int = 3,
    box_xyxy: Optional[Tuple[int, int, int, int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine firm BG and FG seeds before calling cv2.grabCut.

    What this does:
      1) Quantize features to a compact 3D histogram.
      2) Build FG and BG histograms from current firm seeds.
      3) Compute per pixel log odds between FG and BG histograms with Laplace smoothing.
      4) Add a light saliency prior.
      5) Estimate inside versus outside histogram overlap to adapt aggressiveness.
      6) Promote confident pixels near user scribbles to firm FG or firm BG.

    Returns, (seeds_fg_refined, seeds_bg_refined) as boolean arrays.
    """
    if img_feats_u8.dtype != np.uint8:
        img_feats_u8 = np.clip(img_feats_u8, 0, 255).astype(np.uint8)
    H, W, C = img_feats_u8.shape
    if C != 3:
        raise ValueError(f"Expected HxWx3 features, got {img_feats_u8.shape}")
    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError("Seed shapes must match image features height and width")

    if not np.any(seeds_fg) or not np.any(seeds_bg):
        return seeds_fg, seeds_bg

    # 1,2) Appearance quantization and histograms
    K = int(bins) * int(bins) * int(bins)
    q = _quantize3(img_feats_u8, bins=int(bins))
    h_fg = _build_hist(q, seeds_fg, K)
    h_bg = _build_hist(q, seeds_bg, K)

    # Laplace smoothing to avoid zero bins
    alpha = 1.0
    p_fg = (h_fg + alpha) / (h_fg.sum() + alpha * K)
    p_bg = (h_bg + alpha) / (h_bg.sum() + alpha * K)

    # 3) Per pixel appearance score, log odds
    pf = p_fg[q]
    pb = p_bg[q]
    eps = 1e-12
    s_app = np.log((pf + eps) / (pb + eps)).astype(np.float32)

    # 4) Saliency prior
    gray = img_feats_u8.mean(axis=2).astype(np.float32)
    s_sal = _global_saliency(gray)

    # 5) Adaptive aggressiveness from inside versus outside histogram overlap
    # Build inside and outside regions, prefer user provided box, else tight box around FG seeds
    if box_xyxy is None:
        box_xyxy = _bbox_from_fg_seeds(seeds_fg)
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        inside = np.zeros((H, W), np.bool_)
        inside[y1 : y2 + 1, x1 : x2 + 1] = True
        outside = ~inside
    else:
        # Fallback, use dilated FG as inside, complement as outside
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        inside = cv.dilate(seeds_fg.astype(np.uint8), k, iterations=2).astype(bool)
        outside = ~inside

    h_in = _build_hist(q, inside, K)
    h_out = _build_hist(q, outside, K)
    p_in = (h_in + alpha) / (h_in.sum() + alpha * K)
    p_out = (h_out + alpha) / (h_out.sum() + alpha * K)
    overlap = _hist_overlap_l1(p_in, p_out)  # 0 to 1, higher means less separable

    # Map overlap to aggressiveness A, lower overlap means we can be more aggressive
    A = 1.0 - float(np.clip(overlap, 0.0, 1.0))  # 0 to 1

    # Weight saliency more when overlap is high, that is, appearance is weak
    w_sal = 0.2 + 0.3 * float(overlap)  # range about 0.2 to 0.5
    s = 0.8 * s_app + w_sal * (s_sal - 0.5)

    # Normalize for robust thresholds
    p10, p90 = np.percentile(s, [10, 90])
    if p90 > p10:
        s_norm = np.clip((s - p10) / (p90 - p10), 0.0, 1.0)
    else:
        s_norm = 0.5 * np.ones_like(s, dtype=np.float32)

    # 6) Thresholds and locality
    ksize = max(int(smooth_kernel), 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
    fg_d = cv.dilate(seeds_fg.astype(np.uint8), kernel, iterations=1).astype(bool)
    bg_d = cv.dilate(seeds_bg.astype(np.uint8), kernel, iterations=1).astype(bool)

    base_hi = float(np.percentile(s_norm[fg_d], 70)) if np.any(fg_d) else 0.7
    base_lo = float(np.percentile(s_norm[bg_d], 30)) if np.any(bg_d) else 0.3

    # Make thresholds adaptive, when A is high, be more willing to expand FG and BG
    hi = float(np.clip(base_hi - 0.15 * A, 0.50, 0.95))
    lo = float(np.clip(base_lo + 0.15 * A, 0.05, 0.50))

    cand_fg = (s_norm >= hi) & (~seeds_bg)
    cand_bg = (s_norm <= lo) & (~seeds_fg)

    # Trust candidates near the scribbles, expand a little more when A is high
    near_iter = 2 + int(round(1 * A))  # 2 to 3 iterations
    near_fg = cv.dilate(seeds_fg.astype(np.uint8), kernel, iterations=near_iter).astype(bool)
    near_bg = cv.dilate(seeds_bg.astype(np.uint8), kernel, iterations=near_iter).astype(bool)

    add_fg = cand_fg & near_fg
    add_bg = cand_bg & near_bg

    # Resolve conflicts, original seeds win
    add_fg[seeds_bg] = False
    add_bg[seeds_fg] = False
    both = add_fg & add_bg
    add_fg[both] = False
    add_bg[both] = False

    seeds_fg_ref = seeds_fg | add_fg
    seeds_bg_ref = seeds_bg | add_bg

    return seeds_fg_ref, seeds_bg_ref

# ---------- optional post step, border smoothing and small artifact cleanup ----------

def ao_post_smooth_mask(
    img_feats_u8: np.ndarray,
    bin_mask01: np.ndarray,
    *,
    min_region_ratio: float = 0.001,
    close_kernel: int = 3,
    open_kernel: int = 2
) -> np.ndarray:
    """
    Light border smoothing and artifact cleanup on a 0,1 binary mask.
    1) Close small gaps, then open to smooth jagged edges a bit.
    2) Remove tiny foreground islands and fill tiny background holes relative to image area.
    Returns uint8 mask with values 0 or 1.
    """
    if bin_mask01.dtype != np.uint8:
        m = (bin_mask01 > 0).astype(np.uint8)
    else:
        m = (bin_mask01 > 0).astype(np.uint8)

    H, W = m.shape[:2]
    min_area = max(int(min_region_ratio * H * W), 1)

    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(close_kernel, 1), max(close_kernel, 1)))
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(open_kernel, 1), max(open_kernel, 1)))

    m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_close, iterations=1)
    m = cv.morphologyEx(m, cv.MORPH_OPEN, k_open, iterations=1)

    # Remove small FG islands
    m = _remove_small_regions(m, min_area=min_area, fg_value=1)
    # Fill small BG holes by removing small components in the inverted mask
    inv = 1 - m
    inv = _remove_small_regions(inv, min_area=min_area, fg_value=1)
    m = 1 - inv

    return m.astype(np.uint8)
