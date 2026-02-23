"""Minimal core algorithms for Grab-E.

Provides three public functions consumed by ``mgc_api.py``:

    - :func:`edges_structured_forests`: Edge probability map via Structured
      Forests (opencv-contrib ``cv.ximgproc``).
    - :func:`expand_seeds`: Geodesic seed expansion with Lab-space confidence
      gating.
    - :func:`guided_snap`: Guided-filter boundary snapping (He et al., 2010).

Internal helpers handle geodesic distance (via the ``fastgeo`` C++ extension),
morphological-kernel caching, and colour-based confidence estimation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from . import fastgeo

# Structured Forests edge detector (opencv-contrib).
# Eagerly resolve so missing installs fail at import time.
_XIMGPROC = cv.ximgproc  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_SED_CACHE: Dict[str, object] = {}
"""SED detector instances keyed by model file path."""

_KERNELS: Dict[Tuple[int, int, int], np.ndarray] = {}
"""Morphological kernels keyed by (width, height, shape_enum)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_sed(model_path: str) -> object:
    """Return a cached StructuredEdgeDetection instance.

    Args:
        model_path: Filesystem path to the ``.yml.gz`` model file.

    Returns:
        A ``cv.ximgproc.StructuredEdgeDetection`` object ready for
        ``detectEdges`` / ``computeOrientation`` / ``edgesNms``.
    """
    sed = _SED_CACHE.get(model_path)
    if sed is None:
        sed = cv.ximgproc.createStructuredEdgeDetection(model_path)
        _SED_CACHE[model_path] = sed
    return sed


def _k_ellipse(w: int, h: int) -> np.ndarray:
    """Return a cached elliptical structuring element.

    Args:
        w: Kernel width in pixels.
        h: Kernel height in pixels.

    Returns:
        Binary uint8 structuring element of shape ``(h, w)``.
    """
    key = (w, h, 1)
    if key not in _KERNELS:
        _KERNELS[key] = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w, h))
    return _KERNELS[key]


def _geodesic_cpp(
    cost: np.ndarray,
    seeds: np.ndarray,
    eight_connected: bool,
) -> np.ndarray:
    """Compute geodesic distance via the fastgeo C++ backend.

    Args:
        cost: Cost map, shape ``(H, W)``, dtype castable to float64.
        seeds: Binary seed mask, shape ``(H, W)``, dtype castable to uint8.
        eight_connected: Whether to use 8-connectivity (True) or 4-connectivity.

    Returns:
        Geodesic distance map, shape ``(H, W)``, dtype float64.
    """
    c = np.ascontiguousarray(cost.astype(np.float64, copy=False))
    s = np.ascontiguousarray(seeds.astype(np.uint8, copy=False))
    return fastgeo.geodesic(c, s, bool(eight_connected))


def geodesic_distance(
    cost: np.ndarray,
    seeds: np.ndarray,
    eight_connected: bool = True,
) -> np.ndarray:
    """Compute geodesic distance from seeds over a cost surface.

    This is a thin wrapper around the mandatory C++ backend (no Python fallback).

    Args:
        cost: Non-negative cost map, shape ``(H, W)``.
        seeds: Binary seed mask, shape ``(H, W)``.
        eight_connected: Use 8-connected neighbourhood. Defaults to True.

    Returns:
        Distance map, shape ``(H, W)``, dtype float64.
    """
    return _geodesic_cpp(cost, seeds, eight_connected)


def _seeds_confidence_lab(
    img_rgb: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    tau: float = 0.75,
    return_score: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Estimate foreground confidence from seed colours in CIE-Lab.

    PRE-GRABCUT COLOR LIKELIHOOD ESTIMATOR

    Color Space: CIE Lab (perceptually uniform; separates luminance from
    chrominance).

    Model: Simplified single-component GMM (maximum speed; prevents
    over-fragmentation of color clusters common in k=5 GMMs for small
    scribbles).

    Fits a single-component Gaussian to the foreground and background scribbles
    in Lab space, then applies a sigmoid on the log-likelihood ratio to produce
    a soft probability map. Pixels exceeding ``tau`` are returned as confident
    foreground.

    Args:
        img_rgb: Source image, shape ``(H, W, 3)``, dtype uint8, RGB order.
        seeds_fg: Binary foreground seed mask, shape ``(H, W)``.
        seeds_bg: Binary background seed mask, shape ``(H, W)``.
        tau: Confidence threshold applied to the sigmoid probability map.
            Defaults to 0.75.
        return_score: If True, also return the raw float32 probability map
            alongside the boolean mask.

    Returns:
        If ``return_score`` is False (default): boolean mask of confident
        foreground pixels, shape ``(H, W)``.

        If ``return_score`` is True: tuple ``(mask, probability_map)``.
    """
    # Safety check: If no foreground scribbles exist, return empty results.
    if not np.any(seeds_fg):
        empty = np.zeros(img_rgb.shape[:2], dtype=bool)
        return (empty, empty.astype(np.float32)) if return_score else empty

    # Convert to Lab for perceptually uniform distance calculations.
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab).astype(np.float32)

    # Extract Lab samples under each scribble set.
    xs, ys = np.where(seeds_fg)
    fg_samples = lab[xs, ys] if xs.size else lab[seeds_fg]
    bg_samples = lab[seeds_bg] if np.any(seeds_bg) else lab[~seeds_fg]

    # Mean colour (centroid) per distribution.
    mu_f = fg_samples.mean(axis=0)
    mu_b = bg_samples.mean(axis=0) if bg_samples.size else mu_f + 1.0

    # Global variance (spread) per distribution.
    sf = np.var(fg_samples, axis=0).mean() + 1e-3
    sb = (np.var(bg_samples, axis=0).mean() + 1e-3) if bg_samples.size else sf * 4

    # Log-likelihood helper for a single Gaussian.
    def _gauss_ll(x: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
        return -0.5 * np.sum((x - mu) ** 2, axis=2) / float(s)

    # Compute log-likelihood ratio (FG vs BG).
    ll_f = _gauss_ll(lab, mu_f, sf)
    ll_b = _gauss_ll(lab, mu_b, sb)
    delta = np.clip((ll_f - ll_b).astype(np.float32), -60.0, 60.0)

    # Stable sigmoid -> probability map in [0, 1].
    post = np.where(
        delta >= 0.0,
        1.0 / (1.0 + np.exp(-delta)),
        np.exp(delta) / (1.0 + np.exp(delta)),
    )
    post = np.nan_to_num(post, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    mask = post >= float(tau)
    if return_score:
        return mask, post.astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def edges_structured_forests(
    img_rgb: np.ndarray,
    model_path: Optional[str],
) -> np.ndarray:
    """Compute an edge-probability map using Structured Forests (SED).

    Follows the standard OpenCV pipeline for SED:
    ``detectEdges`` -> ``computeOrientation`` -> ``edgesNms`` (when available)
    -> global-max normalisation to [0, 1].

    References:
        Source:
            https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/samples/edgeboxes_demo.py
        Model:
            https://github.com/opencv/opencv_extra/tree/master/testdata/cv/ximgproc

    Args:
        img_rgb: Input image, shape ``(H, W, 3)``, dtype uint8, RGB order.
        model_path: Path to the Structured Forests ``.yml.gz`` model file.

    Returns:
        Edge probability map, shape ``(H, W)``, dtype float32 in [0, 1].

    Raises:
        FileNotFoundError: If ``model_path`` is None or does not exist.
    """
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            "Structured Forests model file not found (set --structured_model)."
        )

    # Pipeline: detectEdges -> computeOrientation -> edgesNms -> normalise.
    rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR).astype(np.float32) / 255.0
    sed = _get_sed(model_path)
    edge = sed.detectEdges(rgb).astype(np.float32)
    orient = sed.computeOrientation(edge).astype(np.float32)
    if hasattr(sed, "edgesNms"):
        edge = sed.edgesNms(edge, orient)

    # Global-max normalisation -> probability / cost map.
    max_val = float(edge.max()) + 1e-6  # avoid divide-by-zero
    return (edge / max_val).astype(np.float32, copy=False)


def expand_seeds(
    img_rgb: np.ndarray,
    E: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    r_geo: int,
    edge_alpha: float,
    conf_tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand user scribbles via geodesic distance and colour confidence.

    Steps:

    1. Normalise and smooth the edge map so a few outlier pixels do not dominate
       the cost surface.
    2. Build a cost map ``1 + edge_alpha * E^0.8`` and compute geodesic distances
       from the FG and BG seeds.
    3. Label pixels within ``r_geo`` of each seed set as geodesic-expanded seeds.
    4. Compute Lab-space colour confidence and gate it so only pixels that are
       (a) nearer to FG than BG, (b) within 1.25 x ``r_geo``, and (c) not already
       claimed by BG are promoted.
    5. Morphologically open the gated confidence mask to remove speckle.
    6. Merge geodesic FG and gated colour seeds, subtract BG claims.

    Args:
        img_rgb: Source image, shape ``(H, W, 3)``, dtype uint8, RGB.
        E: Edge probability map, shape ``(H, W)``, dtype float32 in [0, 1].
        seeds_fg: Binary foreground seed mask, shape ``(H, W)``.
        seeds_bg: Binary background seed mask, shape ``(H, W)``.
        r_geo: Geodesic expansion radius (pixels).
        edge_alpha: Edge cost multiplier.
        conf_tau: Confidence threshold for the Lab-space probability map.

    Returns:
        Tuple ``(expanded_bg, expanded_fg)`` of boolean masks, each shape
        ``(H, W)``.
    """
    # --- 1. Edge normalisation & smoothing ---
    # Ensures the cost map is not dominated by a few outlier pixels.
    E = E.astype(np.float32, copy=False)
    p95 = np.percentile(E, 95.0)
    if p95 > 1e-6:
        E = np.clip(E / p95, 0.0, 1.0)

    # Widen capture range of semantic boundaries.
    E = cv.GaussianBlur(E, (0, 0), 0.8)

    # --- 2. Cost map & geodesic distances ---
    # Non-linear compression (E^0.8) before scaling.
    e_soft = np.power(E, 0.8, dtype=np.float32)
    cost = 1.0 + float(edge_alpha) * e_soft

    d_fg = geodesic_distance(cost, seeds_fg.astype(bool), True)
    d_bg = geodesic_distance(cost, seeds_bg.astype(bool), True)

    # --- 3. Geodesic expansion zones ---
    geo_fg = d_fg <= float(r_geo)
    geo_bg = d_bg <= float(r_geo)

    # --- 4. Lab-space confidence gating ---
    conf_mask = _seeds_confidence_lab(img_rgb, seeds_fg, seeds_bg, tau=conf_tau)

    # Prefer the nearer scribble family.
    gate_near_fg = d_fg < d_bg

    # Restrict confidence additions to a 1.25x geodesic halo so that similar
    # colours in distant, unrelated regions are not captured.
    gate_within = d_fg <= (1.25 * float(r_geo))

    # Never flip pixels claimed by BG seeds or their geodesic zone.
    no_bg_lock = ~(seeds_bg.astype(bool) | geo_bg)

    conf_mask_gated = conf_mask & gate_near_fg & gate_within & no_bg_lock

    # --- 5. Morphological cleanup (remove isolated speckles) ---
    conf_mask_gated = cv.morphologyEx(
        conf_mask_gated.astype(np.uint8),
        cv.MORPH_OPEN,
        _k_ellipse(3, 3),
        iterations=1,
    ).astype(bool)

    # --- 6. Merge seeds ---
    # Final foreground seed synthesis combining geodesic reach and gated colour
    # confidence. Explicitly subtracts any background-claimed regions.
    seeds_fg_expanded = (geo_fg | conf_mask_gated) & ~geo_bg & ~seeds_bg.astype(bool)
    seeds_bg_expanded = geo_bg | seeds_bg.astype(bool)

    return seeds_bg_expanded, seeds_fg_expanded


def guided_snap(img_rgb: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """Refine a binary mask using guided image filtering.

    Applies ``cv.ximgproc.guidedFilter`` with defaults from He et al. (2010)
    as referenced by Zhang & Chai (2020).

    References:
        K. He, J. Sun, and X. Tang, "Guided Image Filtering,"
        *IEEE TPAMI*, vol. 35, no. 6, pp. 1397-1409, 2013.

    Args:
        img_rgb: Guide image, shape ``(H, W, 3)``, dtype uint8, RGB.
        bin_mask: Binary mask, shape ``(H, W)``, values 0 or 1.

    Returns:
        Refined binary mask, shape ``(H, W)``, dtype uint8, values 0 or 1.
    """
    # Radius 4 covers small spikes/sags; eps 1e-6 ensures tight edge snapping.
    r = 4
    eps = 1e-6

    guide = img_rgb.astype(np.float32)
    src = bin_mask.astype(np.float32)

    refined_soft = cv.ximgproc.guidedFilter(guide=guide, src=src, radius=r, eps=eps)
    return (refined_soft >= 0.5).astype(np.uint8)
