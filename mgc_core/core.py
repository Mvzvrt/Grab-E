"""
Core algorithms for interactive GrabCut.
Implements edge probability mapping, geodesic seed expansion, and guided-filter snapping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from . import fastgeo

# Structured Forests edge detector interface
_XIMGPROC = cv.ximgproc  # type: ignore[attr-defined]

# SED detector instances keyed by model file path
_SED_CACHE: Dict[str, object] = {}

# Morphological kernels cached by (width, height, shape_enum)
_KERNELS: Dict[Tuple[int, int, int], np.ndarray] = {}

def _get_sed(model_path: str) -> object:
    """
    Retrieves or creates a cached StructuredEdgeDetection instance.
    Enables efficient edge detection without reloading model files.
    """
    sed = _SED_CACHE.get(model_path)
    if sed is None:
        sed = cv.ximgproc.createStructuredEdgeDetection(model_path)
        _SED_CACHE[model_path] = sed
    return sed


def _k_ellipse(w: int, h: int) -> np.ndarray:
    """
    Returns a cached elliptical structuring element.
    Used for morphological operations like opening and closing.
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
    """
    Invokes the C++ backend for fast geodesic distance calculation.
    Computes shortest paths over a pixel cost surface.
    """
    c = np.ascontiguousarray(cost.astype(np.float64, copy=False))
    s = np.ascontiguousarray(seeds.astype(np.uint8, copy=False))
    return fastgeo.geodesic(c, s, bool(eight_connected))


def geodesic_distance(
    cost: np.ndarray,
    seeds: np.ndarray,
    eight_connected: bool = True,
) -> np.ndarray:
    """
    Public wrapper for geodesic distance computation.
    Requires the fastgeo C++ extension to be properly built.
    """
    return _geodesic_cpp(cost, seeds, eight_connected)


def _seeds_confidence_lab(
    img_rgb: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    tau: float = 0.75,
    return_score: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Estimates pixel-wise foreground probability using Lab color distributions.
    Fits a single-component Gaussian to FG/BG seeds and computes a log-likelihood ratio.
    Simplified version of the GMM used by GrabCut for speed and simplicity for a seed expansion use case.

    Source: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/grabcut.cpp#L60
    """
    if not np.any(seeds_fg):
        empty = np.zeros(img_rgb.shape[:2], dtype=bool)
        return (empty, empty.astype(np.float32)) if return_score else empty

    # Use Lab color space for perceptually linear color comparison
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab).astype(np.float32)

    # Sample color values at user seed locations
    # Aligns with Lines 374 - 385
    xs, ys = np.where(seeds_fg)
    fg_samples = lab[xs, ys] if xs.size else lab[seeds_fg]
    bg_samples = lab[seeds_bg] if np.any(seeds_bg) else lab[~seeds_fg]

    # Calculate mean color and variance for foreground and background classes
    # Aligns with Lines 191 - 192
    mu_f = fg_samples.mean(axis=0)
    mu_b = bg_samples.mean(axis=0) if bg_samples.size else mu_f + 1.0

    # Aligns with Lines 194 - 197
    sf = np.var(fg_samples, axis=0).mean() + 1e-3
    sb = (np.var(bg_samples, axis=0).mean() + 1e-3) if bg_samples.size else sf * 4

    def _gauss_ll(x: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
        """
        Calculates the Log-Likelihood of the Gaussian distribution.
        
        Equivalence to GrabCut (grabcut.cpp):
        In GrabCut, 'mult' represents the Mahalanobis Distance: (x-μ)ᵀ Σ⁻¹ (x-μ).
        By assuming a spherical covariance Σ (variance 's' in all directions), 
        Σ⁻¹ becomes the scalar 1/s. The formula collapses to:
        Σ(x_i - μ_i)² / s, which is exactly np.sum((x - mu) ** 2) / s.
        """
        return -0.5 * np.sum((x - mu) ** 2, axis=2) / float(s)

    # Compute color-based log-likelihood ratio
    ll_f = _gauss_ll(lab, mu_f, sf)
    ll_b = _gauss_ll(lab, mu_b, sb)
    delta = np.clip((ll_f - ll_b).astype(np.float32), -60.0, 60.0)

    # Map likelihood difference to [0, 1] probability via sigmoid
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

def edges_structured_forests(
    img_rgb: np.ndarray,
    model_path: Optional[str],
) -> np.ndarray:
    """
    Detects semantic boundaries using Structured Edge Forests (SED).
    Pre-processes input, runs inference, and applies non-maximum suppression (NMS).
    
    Source: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/samples/edgeboxes_demo.py
    Model: https://github.com/opencv/opencv_extra/tree/master/testdata/cv/ximgproc
    """
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            "Structured Forests model file not found (set --structured_model)."
        )

    # Normalize image and convert to BGR for SED library
    rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR).astype(np.float32) / 255.0
    sed = _get_sed(model_path)
    
    # Compute probability and orientation maps
    edge = sed.detectEdges(rgb).astype(np.float32)
    orient = sed.computeOrientation(edge).astype(np.float32)
    
    # Clean up edges via NMS to improve edge precision
    if hasattr(sed, "edgesNms"):
        edge = sed.edgesNms(edge, orient)

    return edge


def expand_seeds(
    img_rgb: np.ndarray,
    E: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    r_geo: int,
    edge_alpha: float,
    conf_tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands user scribbles by combining edge-guided geodesic distance and color confidence.
    Propagates seeds into structural regions while ensuring consistency via color-space gating.
    """
    # Clip and smooth edge map to create a more robust cost surface
    E = E.astype(np.float32, copy=False)
    p95 = np.percentile(E, 95.0)
    if p95 > 1e-6:
        E = np.clip(E / p95, 0.0, 1.0)
    E = cv.GaussianBlur(E, (0, 0), 0.8)

    # Convert probability to expansion cost using non-linear compression
    e_soft = np.power(E, 0.8, dtype=np.float32)
    cost = 1.0 + float(edge_alpha) * e_soft

    # Calculate shortest-path geodesic distance from user defined seeds
    d_fg = geodesic_distance(cost, seeds_fg.astype(bool), True)
    d_bg = geodesic_distance(cost, seeds_bg.astype(bool), True)

    # Define zones within the expansion radius
    geo_fg = d_fg <= float(r_geo)
    geo_bg = d_bg <= float(r_geo)

    # Select confident foreground pixels based on color similarity to current scribbles
    conf_mask = _seeds_confidence_lab(img_rgb, seeds_fg, seeds_bg, tau=conf_tau)

    # Restrict confidence-based additions to regions closer to FG than BG
    gate_near_fg = d_fg < d_bg
    gate_within = d_fg <= (1.25 * float(r_geo))
    no_bg_lock = ~(seeds_bg.astype(bool) | geo_bg)

    # Intersection of color confidence and spatial proximity gates
    conf_mask_gated = conf_mask & gate_near_fg & gate_within & no_bg_lock

    # Eliminate isolated noise using morphological opening
    conf_mask_gated = cv.morphologyEx(
        conf_mask_gated.astype(np.uint8),
        cv.MORPH_OPEN,
        _k_ellipse(3, 3),
        iterations=1,
    ).astype(bool)

    # Synthesize final expanded seed masks, subtracting overlapping background areas
    seeds_fg_expanded = (geo_fg | conf_mask_gated) & ~geo_bg & ~seeds_bg.astype(bool)
    seeds_bg_expanded = geo_bg | seeds_bg.astype(bool)

    return seeds_bg_expanded, seeds_fg_expanded


def guided_snap(img_rgb: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """
    Refines a binary segmentation mask using guided image filtering.
    Smooths boundaries and snaps them to high-contrast edges in the guide image.
    """
    # Use standard radius and epsilon for precise edge alignment
    r = 4
    eps = 1e-6

    guide = img_rgb.astype(np.float32)
    src = bin_mask.astype(np.float32)

    # Perform guided filtering to align mask edges with image structures
    refined_soft = cv.ximgproc.guidedFilter(guide=guide, src=src, radius=r, eps=eps)
    return (refined_soft >= 0.5).astype(np.uint8)
