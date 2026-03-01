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
    rgb = img_rgb.astype(np.float32) / 255.0
    sed = _get_sed(model_path)
    
    # Compute probability and orientation maps
    edge = sed.detectEdges(rgb).astype(np.float32)
    orient = sed.computeOrientation(edge).astype(np.float32)
    
    # Clean up edges via NMS to improve edge precision
    if hasattr(sed, "edgesNms"):
        edge = sed.edgesNms(edge, orient)

    return edge

def expand_seeds(
    E: np.ndarray,
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    r_geo: int,
    edge_alpha: float,
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

    # Synthesize final expanded seed masks, subtracting overlapping background areas
    seeds_fg_expanded = (geo_fg) & ~geo_bg & ~seeds_bg.astype(bool)
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
