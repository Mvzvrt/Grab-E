# -*- coding: utf-8 -*-
"""
MGC API: Pre- and post-processing wrappers for interactive GrabCut.
Uses edge-guided geodesic seed expansion and guided filter smoothing.
"""
from __future__ import annotations

# Standard library imports
import os
from typing import Any
from typing import Tuple

# Third-party imports
import numpy as np

# Local application imports
import mgc_core.core as core  # type: ignore

# Base directory for resolving relative paths
_BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Path to the Structured Edge Detection model file
STRUCTURED_MODEL_PATH: str = os.path.join(
    _BASE_DIR, "mgc_core", "third_party", "sed", "model.yml.gz"
)

# Edge detection backend: 'structured' (Forests) or 'composite'
DEFAULT_EDGE_BACKEND: str = "structured"

# Radius for geodesic distance-based seed expansion in pixels
DEFAULT_GEO_RADIUS: int = 12

# Weighting factor for edge costs in geodesic expansion
DEFAULT_EDGE_ALPHA: float = 3.0

# Confidence threshold for color-space gating during expansion
DEFAULT_CONF_TAU: float = 0.85

def _get_default_params() -> dict[str, Any]:
    """
    Constructs a dictionary of default parameters from module constants.
    Used to initialize seed expansion logic.
    """
    return {
        "edge_backend": DEFAULT_EDGE_BACKEND,
        "structured_model": STRUCTURED_MODEL_PATH,
        "geo_radius": DEFAULT_GEO_RADIUS,
        "edge_alpha": DEFAULT_EDGE_ALPHA,
        "conf_tau": DEFAULT_CONF_TAU,
    }


# Cache variables to avoid redundant edge map computation
_LAST_EDGE_KEY: tuple | None = None
_LAST_EDGE_VALUE: np.ndarray | None = None


def _compute_edge_cache_key(img: np.ndarray) -> tuple:
    """
    Computes a cache key based on memory pointer and image geometry.
    Used to skip expensive edge detection if the input image hasn't changed.
    """
    ptr = int(img.__array_interface__["data"][0])
    return (ptr, img.shape[0], img.shape[1], img.strides)


def _ensure_bool_mask(arr: np.ndarray) -> np.ndarray:
    """
    Standardizes input arrays into binary boolean masks.
    Ensures non-zero values are treated consistently as True.
    """
    return arr.astype(np.uint8) > 0


def _expand_seeds(
    img_rgb_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    conf_img: np.ndarray | None = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refines seed masks via edge-guided geodesic expansion.
    Propagates user scribbles into nearby pixels based on color similarity and edge boundaries.
    """
    global _LAST_EDGE_KEY, _LAST_EDGE_VALUE

    # Update default expansion parameters with runtime overrides
    params = _get_default_params()
    params.update(kwargs or {})

    # Fallback to composite edge detection if structured model is missing
    model_path = params["structured_model"]
    if params["edge_backend"] == "structured" and not os.path.exists(model_path):
        params["edge_backend"] = "composite"

    # Reuse cached edge map if image identity (pointer/shape) matches
    cache_key = _compute_edge_cache_key(img_rgb_u8)
    if _LAST_EDGE_KEY == cache_key and _LAST_EDGE_VALUE is not None:
        edge_map = _LAST_EDGE_VALUE
    else:
        # Compute forests-based edges for structural guidance
        edge_map = core.edges_structured_forests(
            img_rgb_u8,
            model_path=params["structured_model"],
        )
        _LAST_EDGE_KEY = cache_key
        _LAST_EDGE_VALUE = edge_map

    # Apply core expansion logic in C++ extension
    refined_bg, refined_fg = core.expand_seeds(
        img_rgb=img_rgb_u8,
        E=edge_map,
        seeds_fg=_ensure_bool_mask(seeds_fg),
        seeds_bg=_ensure_bool_mask(seeds_bg),
        r_geo=int(params["geo_radius"]),
        edge_alpha=float(params["edge_alpha"]),
        conf_tau=float(params["conf_tau"]),
    )

    return refined_fg.astype(bool), refined_bg.astype(bool)


def _apply_guided_filter(
    img_rgb_u8: np.ndarray,
    bin_mask01: np.ndarray,
) -> np.ndarray:
    """
    Snaps binary mask boundaries to image edges using guided filtering.
    Produces cleaner segmentation by aligning mask transitions with color discontinuities.
    """
    smoothed = core.guided_snap(img_rgb_u8, bin_mask01)
    return (smoothed > 0).astype(np.uint8)
