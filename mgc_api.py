# -*- coding: utf-8 -*-
"""MGC API: Pre- and post-processing wrappers for modern GrabCut.

Thin wrappers that expose preprocessing and postprocessing used in
modern_grabcut.py behind unified names for integration.

Public API:
    mgc_refine_seeds: Edge-guided geodesic seed expansion with confidence gating.
    mgc_post_smooth_mask: Guided image filtering for mask cleanup.

Design Notes:
    - Edge maps are computed from the RGB image to stabilize geometry across
      color spaces.
    - ``conf_img`` in ``mgc_refine_seeds`` allows the Lab confidence gate to use
      a different color space while keeping edge geometry tied to RGB.
    - Defaults match modern_grabcut tuned flags.

Assumptions:
    - fastgeo C++ extension (built from fastgeo_core.cpp) is on the Python path.
    - modern_grabcut.py is importable from mgc_core.
"""
from __future__ import annotations

# Standard library imports
import os
from typing import Any
from typing import Tuple

# Third-party imports
import numpy as np

# Local application imports
import mgc_core.modern_grabcut as mgc  # type: ignore


# ---------------------------------------------------------------------------
# Module-level constants: Default parameters for seed expansion
# ---------------------------------------------------------------------------

_BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
"""Base directory for resolving relative paths."""

STRUCTURED_MODEL_PATH: str = os.path.join(
    _BASE_DIR, "mgc_core", "third_party", "sed", "model.yml.gz"
)
"""Default path to the Structured Edge Detection model file."""

# Edge detection defaults
DEFAULT_EDGE_BACKEND: str = "structured"
"""Edge detection backend: 'structured' (Structured Forests) or 'composite'."""

# Geodesic expansion defaults
DEFAULT_GEO_RADIUS: int = 12
"""Radius for geodesic distance-based seed expansion (pixels)."""

DEFAULT_EDGE_ALPHA: float = 3.0
"""Weighting factor for edge costs in geodesic expansion."""

DEFAULT_CONF_TAU: float = 0.85
"""Confidence threshold (tau) for Lab-space confidence gating during expansion."""


def _get_default_params() -> dict[str, Any]:
    """Build dictionary of default parameters from module constants.

    Returns:
        Dictionary mapping parameter names to their default values.
    """
    return {
        "edge_backend": DEFAULT_EDGE_BACKEND,
        "structured_model": STRUCTURED_MODEL_PATH,
        "geo_radius": DEFAULT_GEO_RADIUS,
        "edge_alpha": DEFAULT_EDGE_ALPHA,
        "conf_tau": DEFAULT_CONF_TAU,
    }


# ---------------------------------------------------------------------------
# Internal: Edge map caching
# ---------------------------------------------------------------------------

_LAST_EDGE_KEY: tuple | None = None
_LAST_EDGE_VALUE: np.ndarray | None = None


def _compute_edge_cache_key(img: np.ndarray) -> tuple:
    """Compute a cache key for an image based on memory location and shape.

    Args:
        img: Input image array.

    Returns:
        Tuple of (data pointer, height, width, strides) for cache lookup.
    """
    ptr = int(img.__array_interface__["data"][0])
    return (ptr, img.shape[0], img.shape[1], img.strides)


def _ensure_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Convert an array to boolean mask format.

    Args:
        arr: Input array of any numeric dtype.

    Returns:
        Boolean array where True indicates non-zero values.
    """
    return arr.astype(np.uint8) > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mgc_refine_seeds(
    img_rgb_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    conf_img: np.ndarray | None = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine seed masks using edge-guided geodesic expansion.

    Expands background and foreground seeds using geodesic distance computation
    guided by edge maps, combined with Lab-space color confidence gating.

    The edge map is always computed from the RGB image to maintain consistent
    geometry across color spaces. The confidence gating can optionally use
    a different image (e.g., in a different color space) via ``conf_img``.

    Args:
        img_rgb_u8: RGB image, shape (H, W, 3), dtype uint8. Used for edge
            map computation and geodesic cost calculation.
        seeds_bg: Background seed mask, shape (H, W). Non-zero values indicate
            definite background pixels.
        seeds_fg: Foreground seed mask, shape (H, W). Non-zero values indicate
            definite foreground pixels.
        conf_img: Optional confidence image, shape (H, W, 3), dtype uint8.
            Used for Lab-space confidence checks during expansion. If None,
            uses ``img_rgb_u8``.
        **kwargs: Override default parameters. Supported keys:
            - edge_backend: 'structured' or 'composite'
            - geo_radius: Geodesic expansion radius (int)
            - edge_alpha: Edge cost weight (float)
            - conf_tau: Confidence threshold (float)

    Returns:
        Tuple of (refined_fg_seeds, refined_bg_seeds), both boolean arrays
        with shape (H, W).
    """
    global _LAST_EDGE_KEY, _LAST_EDGE_VALUE

    # Build parameter dict from defaults + overrides.
    params = _get_default_params()
    params.update(kwargs or {})

    # Resolve model path; fall back to composite edges if model file not found.
    model_path = params["structured_model"]
    if params["edge_backend"] == "structured" and not os.path.exists(model_path):
        params["edge_backend"] = "composite"

    # Build edge map once per image (cached by memory location).
    cache_key = _compute_edge_cache_key(img_rgb_u8)
    if _LAST_EDGE_KEY == cache_key and _LAST_EDGE_VALUE is not None:
        edge_map = _LAST_EDGE_VALUE
    else:
        edge_map = mgc.edges_structured_forests(
            img_rgb_u8,
            model_path=params["structured_model"],
        )
        _LAST_EDGE_KEY = cache_key
        _LAST_EDGE_VALUE = edge_map

    # Use provided confidence image or fall back to RGB.
    conf = conf_img if conf_img is not None else img_rgb_u8

    # Expand seeds geodesically with confidence gating.
    refined_bg, refined_fg = mgc.expand_seeds(
        img_rgb=conf,
        E=edge_map,
        seeds_fg=_ensure_bool_mask(seeds_fg),
        seeds_bg=_ensure_bool_mask(seeds_bg),
        r_geo=int(params["geo_radius"]),
        edge_alpha=float(params["edge_alpha"]),
        conf_tau=float(params["conf_tau"]),
    )

    return refined_fg.astype(bool), refined_bg.astype(bool)


def mgc_post_smooth_mask(
    img_rgb_u8: np.ndarray,
    bin_mask01: np.ndarray,
) -> np.ndarray:
    """Apply guided image filtering to smooth a binary segmentation mask.

    Uses guided filter snapping to refine mask boundaries based on image
    edges, producing cleaner segmentation results.

    Args:
        img_rgb_u8: Guide image (typically RGB), shape (H, W, 3), dtype uint8.
            Used to guide the filtering based on image structure.
        bin_mask01: Binary input mask, shape (H, W). Values should be 0 or 1.

    Returns:
        Refined binary mask, shape (H, W), dtype uint8 with values 0 or 1.
    """
    smoothed = mgc.guided_snap(img_rgb_u8, bin_mask01)
    return (smoothed > 0).astype(np.uint8)
