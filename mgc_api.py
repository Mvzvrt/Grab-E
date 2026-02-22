# -*- coding: utf-8 -*-
"""
gc_prepost.py
Thin wrappers that expose preprocessing and postprocessing used in modern_grabcut.py
behind AO-style names so that ao plus GrabCut is treated as GrabCut in integration.

Public API
  - ao_refine_seeds(img_rgb_u8, seeds_bg, seeds_fg, conf_img=None, **kwargs)
  - ao_post_smooth_mask(img_rgb_u8, bin_mask01, guide_img=None, **kwargs)

Internal notes, what changed, and why
  - We keep the edge map from the RGB image, this stabilizes geometry across color spaces.
  - We added conf_img to ao_refine_seeds, the Lab confidence gate and color checks use conf_img if provided,
    so the chosen feature color space still affects seed expansion.
  - We added guide_img to ao_post_smooth_mask, guided snapping can use a provided guide image,
    else it falls back to RGB.
  - Defaults match modern_grabcut tuned flags. The Structured Forests model path is
    ./third_party/sed/model.yml.gz by default.

Assumptions
  - fastgeo C++ extension, built from fastgeo_core.cpp, is on the Python path as fastgeo.
  - modern_grabcut.py is importable. If your file layout uses a package folder, adjust the import below.
"""

from __future__ import annotations
from typing import Tuple
import os

import numpy as np
import cv2 as cv

# Adjust this import to your tree.
import mgc_core.modern_grabcut as mgc  # type: ignore

# micro cache for the most recent edge map, keyed by data pointer and shape
_LAST_E_KEY = None
_LAST_E_VAL = None

def _edge_key(img: np.ndarray) -> tuple:
    ptr = int(img.__array_interface__['data'][0])
    return (ptr, img.shape[0], img.shape[1], img.strides)



# ---------- Defaults mirroring modern_grabcut CLI ----------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STRUCTURED_MODEL_DEFAULT = os.path.join(_BASE_DIR, "mgc_core", "third_party", "sed", "model.yml.gz")

_DEFAULTS = dict(
    edge_backend="structured",                 # "structured" or "composite"
    structured_model=_STRUCTURED_MODEL_DEFAULT,
    texture_edges=False,
    geo_radius=12,
    edge_alpha=3.0,
    adaptive_edges=True,
    conf_tau=0.85,
    star_prior=True,
    star_tau_percentile=82.0,
    cleanup_area_frac=0.0005,
    superpixel_snap_flag=False,
    sp_region_size=18,
    sp_compactness=12.0,
    snap_guided=True,
    snap_r=4,
    snap_eps=1e-3,
    snap_thresh=0.50,
)


def _ensure_bool(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.uint8) > 0)


# ---------- Public API, AO-compatible wrappers ----------

def mgc_refine_seeds(img_rgb_u8: np.ndarray,
                    seeds_bg: np.ndarray,
                    seeds_fg: np.ndarray,
                    conf_img: np.ndarray | None = None,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine BG and FG firm seeds using modern_grabcut edge guided geodesic expansion
    combined with color confidence gating. Returns (seeds_fg_ref, seeds_bg_ref) as booleans.

    img_rgb_u8, image used for edge map and geodesic costs.
    conf_img, optional image used for Lab confidence checks inside expand_seeds. If None, uses img_rgb_u8.
    """
    p = _DEFAULTS.copy()
    p.update(kwargs or {})
    # Ensure structured model path is absolute and exists; otherwise fallback to composite edges
    if p.get("edge_backend", "structured") == "structured":
        mpath = p.get("structured_model", _STRUCTURED_MODEL_DEFAULT)
        if not os.path.isabs(mpath):
            mpath = os.path.join(_BASE_DIR, mpath)
        if os.path.exists(mpath):
            p["structured_model"] = mpath
        else:
            p["edge_backend"] = "composite"

    # Build edge map once per image, geometry is tied to RGB
    key = _edge_key(img_rgb_u8)
    global _LAST_E_KEY, _LAST_E_VAL
    if _LAST_E_KEY == key and _LAST_E_VAL is not None:
        E = _LAST_E_VAL
    else:
        E = mgc.edges_structured_forests(
            img_rgb_u8,
            model_path=p["structured_model"],
        )
        _LAST_E_KEY, _LAST_E_VAL = key, E

    # Confidence image, this lets the chosen color space influence refinement
    conf = img_rgb_u8 if conf_img is None else conf_img

    # Expand seeds geodesically with optional adaptive edges and confidence gate
    seeds_bg2, seeds_fg2 = mgc.expand_seeds(
        img_rgb=conf,
        E=E,
        seeds_fg=_ensure_bool(seeds_fg),
        seeds_bg=_ensure_bool(seeds_bg),
        r_geo=int(p["geo_radius"]),
        edge_alpha=float(p["edge_alpha"]),
        adaptive=bool(p["adaptive_edges"]),
        conf_tau=float(p["conf_tau"]),
        dbg=None,
        save_geo=False
    )

    return seeds_fg2.astype(bool), seeds_bg2.astype(bool)


def mgc_post_smooth_mask(img_rgb_u8: np.ndarray,
                        bin_mask01: np.ndarray,
                        guide_img: np.ndarray | None = None,
                        return_intermediates: bool = False,
                        **kwargs) -> np.ndarray:
    """
    Post step that cleans and snaps the mask, then returns a {0,1} uint8 mask.

    Steps
      1) edge-aware small region cleanup
      2) optional SLIC superpixel majority snap
      3) guided filter snapping

    img_rgb_u8, image used for edge map in step 1.
    guide_img, optional guidance image for the guided snap in step 3, else uses img_rgb_u8.
    return_intermediates, when True returns (final_mask, intermediates_dict) where intermediates_dict contains:
        - 'superpixel_seg': superpixel segmentation map (if superpixel_snap_flag is True)
        - 'guided_soft': soft mask from guided filter (if snap_guided is True)
    """
    p = _DEFAULTS.copy()
    p.update(kwargs or {})
    # Ensure structured model path is absolute and exists; otherwise fallback to composite edges
    if p.get("edge_backend", "structured") == "structured":
        mpath = p.get("structured_model", _STRUCTURED_MODEL_DEFAULT)
        if not os.path.isabs(mpath):
            mpath = os.path.join(_BASE_DIR, mpath)
        if os.path.exists(mpath):
            p["structured_model"] = mpath
        else:
            p["edge_backend"] = "composite"
    
    intermediates = {}

    # Edge map for cleanup preservation
    key = _edge_key(img_rgb_u8)
    global _LAST_E_KEY, _LAST_E_VAL
    if _LAST_E_KEY == key and _LAST_E_VAL is not None:
        E = _LAST_E_VAL
    else:
        E = mgc.get_edge_map(
            img_rgb_u8,
            edge_backend=p["edge_backend"],
            structured_model=p["structured_model"],
            use_texture=bool(p["texture_edges"]),
            dbg=None,
            tag="edge_map.png"
        )
        _LAST_E_KEY, _LAST_E_VAL = key, E

    y = (bin_mask01.astype(np.uint8) > 0).astype(np.uint8)

    # Cleanup tiny bits while preserving strong edges
    y = mgc.cleanup_mask(y, E, min_area_frac=float(p["cleanup_area_frac"]))

    # Optional SLIC majority snap
    if bool(p["superpixel_snap_flag"]):
        if return_intermediates:
            y, seg = mgc.superpixel_majority_snap(
                img_rgb_u8, y,
                region_size=int(p["sp_region_size"]),
                compactness=float(p["sp_compactness"]),
                tau=0.85,
                return_segmentation=True
            )
            intermediates['superpixel_seg'] = seg
        else:
            y = mgc.superpixel_majority_snap(
                img_rgb_u8, y,
                region_size=int(p["sp_region_size"]),
                compactness=float(p["sp_compactness"]),
                tau=0.85
            )

    # Guided snapping to image edges
    if bool(p["snap_guided"]):
        guide = img_rgb_u8 if guide_img is None else guide_img
        if return_intermediates:
            y, soft = mgc.guided_snap(
                guide, y,
                r=int(p["snap_r"]),
                eps=float(p["snap_eps"]),
                thresh=float(p["snap_thresh"]),
                return_soft=True
            )
            intermediates['guided_soft'] = soft
        else:
            y = mgc.guided_snap(
                guide, y,
                r=int(p["snap_r"]),
                eps=float(p["snap_eps"]),
                thresh=float(p["snap_thresh"]),
                return_soft=False
            )

    final = (y > 0).astype(np.uint8)
    
    if return_intermediates:
        return final, intermediates
    return final
