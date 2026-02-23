# Filename: modern_grabcut.py
# -*- coding: utf-8 -*-
"""
Modern GrabCut (non-deep), high-accuracy + fast + DEBUG-by-component

Adds classical (non-DL) upgrades:
  A) Edge map backends:
     - composite edges (Canny + ZC + grad-mag)
     - optional Structured Forests edges (requires cv.ximgproc + model)
     - optional texture edges (multi-scale Gabor) fused into composite
  B) Geodesic seed expansion upgrades:
     - adaptive edge weights (local-contrast-scaled)
     - confidence-based propagation from seed color stats in Lab
  C) Multi-resolution GMM initialization:
     - 1-pass coarse GC to learn GMMs, upsample mask & reuse models at full res
  D) ROI cropping (conservative, scale-aware)
  E) Edge-aware boundary snapping (post):
     - Guided Filter (He et al., 2010) on a soft mask; snaps to strong edges
     - optional superpixel majority snap via SLIC

NEW: Component-scoped debug (activated by --debug_dir AND --debug_comp)
  --debug_comp is a comma-separated list from:
    edges, geodesic (geo), grabcut (gc), roi, ms_init (ms/pyramid),
    post, overlap, all/full
Only artifacts from the selected components are saved.

Contract, I/O, palette and JSON summary remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from heapq import heappush, heappop
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

# ---- mandatory extras (no fallbacks) ----
# fast geodesic C++ extension (pybind11): fastgeo.geodesic(cost, seeds, eight_connected) -> float64
from . import fastgeo
# SLIC superpixels
from skimage.segmentation import slic
# Structured Forests edge detector (opencv-contrib)
_XIMGPROC = cv.ximgproc  # type: ignore[attr-defined]
# cache SED detector per model path
_SED_CACHE = {}

def _get_sed(model_path: str):
    sed = _SED_CACHE.get(model_path)
    if sed is None:
        sed = cv.ximgproc.createStructuredEdgeDetection(model_path)
        _SED_CACHE[model_path] = sed
    return sed

# ---------- constants / palette ----------
NUM_VOC_CLASSES = 21
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_KERNELS: Dict[Tuple[int, int, int], np.ndarray] = {}

def _k_ellipse(w: int, h: int) -> np.ndarray:
    k = (w, h, 1)
    if k not in _KERNELS:
        _KERNELS[k] = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w, h))
    return _KERNELS[k]

def edges_structured_forests(img_rgb: np.ndarray, model_path: Optional[str]) -> np.ndarray:
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("Structured Forests model file not found (set --structured_model).")

    """ 
    Follows the standard OpenCV pipeline for SED: detectEdges -> computeOrientation -> edgesNms (if available) -> normalize to [0,1] in 

    Source: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/samples/edgeboxes_demo.py

    Model Source: https://github.com/opencv/opencv_extra/tree/master/testdata/cv/ximgproc
    """
    rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR).astype(np.float32) / 255.0
    sed = _get_sed(model_path)
    E = sed.detectEdges(rgb).astype(np.float32)
    O = sed.computeOrientation(E).astype(np.float32)
    if hasattr(sed, "edgesNms"):
        E = sed.edgesNms(E, O)
    
    """
    Performs a Global-Max Normalization
    Find strongest edge in the entire image m and dividing every pixel by it
    Transforms the edge map into a probability/cost map
    """
    m = float(E.max()) + 1e-6 ### 1e-6 to avoid divide-by-zero
    return (E / m).astype(np.float32, copy=False)

# ---------- Geodesic helpers ----------
def _geodesic_cpp(cost: np.ndarray, seeds: np.ndarray, eight_connected: bool) -> np.ndarray:
    c = np.ascontiguousarray(cost.astype(np.float64, copy=False))
    s = np.ascontiguousarray(seeds.astype(np.uint8, copy=False))
    return fastgeo.geodesic(c, s, bool(eight_connected))

def geodesic_distance(cost: np.ndarray, seeds: np.ndarray, eight_connected: bool=True) -> np.ndarray:
    # mandatory C++ backend (no Python fallback)
    return _geodesic_cpp(cost, seeds, eight_connected)

# ---------- Seed expansion: adaptive + confidence ----------
def local_contrast(gray: np.ndarray, r: int=3) -> np.ndarray:
    """
    Reference: J. -S. Lee, "Digital Image Enhancement and Noise Filtering by Use of Local Statistics," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-2, no. 2, pp. 165-168, March 1980, doi: 10.1109/TPAMI.1980.4766994.
    keywords: {Digital images;Digital filters;Statistics;Signal processing algorithms;Filtering algorithms;Additive noise;Image processing;Pixel;Image enhancement;Frequency domain analysis;Digital image enhancement;local statistics;noise filtering;real-time processing},
    """

    g = cv.GaussianBlur(gray, (0,0), 1.2)
    """
    2. Local Mean Computation (Lee Eq. 1)
    Logic: cv::blur calls cv::boxFilter(..., normalize=true).
    Internally: 
    - RowSumFilter/ColumnSumFilter compute the double summation: sum(sum(x))
    - Logic: scale = 1.0 / (ksize.width * ksize.height)
    - Result: mu = scale * sum(sum(g)) -> Arithmetic Mean
    """
    mu = cv.blur(g, (2*r+1, 2*r+1)).astype(np.float32)

    """
    3. Local Variance Computation (Lee Eq. 2)
    Logic: Map-Reduce approach to the variance formula: Var = E[(X - E[X])^2]
    Internally:
    - (g - mu)**2 calculates the squared deviation (x - m)^2 per pixel.
    - cv::blur sums these squared deviations and applies the 'scale' factor (1/Area).
    - Result: var = (1/A) * sum(sum((x - m)^2)) -> Local Variance
    """
    var = cv.blur((g.astype(np.float32) - mu)**2, (2*r+1, 2*r+1))

    std = np.sqrt(var + 1e-6)
    p5, p95 = np.percentile(std, [5, 95])
    std = (std - p5) / (p95 - p5 + 1e-6)
    std = np.clip(std, 0.0, 1.0)
    std = cv.GaussianBlur(std, (0,0), 1.0)
    return std

def seeds_confidence_lab(img_rgb: np.ndarray, seeds_fg: np.ndarray, seeds_bg: np.ndarray,
                        tau: float=0.75, return_score: bool=False):
    """
    PRE-GRABCUT COLOR LIKELIHOOD ESTIMATOR
    Color Space: CIE Lab (Perceptually uniform; separates Luminance from Chrominance).
    Model: Simplified Single-Component GMM (Maximum speed; prevents over-fragmentation 
           of color clusters common in k=5 GMMs for small scribbles).
    Inputs: img_rgb (Source), seeds_fg/bg (User scribbles), tau (Confidence threshold).
    Outputs: Boolean mask of confident foreground pixels and/or raw probability scores.
    """
    # Safety check: If no foreground scribbles exist, return empty results
    if not np.any(seeds_fg):
        empty = np.zeros(img_rgb.shape[:2], dtype=bool)
        return (empty, empty.astype(np.float32)) if return_score else empty

    # Convert to Lab and float32 for precise distance and variance calculations
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab).astype(np.float32)

    # Identify pixel coordinates for user-defined foreground scribbles
    xs, ys = np.where(seeds_fg)
    
    # Extract the specific Lab color values under the foreground and background scribbles
    fg_samples = lab[xs, ys] if xs.size else lab[seeds_fg]
    bg_samples = lab[seeds_bg] if np.any(seeds_bg) else lab[~seeds_fg]

    # Calculate the mean color (centroid) for both foreground and background distributions
    mu_f = fg_samples.mean(axis=0)
    mu_b = bg_samples.mean(axis=0) if bg_samples.size else mu_f + 1.0

    # Calculate global variance (spread) for both distributions to determine color 'tightness'
    sf = (np.var(fg_samples, axis=0).mean() + 1e-3)
    sb = (np.var(bg_samples, axis=0).mean() + 1e-3) if bg_samples.size else sf*4

    # Internal helper to calculate the log-likelihood of a pixel belonging to a Gaussian cluster
    def gauss_ll(x, mu, s):
        return -0.5 * np.sum((x - mu)**2, axis=2) / float(s)

    # Compute log-likelihood maps for the entire image against the FG and BG models
    ll_f = gauss_ll(lab, mu_f, sf)
    ll_b = gauss_ll(lab, mu_b, sb)

    # Calculate the log-ratio (delta) between clusters to determine which is more likely
    delta = (ll_f - ll_b).astype(np.float32)

    # Clip values to prevent exponential overflow during the sigmoid transformation
    delta = np.clip(delta, -60.0, 60.0)

    # Transform the log-ratio into a normalized 0-1 probability map using a stable sigmoid
    # Values > 0.5 favor foreground; < 0.5 favor background
    post = np.where(delta >= 0.0,
                    1.0 / (1.0 + np.exp(-delta)),
                    np.exp(delta) / (1.0 + np.exp(delta)))

    # Clean up any potential mathematical errors (NaNs or Infs) to ensure array stability
    post = np.nan_to_num(post, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # Threshold the probability map by 'tau' to create the final confident seed mask
    mask = (post >= float(tau))

    if return_score:
        return mask, post.astype(np.float32)
    return mask

def expand_seeds(img_rgb: np.ndarray, E: np.ndarray, seeds_fg: np.ndarray, seeds_bg: np.ndarray,
                 r_geo: int, edge_alpha: float, conf_tau: float) -> Tuple[np.ndarray,np.ndarray]:
    """
    Ensures that the cost map is not dominated by a few outlier pixels
    """
    E = E.astype(np.float32, copy=False)
    p95 = np.percentile(E, 95.0)
    if p95 > 1e-6:
        E = np.clip(E / p95, 0.0, 1.0)

    """
    Widens the capture range of the semantic boundaries
    """
    E = cv.GaussianBlur(E, (0,0), 0.8)

    """
    Create cost map prior to geodesic distance computation
    """
    E_soft = np.power(E, 0.8, dtype=np.float32) # Applies non-linear compression
    cost = 1.0 + (float(edge_alpha) * E_soft)

    # Distances from FG and BG scribbles
    d_fg = geodesic_distance(cost, seeds_fg.astype(bool), True)
    d_bg = geodesic_distance(cost, seeds_bg.astype(bool), True)
    geo_fg = d_fg <= float(r_geo)
    geo_bg = d_bg <= float(r_geo)

    # Lab confidence (color-only)
    conf_mask = seeds_confidence_lab(img_rgb, seeds_fg, seeds_bg, tau=conf_tau)

   # Preference for the nearest scribble family
    gate_near_fg = d_fg < d_bg

    """
    Ensuring confidence additions remain within a restricted 'geodesic halo'.
    This prevents similar colors in distant, unrelated parts of the image from 
    being incorrectly flagged as foreground.
    """
    gate_within = d_fg <= (1.25 * float(r_geo))   

    """
    Protecting background integrity by preventing any foreground expansion 
    from flipping pixels already claimed by background seeds or their 
    geodesic expansion zone.
    """
    no_bg_lock  = ~(seeds_bg.astype(bool) | geo_bg)

    # Intersection of all spatial and logical constraints
    conf_mask_gated = conf_mask & gate_near_fg & gate_within & no_bg_lock

    """
    Morphological cleanup to remove isolated speckles and noise.
    Using an 'Open' operation ensures only spatially coherent color 
    clusters are promoted to seeds.
    """
    conf_mask_gated = cv.morphologyEx(conf_mask_gated.astype(np.uint8), cv.MORPH_OPEN, _k_ellipse(3,3), iterations=1).astype(bool)

    """
    Final foreground seed synthesis combining geodesic reach and gated color confidence.
    Explicitly subtracts any background-claimed regions to maintain strict seed separation.
    """
    seeds_fg2 = (geo_fg | conf_mask_gated) & (~geo_bg) & (~seeds_bg.astype(bool))

    # Preserving background geodesic claims and original user input
    seeds_bg2 = geo_bg | seeds_bg.astype(bool)

    return seeds_bg2, seeds_fg2

def guided_snap(img_rgb: np.ndarray, bin_mask: np.ndarray):
    """
    Refines the GrabCut binary mask using the original image as a guide.
    Uses defaults from He et al. (2010) as referenced by Zhang & Chai (2020).
    """
    # Standard defaults for mask refinement
    # Radius (r): 4 (covers small spikes/sags)
    # Epsilon (eps): 1e-6 (very small to ensure tight edge snapping)
    r = 4
    eps = 1e-6
    
    # Guide (I) must be the RGB image; Input (p) is the 0/1 float mask
    guide = img_rgb.astype(np.float32)
    src = bin_mask.astype(np.float32)
    
    # Apply Official Guided Filter
    refined_soft = cv.ximgproc.guidedFilter(guide=guide, src=src, radius=r, eps=eps)
    
    # Threshold to return to binary
    return (refined_soft >= 0.5).astype(np.uint8)