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

def voc_palette() -> np.ndarray:
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lab = i
        for j in range(8):
            pal[i, 0] |= (((lab >> 0) & 1) << (7 - j))
            pal[i, 1] |= (((lab >> 1) & 1) << (7 - j))
            pal[i, 2] |= (((lab >> 2) & 1) << (7 - j))
            lab >>= 3
    return pal

def save_indexed_png(arr: np.ndarray, path: str) -> None:
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    img.putpalette(voc_palette().ravel().tolist())
    img.save(path)

# --- debug helpers -----------------------------------------------------------
def _ensure_u8_gray01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    m, M = float(np.min(a)), float(np.max(a))
    if M - m < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip(255 * (a - m) / (M - m), 0, 255).astype(np.uint8)

def _overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    base = img_rgb.copy().astype(np.float32)
    ov = np.zeros_like(base); ov[..., 0] = color[0]; ov[..., 1] = color[1]; ov[..., 2] = color[2]
    m = (mask > 0)[..., None].astype(np.float32)
    out = base * (1 - alpha * m) + ov * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)

class DebugRecorder:
    def __init__(self, root: Optional[Path], tag: str):
        self.enabled = root is not None
        self.root = None
        self.meta = {}
        if self.enabled:
            self.root = (root / tag)
            self.root.mkdir(parents=True, exist_ok=True)

    def note(self, **kwargs):
        if not self.enabled: return
        self.meta.update(kwargs)

    def save_gray(self, name: str, arr: np.ndarray):
        if not self.enabled: return
        Image.fromarray(_ensure_u8_gray01(arr)).save(self.root / name)

    def save_img(self, name: str, img_rgb: np.ndarray):
        if not self.enabled: return
        Image.fromarray(img_rgb.astype(np.uint8)).save(self.root / name)

    def save_overlay(self, name: str, img_rgb: np.ndarray, mask: np.ndarray, color=(255,0,0), alpha=0.5):
        if not self.enabled: return
        self.save_img(name, _overlay_mask(img_rgb, mask, color=color, alpha=alpha))

    def dump_meta(self, name: str = "meta.json"):
        if not self.enabled: return
        with open(self.root / name, "w") as f:
            json.dump(self.meta, f, indent=2)

# --- debug plan --------------------------------------------------------------
class DebugPlan:
    """
    Choose which pipeline component(s) to debug. Comma-separated names.
    Valid names (case-insensitive): edges, geodesic (geo), grabcut (gc),
    roi, ms_init (ms/pyramid), post, overlap, all/full
    """
    def __init__(self, comp: str):
        s = {t.strip().lower() for t in comp.split(",") if t.strip()}
        self.enabled  = bool(s)
        self.all      = ("all" in s) or ("full" in s)
        def has(x): return self.all or (x in s)
        self.edges    = has("edges")
        self.geodesic = has("geodesic") or has("geo")
        self.grabcut  = has("grabcut")  or has("gc")
        self.roi      = has("roi")
        self.ms_init  = has("ms_init")  or has("ms") or has("pyramid")
        self.post     = has("post")
        self.overlap  = has("overlap")

# ---------- I/O ----------
def load_img(p: Path) -> np.ndarray:
    img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be RGB uint8")
    return img

def load_anns(p: Path) -> np.ndarray:
    ext = p.suffix.lower()
    if ext == ".npy":
        return np.load(p).astype(np.int32, copy=False)
    if ext in (".png", ".bmp", ".tif", ".tiff"):
        return np.array(Image.open(p).convert("P"), dtype=np.int32)
    raise ValueError(f"Unsupported annotation format: {ext}")

def find_image(base: str, images_dir: Path) -> Optional[Path]:
    for e in _IMG_EXTS:
        q = images_dir / f"{base}{e}"
        if q.exists():
            return q
    return None

def base_from_ann_name(name: str) -> str:
    for sfx in ("_anns_scribbleids", "_scribbleids", "_anns"):
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name

# ---------- Edge maps ----------
def edges_composite(img_rgb: np.ndarray, use_texture: bool=False) -> np.ndarray:
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    gray_blur = cv.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=7)
    med = float(np.median(gray_blur))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    can = cv.Canny(gray_blur, lo, hi).astype(np.float32) * (1.0/255.0)
    lap = cv.Laplacian(gray_blur, cv.CV_32F, ksize=3)
    zc = (np.sign(lap) != np.sign(cv.GaussianBlur(lap, (3, 3), 0))).astype(np.float32)
    gx = cv.Sobel(gray_blur, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray_blur, cv.CV_32F, 0, 1, ksize=3)
    gmag = cv.magnitude(gx, gy)
    gmag *= 1.0 / (float(gmag.max()) + 1e-6)
    E = 0.5 * can + 0.25 * zc + 0.25 * gmag

    if use_texture:
        gE = np.zeros_like(gmag)
        for ksize, sigma, lambd, gamma in [(9,3,6,0.5),(13,4,9,0.5)]:
            kern = cv.getGaborKernel((ksize, ksize), sigma, 0, lambd, gamma, 0, ktype=cv.CV_32F)
            resp = cv.filter2D(gray_blur, cv.CV_32F, kern)
            gE = np.maximum(gE, np.abs(resp))
        gE *= 1.0 / (float(gE.max()) + 1e-6)
        E = np.clip(E*0.8 + 0.2*gE, 0.0, 1.0, out=E)

    return E.astype(np.float32, copy=False)

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

# ---------- Guided filter (edge-aware smoothing for snapping) ----------
def _boxfilter(img: np.ndarray, r: int) -> np.ndarray:
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    out = np.empty_like(img, dtype=np.float32)
    for c in range(C):
        I = img[..., c]
        integral = cv.integral(I)
        y1 = np.arange(H) - r; y2 = np.arange(H) + r + 1
        x1 = np.arange(W) - r; x2 = np.arange(W) + r + 1
        y1 = np.clip(y1, 0, H); y2 = np.clip(y2, 0, H)
        x1 = np.clip(x1, 0, W); x2 = np.clip(x2, 0, W)
        A = integral[y1[:,None], x1]
        B = integral[y1[:,None], x2]
        Cc= integral[y2[:,None], x1]
        D = integral[y2[:,None], x2]
        out[..., c] = (D - B - Cc + A)
    return out.squeeze()

def guided_filter_color(I_rgb: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    I = I_rgb.astype(np.float32, copy=False) / 255.0
    p = p.astype(np.float32, copy=False)
    ones = np.ones(p.shape, dtype=np.float32)
    N = _boxfilter(ones, r)

    mean_Ir = _boxfilter(I[...,0], r) / N
    mean_Ig = _boxfilter(I[...,1], r) / N
    mean_Ib = _boxfilter(I[...,2], r) / N
    mean_p  = _boxfilter(p, r) / N

    mean_Ip_r = _boxfilter(I[...,0]*p, r) / N
    mean_Ip_g = _boxfilter(I[...,1]*p, r) / N
    mean_Ip_b = _boxfilter(I[...,2]*p, r) / N

    cov_Ip_r = mean_Ip_r - mean_Ir * mean_p
    cov_Ip_g = mean_Ip_g - mean_Ig * mean_p
    cov_Ip_b = mean_Ip_b - mean_Ib * mean_p

    var_I_rr = _boxfilter(I[...,0]*I[...,0], r)/N - mean_Ir*mean_Ir + eps
    var_I_rg = _boxfilter(I[...,0]*I[...,1], r)/N - mean_Ir*mean_Ig
    var_I_rb = _boxfilter(I[...,0]*I[...,2], r)/N - mean_Ir*mean_Ib
    var_I_gg = _boxfilter(I[...,1]*I[...,1], r)/N - mean_Ig*mean_Ig + eps
    var_I_gb = _boxfilter(I[...,1]*I[...,2], r)/N - mean_Ig*mean_Ib
    var_I_bb = _boxfilter(I[...,2]*I[...,2], r)/N - mean_Ib*mean_Ib + eps

    det = (var_I_rr*var_I_gg*var_I_bb
           + 2*var_I_rg*var_I_rb*var_I_gb
           - var_I_rr*var_I_gb*var_I_gb
           - var_I_gg*var_I_rb*var_I_rb
           - var_I_bb*var_I_rg*var_I_rg) + 1e-12

    inv_rr = (var_I_gg*var_I_bb - var_I_gb*var_I_gb) / det
    inv_rg = (var_I_rb*var_I_gb - var_I_rg*var_I_bb) / det
    inv_rb = (var_I_rg*var_I_gb - var_I_rb*var_I_gg) / det
    inv_gg = (var_I_rr*var_I_bb - var_I_rb*var_I_rb) / det
    inv_gb = (var_I_rb*var_I_rg - var_I_rr*var_I_gb) / det
    inv_bb = (var_I_rr*var_I_gg - var_I_rg*var_I_rg) / det

    a_r = inv_rr*cov_Ip_r + inv_rg*cov_Ip_g + inv_rb*cov_Ip_b
    a_g = inv_rg*cov_Ip_r + inv_gg*cov_Ip_g + inv_gb*cov_Ip_b
    a_b = inv_rb*cov_Ip_r + inv_gb*cov_Ip_g + inv_bb*cov_Ip_b

    b = mean_p - a_r*mean_Ir - a_g*mean_Ig - a_b*mean_Ib

    mean_a_r = _boxfilter(a_r, r) / N
    mean_a_g = _boxfilter(a_g, r) / N
    mean_a_b = _boxfilter(a_b, r) / N
    mean_b   = _boxfilter(b, r) / N

    q = (mean_a_r*I[...,0] + mean_a_g*I[...,1] + mean_a_b*I[...,2] + mean_b)
    return np.clip(q, 0.0, 1.0).astype(np.float32, copy=False)

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


# ---------- ROI helper (conservative, scale-aware) ----------
def _tight_roi_from_mask(mask: np.ndarray, pad: int, H: int, W: int) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, H, 0, W
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + 1 + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + 1 + pad)
    return y1, y2, x1, x2

# ---------- GrabCut core (with multi-resolution GMM init) ----------
def grabcut_once_extended(img_rgb: np.ndarray,
                          seeds_bg: np.ndarray,
                          seeds_fg: np.ndarray,
                          iters: int = 2,
                          edge_map: Optional[np.ndarray] = None,
                          geo_radius: int = 7,
                          edge_alpha: float = 4.0,
                          adaptive_edges: bool = True,
                          conf_tau: float = 0.75,
                          star_prior: bool = False,
                          star_tau_percentile: float = 80.0,
                          downscale: float = 1.0,
                          eval_freeze_after: int = 1,
                          *,
                          roi_pad: int = 96,
                          use_roi: bool = True,
                          ms_init: bool = True,
                          ms_scale: float = 0.5,
                          dbg: Optional[DebugRecorder]=None,
                          dbg_prefix: str="",
                          save_edges: bool=False,
                          save_geo: bool=False,
                          save_roi: bool=False,
                          save_ms: bool=False,
                          save_gc: bool=False) -> np.ndarray:
    """Run OpenCV GrabCut with advanced init/prior and conservative ROI."""
    H, W, _ = img_rgb.shape
    if not np.any(seeds_fg):
        return np.zeros((H, W), dtype=np.uint8)

    work_img = img_rgb
    seeds_bg = seeds_bg.astype(bool, copy=False)
    seeds_fg = seeds_fg.astype(bool, copy=False)

    if downscale < 1.0:
        newW, newH = max(1, int(W * downscale)), max(1, int(H * downscale))
        work_img = cv.resize(work_img, (newW, newH), interpolation=cv.INTER_AREA)
        seeds_bg = cv.resize(seeds_bg.astype(np.uint8), (newW, newH), interpolation=cv.INTER_NEAREST).astype(bool)
        seeds_fg = cv.resize(seeds_fg.astype(np.uint8), (newW, newH), interpolation=cv.INTER_NEAREST).astype(bool)

    E = edge_map if edge_map is not None else edges_composite(work_img, use_texture=False)
    if dbg and save_edges: dbg.save_gray(f"{dbg_prefix}edge_map.png", E)

    seeds_bg, seeds_fg = expand_seeds(work_img, E, seeds_fg, seeds_bg,
                                      r_geo=int(geo_radius), edge_alpha=float(edge_alpha),
                                      adaptive=bool(adaptive_edges), conf_tau=float(conf_tau),
                                      dbg=(dbg if save_geo else None), save_geo=save_geo)

    feasibility = None
    if star_prior and np.any(seeds_fg):
        cost = 1.0 + float(edge_alpha) * E.astype(np.float64, copy=False)
        d_fg = geodesic_distance(cost, seeds_fg, eight_connected=True)
        finite_vals = d_fg[np.isfinite(d_fg)]
        if finite_vals.size:
            tau = np.percentile(finite_vals, float(star_tau_percentile))
            feasibility = (d_fg <= tau)
            if dbg and (save_geo or save_roi):
                dbg.save_overlay(f"{dbg_prefix}feasible_fg.png", work_img, feasibility, color=(255,255,0), alpha=0.5)

    H2, W2 = work_img.shape[:2]
    base_mask = np.full((H2, W2), cv.GC_PR_BGD, dtype=np.uint8)
    base_mask[seeds_bg] = cv.GC_BGD
    base_mask[seeds_fg] = cv.GC_FGD
    if feasibility is not None:
        base_mask[~feasibility] = cv.GC_BGD

    if use_roi:
        eff_pad = int(max(16, round(roi_pad * (work_img.shape[1] / float(W)))))
        roi_mask = seeds_fg.astype(np.uint8, copy=False)
        if roi_mask.any():
            inv = 1 - roi_mask
            dist = cv.distanceTransform(inv, cv.DIST_L2, 3)
            roi_mask = (dist <= float(eff_pad)).astype(np.uint8)
        else:
            roi_mask = np.zeros_like(roi_mask, dtype=np.uint8)
        if feasibility is not None:
            roi_mask = np.maximum(roi_mask, feasibility.astype(np.uint8, copy=False))
        k = _k_ellipse(5, 5)
        roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, k, iterations=1)
        y1, y2, x1, x2 = _tight_roi_from_mask(roi_mask.astype(bool), eff_pad, H2, W2)
        if dbg and save_roi:
            dbg.save_overlay(f"{dbg_prefix}roi_mask.png", work_img, roi_mask, color=(0,255,255), alpha=0.4)
            dbg.note(roi=[int(y1), int(y2), int(x1), int(x2)], downscale=float(downscale), iters=int(iters))
        work_roi = work_img[y1:y2, x1:x2]
        mask_roi = base_mask[y1:y2, x1:x2]
    else:
        y1, y2, x1, x2 = 0, H2, 0, W2
        work_roi = work_img
        mask_roi = base_mask

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if ms_init and (0.3 <= ms_scale < 1.0):
        sw, sh = max(2, int(work_roi.shape[1]*ms_scale)), max(2, int(work_roi.shape[0]*ms_scale))
        img_s = cv.resize(work_roi, (sw, sh), interpolation=cv.INTER_AREA)
        mask_s = cv.resize(mask_roi, (sw, sh), interpolation=cv.INTER_NEAREST)
        cv.grabCut(img_s, mask_s, None, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
        if dbg and save_ms:
            dbg.save_gray(f"{dbg_prefix}ms_init_mask_coarse.png", ((mask_s==cv.GC_FGD)|(mask_s==cv.GC_PR_FGD)).astype(np.float32))
        mask_roi = cv.resize(mask_s, (work_roi.shape[1], work_roi.shape[0]), interpolation=cv.INTER_NEAREST)

    cv.grabCut(work_roi, mask_roi, None, bgdModel, fgdModel, max(1, int(iters)), cv.GC_INIT_WITH_MASK)
    if dbg and save_gc:
        dbg.save_gray(f"{dbg_prefix}gc_main_mask.png", ((mask_roi==cv.GC_FGD)|(mask_roi==cv.GC_PR_FGD)).astype(np.float32))

    if eval_freeze_after and int(eval_freeze_after) > 0:
        for _ in range(int(eval_freeze_after)):
            cv.grabCut(work_roi, mask_roi, None, bgdModel, fgdModel, 1, cv.GC_EVAL_FREEZE_MODEL)
        if dbg and save_gc:
            dbg.save_gray(f"{dbg_prefix}gc_frozen_mask.png", ((mask_roi==cv.GC_FGD)|(mask_roi==cv.GC_PR_FGD)).astype(np.float32))

    out_small = np.zeros((H2, W2), dtype=np.uint8)
    if use_roi:
        out_small[y1:y2, x1:x2] = ((mask_roi == cv.GC_FGD) | (mask_roi == cv.GC_PR_FGD)).astype(np.uint8)
    else:
        out_small = ((mask_roi == cv.GC_FGD) | (mask_roi == cv.GC_PR_FGD)).astype(np.uint8)

    out = cv.resize(out_small, (W, H), interpolation=cv.INTER_NEAREST) if downscale < 1.0 else out_small

    if dbg and (save_gc or save_roi):
        full = np.zeros((H2, W2), np.float32)
        full[y1:y2, x1:x2] = ((mask_roi==cv.GC_FGD)|(mask_roi==cv.GC_PR_FGD)).astype(np.float32)
        dbg.save_gray(f"{dbg_prefix}gc_out_full.png", full)

    return out.astype(np.uint8, copy=False)

# ---------- Post: cleanup, superpixel snap, guided snap ----------
def cleanup_mask(binary: np.ndarray, edge_map: np.ndarray, min_area_frac: float = 0.0005) -> np.ndarray:
    H, W = binary.shape
    out = binary.astype(np.uint8, copy=False)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(out, connectivity=8)
    thr = max(1, int(min_area_frac * H * W))
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < thr:
            out[labels == i] = 0
    k = _k_ellipse(3, 3)
    er = cv.erode(out, k); dl = cv.dilate(er, k)
    preserve = (edge_map >= 0.7) & (out > 0)
    out = np.where(preserve, out, dl)
    return out.astype(np.uint8, copy=False)

def superpixel_majority_snap(img_rgb: np.ndarray, mask: np.ndarray, region_size: int=20,
                             compactness: float=10.0, tau: float=0.85, return_segmentation: bool=False) -> np.ndarray:
    seg = slic(img_rgb, n_segments=max(100, (img_rgb.shape[0]*img_rgb.shape[1])//(region_size*region_size)),
               compactness=compactness, start_label=0, channel_axis=-1)
    out = mask.copy()
    for sp_id in np.unique(seg):
        sp = (seg == sp_id)
        fg_ratio = float((mask[sp] > 0).mean())
        if fg_ratio >= tau:
            out[sp] = 1
        elif fg_ratio <= (1.0 - tau):
            out[sp] = 0
    if return_segmentation:
        return out, seg
    return out

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

# ---------- Multi-class wrapper ----------
def run_one_vs_rest(img_rgb: np.ndarray,
                    anns: np.ndarray,
                    gc_iters: int = 5,
                    tie_mode: str = "nearest-scribble",
                    geo_radius: int = 7,
                    edge_alpha: float = 4.0,
                    adaptive_edges: bool = True,
                    conf_tau: float = 0.75,
                    star_prior: bool = False,
                    star_tau_percentile: float = 80.0,
                    downscale: float = 1.0,
                    cleanup_area_frac: float = 0.0005,
                    superpixel_snap_flag: bool = False,
                    sp_region_size: int = 20,
                    sp_compactness: float = 10.0,
                    snap_guided: bool = True,
                    snap_r: int = 4,
                    snap_eps: float = 1e-3,
                    snap_thresh: float = 0.5,
                    eval_freeze_after: int = 1,
                    *,
                    edge_backend: str = "auto",
                    structured_model: Optional[str] = None,
                    texture_edges: bool = False,
                    roi_pad: int = 96,
                    use_roi: bool = True,
                    ms_init: bool = True,
                    ms_scale: float = 0.5,
                    debug: Optional[DebugRecorder]=None,
                    debug_save_masks: bool=True,
                    debug_plan: Optional[DebugPlan]=None) -> np.ndarray:

    plan = debug_plan or DebugPlan("")
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        return np.zeros((H, W), dtype=np.uint8)

    # edge map once per image
    E = get_edge_map(img_rgb, edge_backend=edge_backend,
                     structured_model=structured_model, use_texture=texture_edges,
                     dbg=(debug if plan.edges else None), tag="00_edge_map.png")

    fg_masks: Dict[int, np.ndarray] = {}
    for c in classes:
        seeds_fg = (anns == c)
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))

        want_class_debug = (plan.geodesic or plan.grabcut or plan.roi or plan.ms_init or plan.post or plan.all)
        sub = DebugRecorder(debug.root if (debug and want_class_debug) else None, f"class_{c:02d}")

        pred = grabcut_once_extended(
            img_rgb, seeds_bg, seeds_fg,
            iters=int(gc_iters),
            edge_map=E,
            geo_radius=int(geo_radius), edge_alpha=float(edge_alpha),
            adaptive_edges=bool(adaptive_edges), conf_tau=float(conf_tau),
            star_prior=bool(star_prior), star_tau_percentile=float(star_tau_percentile),
            downscale=float(downscale), eval_freeze_after=int(eval_freeze_after),
            roi_pad=int(roi_pad), use_roi=bool(use_roi),
            ms_init=bool(ms_init), ms_scale=float(ms_scale),
            dbg=(sub if want_class_debug else None), dbg_prefix="01_",
            save_edges=False,                 # global edge saved above if requested
            save_geo=plan.geodesic,
            save_roi=plan.roi,
            save_ms=plan.ms_init,
            save_gc=plan.grabcut
        )

        # cleanup
        pred = cleanup_mask(pred, E, min_area_frac=float(cleanup_area_frac))
        if plan.post and sub: sub.save_gray("02_cleanup.png", pred.astype(np.float32))

        # superpixel snap
        if superpixel_snap_flag:
            pred = superpixel_majority_snap(img_rgb, pred, region_size=int(sp_region_size),
                                            compactness=float(sp_compactness), tau=0.85)
            if plan.post and sub: sub.save_gray("03_superpixel_snap.png", pred.astype(np.float32))

        # guided snap
        if snap_guided:
            if plan.post and sub:
                pred, q = guided_snap(img_rgb, pred, r=int(snap_r), eps=float(snap_eps),
                                      thresh=float(snap_thresh), return_soft=True)
                sub.save_gray("04_guided_soft.png", q)
                sub.save_gray("05_guided_mask.png", pred.astype(np.float32))
            else:
                pred = guided_snap(img_rgb, pred, r=int(snap_r), eps=float(snap_eps),
                                   thresh=float(snap_thresh), return_soft=False)

        fg_masks[c] = pred
        if plan.post and sub:
            sub.save_overlay("06_class_mask_overlay.png", img_rgb, pred, color=(255,255,255), alpha=0.5)
            sub.dump_meta()

    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    overlap_count = stack.sum(axis=2)
    any_overlap = (overlap_count > 1).any()
    if plan.overlap and debug:
        debug.save_gray("90_overlap_count.png", overlap_count.astype(np.float32))

    final = np.zeros((H, W), dtype=np.uint8)
    if not any_overlap or tie_mode != "nearest-scribble":
        for c in classes:
            final[fg_masks[c] > 0] = max(1, c - 1)
    else:
        overlap_mask = overlap_count > 1
        classes_for_dt = [c for c in classes if np.any((fg_masks[c] > 0) & overlap_mask)]
        if len(classes_for_dt) >= 1:
            dstack = []
            for c in classes_for_dt:
                seeds = (anns == c)
                ones = np.ones_like(seeds, dtype=np.uint8)
                ones[seeds] = 0
                d = cv.distanceTransform(ones, cv.DIST_L2, 3)
                dstack.append(d)
            dstack = np.stack(dstack, axis=2)
            arg = np.argmin(dstack, axis=2)

            for c in classes:
                m = (fg_masks[c] > 0) & (~overlap_mask)
                final[m] = max(1, c - 1)
            for idx, c in enumerate(classes_for_dt):
                m = overlap_mask & (arg == idx)
                final[m] = max(1, c - 1)
        else:
            for c in classes:
                final[fg_masks[c] > 0] = max(1, c - 1)

    if debug and (plan.post or plan.overlap or plan.all):
        debug.save_overlay("99_final_foreground_overlay.png", img_rgb, (final>0).astype(np.uint8), color=(255,255,255), alpha=0.4)

    return final

# ---------- CLI ----------
def parse_args(argv=None):
    ap = argparse.ArgumentParser("Modern GrabCut (non-DL) — upgraded but fast + component debug")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--anns_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=0, help="0 = all")
    ap.add_argument("--start_one", type=int, default=1, help="1-based index of first file")

    # Core GC and overlap
    ap.add_argument("--gc_iters", type=int, default=3)  # tuned
    ap.add_argument("--tie_mode", type=str, default="nearest-scribble", choices=["nearest-scribble","priority"])

    # Seed expansion / priors (tuned)
    ap.add_argument("--geo_radius", type=int, default=12)
    ap.add_argument("--edge_alpha", type=float, default=3.0)
    ap.add_argument("--adaptive_edges", action="store_true", default=True)
    ap.add_argument("--conf_tau", type=float, default=0.85)
    ap.add_argument("--star_prior", action="store_true", default=True)
    ap.add_argument("--star_tau_percentile", type=float, default=82.0)

    # Pyramid + ROI (tuned)
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--ms_init", action="store_true", default=True)
    ap.add_argument("--ms_scale", type=float, default=0.6)
    ap.add_argument("--roi_pad", type=int, default=96)
    ap.add_argument("--no_roi", action="store_true", default=False)

    # Edges (tuned: structured+NMS)
    ap.add_argument("--edge_backend", type=str, default="structured", choices=["auto","composite","structured"])
    ap.add_argument("--structured_model", type=str, default="./third_party/sed/model.yml.gz",
                    help="path to structured forest model.yml.gz")
    ap.add_argument("--texture_edges", action="store_true", default=False)

    # Post-processing (tuned)
    ap.add_argument("--cleanup_area_frac", type=float, default=0.0005)
    ap.add_argument("--superpixel_snap", dest="superpixel_snap_flag", action="store_true", default=True)
    ap.add_argument("--sp_region_size", type=int, default=18)
    ap.add_argument("--sp_compactness", type=float, default=12.0)
    ap.add_argument("--snap_guided", action="store_true", default=True)
    ap.add_argument("--snap_r", type=int, default=4)
    ap.add_argument("--snap_eps", type=float, default=1e-3)
    ap.add_argument("--snap_thresh", type=float, default=0.50)

    # OpenCV refinement
    ap.add_argument("--eval_freeze_after", type=int, default=1)

    # Debug outputs
    ap.add_argument("--debug_dir", type=str, default="", help="folder to save per-image debug artifacts")
    ap.add_argument("--debug_comp", type=str, default="",
        help=("Comma-separated components to debug: edges, geodesic (geo), grabcut (gc), "
              "roi, ms_init (ms/pyramid), post, overlap, all/full. Empty = no debug."))
    ap.add_argument("--debug_save_geo", action="store_true", default=True,
                    help="(kept for compat; gated by --debug_comp)")
    ap.add_argument("--debug_save_masks", action="store_true", default=True,
                    help="(kept for compat; gated by --debug_comp)")

    return ap.parse_args(argv)


def main(argv=None):
    cv.setUseOptimized(True)
    args = parse_args(argv)

    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = DebugPlan(args.debug_comp)
    debug_root = Path(args.debug_dir) if (args.debug_dir and plan.enabled) else None

    ann_files = sorted([p for p in anns_dir.iterdir() if p.suffix.lower() in (".png",".bmp",".tif",".tiff",".npy")])
    if args.num_images > 0:
        ann_files = ann_files[args.start_one - 1: args.start_one - 1 + args.num_images]

    processed, skipped = 0, 0
    times_ms: List[float] = []
    iterator = tqdm(ann_files, unit="img", desc="ModernGrabCut")

    for ann_path in iterator:
        base = base_from_ann_name(ann_path.stem)
        img_path = find_image(base, images_dir)
        if img_path is None:
            tqdm.write(f"[SKIP] {ann_path.name} (image not found)")
            skipped += 1
            continue
        dbg = DebugRecorder(debug_root, base) if debug_root else None
        try:
            t0 = perf_counter()
            img = load_img(img_path)
            anns = load_anns(ann_path)
            if anns.shape[:2] != img.shape[:2]:
                anns = cv.resize(anns.astype(np.int32), (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

            if dbg and plan.enabled:
                dbg.note(image=str(img_path), ann=str(ann_path), params=vars(args), debug_comp=args.debug_comp)

            pred = run_one_vs_rest(
                img, anns,
                gc_iters=int(args.gc_iters),
                tie_mode=args.tie_mode,
                geo_radius=int(args.geo_radius),
                edge_alpha=float(args.edge_alpha),
                adaptive_edges=bool(args.adaptive_edges),
                conf_tau=float(args.conf_tau),
                star_prior=bool(args.star_prior),
                star_tau_percentile=float(args.star_tau_percentile),
                downscale=float(args.downscale),
                cleanup_area_frac=float(args.cleanup_area_frac),
                superpixel_snap_flag=bool(args.superpixel_snap_flag),
                sp_region_size=int(args.sp_region_size),
                sp_compactness=float(args.sp_compactness),
                snap_guided=bool(args.snap_guided),
                snap_r=int(args.snap_r),
                snap_eps=float(args.snap_eps),
                snap_thresh=float(args.snap_thresh),
                eval_freeze_after=int(args.eval_freeze_after),
                edge_backend=args.edge_backend,
                structured_model=(args.structured_model if args.structured_model else None),
                texture_edges=bool(args.texture_edges),
                roi_pad=int(args.roi_pad),
                use_roi=(not bool(args.no_roi)),
                ms_init=bool(args.ms_init),
                ms_scale=float(args.ms_scale),
                debug=dbg,
                debug_save_masks=bool(args.debug_save_masks),
                debug_plan=plan
            )

            save_indexed_png(pred, str(out_dir / f"{base}_index.png"))
            dt = (perf_counter() - t0) * 1000.0
            times_ms.append(dt)
            processed += 1
            tqdm.write(f"[OK] {base} ({dt:.1f} ms) -> {base}_index.png")

            if dbg and plan.enabled:
                dbg.save_gray("98_final_mask_fg.png", (pred>0).astype(np.float32))
                dbg.dump_meta()
        except Exception as e:
            tqdm.write(f"[ERR] {base}: {e}")

    summary = {
        "output_dir": str(out_dir),
        "processed": processed,
        "skipped": skipped,
        "params": {
            "gc_iters": int(args.gc_iters),
            "tie_mode": args.tie_mode,
            "geo_radius": int(args.geo_radius),
            "edge_alpha": float(args.edge_alpha),
            "adaptive_edges": bool(args.adaptive_edges),
            "conf_tau": float(args.conf_tau),
            "star_prior": bool(args.star_prior),
            "star_tau_percentile": float(args.star_tau_percentile),
            "downscale": float(args.downscale),
            "cleanup_area_frac": float(args.cleanup_area_frac),
            "superpixel_snap": bool(args.superpixel_snap_flag),
            "sp_region_size": int(args.sp_region_size),
            "sp_compactness": float(args.sp_compactness),
            "snap_guided": bool(args.snap_guided),
            "snap_r": int(args.snap_r),
            "snap_eps": float(args.snap_eps),
            "snap_thresh": float(args.snap_thresh),
            "eval_freeze_after": int(args.eval_freeze_after),
            "edge_backend": args.edge_backend,
            "structured_model": bool(args.structured_model),
            "texture_edges": bool(args.texture_edges),
            "roi_pad": int(args.roi_pad),
            "use_roi": (not bool(args.no_roi)),
            "ms_init": bool(args.ms_init),
            "ms_scale": float(args.ms_scale),
            "debug_comp": args.debug_comp
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None),
        "fastgeo": True
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
