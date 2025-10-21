# -*- coding: utf-8 -*-
"""
Debug visualizer for our GrabCut stack (seed preprocessing focus) with
class-labeled refined seed maps.

Outputs include:
  Top-level (shared across classes):
    - edge_map.png
    - edge_cost_map.png
    - edge_cost_map.npy
    - feats_<CS>.png ...
    - seeds_fg_indexed.png         (initial scribbles with original class ids)
    - seeds_bg_indexed.png         (initial background scribbles as label 1)
    - refined_indexed.npy          (unified refined map: 0=unlabeled, 1=BG, >1=FG classes by min DF)
    - refined_indexed.png          (visualization: 0=white, 1=black, >1=VOC palette)
  Per-class (inside class_<id>/):
    - seeds_fg.png, seeds_bg.png   (binary)
    - df.png, db.png               (geodesic distances)
    - conf_posterior.png (if available), conf_mask.png
    - refined_seeds_fg.png, refined_seeds_bg.png (binary)

Run examples:
  # One selected class (15)
  python diagrams.py --image img.jpg --anns anns.npy --out out --class_id 15 \
    --adaptive_edges --edge_alpha 3.0 --conf_tau 0.75 \
    --conf_color_space cielab --color_spaces ruderman_lab oklab jzczhz

  # All classes found in anns (>1)
  python diagrams.py --image img.jpg --anns anns.npy --out out --all_classes \
    --adaptive_edges --edge_alpha 3.0 --conf_tau 0.75 \
    --conf_color_space cielab --color_spaces ruderman_lab oklab jzczhz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2 as cv
import sys, pathlib, json
from typing import Tuple, Optional, List

# Make mgc_core importable (same style as grabcut.py)
sys.path.append(str(pathlib.Path(__file__).parent / "mgc_core"))

# Project imports
from color_space import convert_color_space  # type: ignore
import mgc_core.modern_grabcut as mgc       # type: ignore
from mgc_core import fastgeo                # type: ignore


# ---------- I/O helpers ----------

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


def _load_img_rgb_u8(p: Path) -> np.ndarray:
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def _load_anns(p: Path) -> np.ndarray:
    """Accepts .npy, .png, .bmp, .tif, .tiff -> int32 indexed labels."""
    ext = p.suffix.lower()
    if ext == ".npy":
        a = np.load(p, mmap_mode="r")
        a = np.asarray(a, dtype=np.int32)
    elif ext in (".png", ".bmp", ".tif", ".tiff"):
        a = np.asarray(Image.open(p), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported annotation format: {ext}")
    return a

def _save_u8_gray(path: Path, arr: np.ndarray) -> None:
    """Save float/uint arrays as 8-bit grayscale with min-max normalization with NaN/Inf guard."""
    a = np.asarray(arr).astype(np.float32, copy=False)
    if not np.isfinite(a).any():
        a8 = np.zeros_like(a, dtype=np.uint8)
    else:
        a = np.nan_to_num(a, nan=0.0, posinf=float(np.max(a[np.isfinite(a)])), neginf=0.0)
        m, M = float(a.min()), float(a.max())
        if M - m < 1e-12:
            a8 = np.zeros_like(a, dtype=np.uint8)
        else:
            a8 = np.clip(255.0 * (a - m) / (M - m), 0, 255).astype(np.uint8)
    Image.fromarray(a8).save(path)


def save_binary_mask_with_palette(path: Path, mask: np.ndarray, class_id: int, palette: np.ndarray, fg_is_class: bool = True):
    """
    Save a binary mask as a color PNG: background=white, foreground=VOC palette color for class_id (or black for BG mask).
    If fg_is_class is True, FG is colored with palette[class_id], else FG is black.
    """
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)  # start with white
    m = mask.astype(bool)
    if fg_is_class:
        # Apply VOC offset: our class ids are 2.. -> VOC indices 1..
        # Guard for unexpected values (e.g., class 1 should not occur here but clamp anyway)
        pal_idx = int(class_id) - 1
        if pal_idx < 0:
            pal_idx = 0
        elif pal_idx > 255:
            pal_idx = 255
        out[m] = palette[pal_idx]
    else:
        # Background scribble: draw in black
        out[m] = [0, 0, 0]
    Image.fromarray(out).save(path)

def _save_indexed(path: Path, label_map: np.ndarray, with_palette: bool = False) -> None:
    """Save an int32 indexed label map as PNG (uint16 if needed, else uint8).
    If with_palette=True, apply VOC palette to the image."""
    lbl = np.asarray(label_map)
    if lbl.max() <= 255:
        out = lbl.astype(np.uint8)
        if with_palette:
            img = Image.fromarray(out)
            img = img.convert("P")
            img.putpalette(voc_palette().ravel().tolist())
            img.save(path)
        else:
            Image.fromarray(out).save(path)
    else:
        # fall back to 16-bit if class ids exceed 255 (palette not supported for 16-bit)
        out = lbl.astype(np.uint16)
        Image.fromarray(out).save(path)


def _save_indexed_with_viz(path_npy: Path, label_map: np.ndarray) -> None:
    """Save indexed label map as .npy and create a visualization PNG.
    
    The .npy contains int32 labels: 0=unlabeled, 1=background, >1=foreground classes.
    The visualization PNG shows: 0=pure white, 1=black, >1=VOC palette colors.
    
    VOC ground truth mapping:
      VOC 255 = unlabeled/ignore (but not pure white in standard palette)
      VOC 0   = background (black)  
      VOC 1-20 = foreground classes (palette colors)
    
    Our annotation mapping:
      0 = unlabeled (render as pure white [255, 255, 255])
      1 = background (maps to VOC 0 -> black)
      2-21 = foreground classes (map to VOC 1-20 -> palette colors)
    
    So we apply palette with index offset: palette[annotation_value - 1]
    """
    lbl = np.asarray(label_map, dtype=np.int32)
    
    # Save as .npy (preserves exact integer labels)
    np.save(path_npy, lbl)
    
    # Create visualization by applying VOC palette with -1 offset
    # This accounts for: our_annotation_id = voc_ground_truth_id + 1
    pal = voc_palette()
    
    # Override palette[255] to be pure white for unlabeled visualization
    pal[255] = [255, 255, 255]
    
    # Map annotation labels to VOC indices for palette lookup
    # 0 (unlabeled) -> 255 (now pure white)
    # 1 (background) -> 0 (black in VOC)  
    # 2+ (foreground) -> 1+ (VOC palette)
    viz_indexed = np.where(lbl == 0, 255, lbl - 1).astype(np.uint8)
    
    # Apply palette using PIL (same method as grabcut.py save_indexed_png)
    img = Image.fromarray(viz_indexed)
    img = img.convert("P")
    img.putpalette(pal.ravel())
    img_rgb = img.convert("RGB")
    
    # Save visualization
    viz_path = path_npy.with_suffix('.png')
    img_rgb.save(viz_path)


# ---------- Visualization helpers ----------

def _make_scribble_swatches(img_rgb_u8: np.ndarray, seeds_fg: np.ndarray, seeds_bg: np.ndarray, out_path: Path, tiles: int = 16) -> None:
    """Save a small grid of random FG/BG sample colors for quick visual inspection."""
    rng = np.random.default_rng(123)
    H, W = img_rgb_u8.shape[:2]
    sw = 24  # tile size
    cols = tiles
    rows = 2
    canvas = np.full((rows*sw, cols*sw, 3), 255, np.uint8)
    for r, mask in enumerate([seeds_fg, seeds_bg]):
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        idx = rng.choice(ys.size, size=cols, replace=True)
        for c, k in enumerate(idx):
            y, x = ys[k], xs[k]
            patch = img_rgb_u8[max(0, y-1):y+2, max(0, x-1):x+2]
            color = np.mean(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            canvas[r*sw:(r+1)*sw, c*sw:(c+1)*sw] = color[None, None, :]
    Image.fromarray(canvas).save(out_path)


def _lab_ab_scatter(img_rgb_u8: np.ndarray, seeds_fg: np.ndarray, seeds_bg: np.ndarray, out_path: Path) -> None:
    """Save FG/BG scribble points in Lab a–b plane (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except Exception:
        return  # silently skip if matplotlib not installed
    
    lab = cv.cvtColor(img_rgb_u8, cv.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    Af, Bf = A[seeds_fg], B[seeds_fg]
    Ab, Bb = A[seeds_bg], B[seeds_bg]
    
    if Af.size == 0 and Ab.size == 0:
        return  # no seeds to plot
    
    plt.figure(figsize=(4, 4), dpi=200)
    if Ab.size > 0:
        plt.scatter(Ab, Bb, s=2, c='k', alpha=0.25, label='BG')
    if Af.size > 0:
        plt.scatter(Af, Bf, s=2, c='r', alpha=0.6, label='FG')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.legend(loc='best', fontsize=6)
    plt.title('Scribble colors in Lab a–b plane')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _posterior_mosaic(img_rgb_u8: np.ndarray, conf_post_u8: np.ndarray, out_path: Path, block: int = 16, two_color: bool = False) -> None:
    """
    Downscale to blocks and visualize posterior per block.
    - If two_color=False: grayscale by mean posterior
    - If two_color=True: red/blue with opacity by posterior
    """
    H, W = conf_post_u8.shape
    h2, w2 = H // block, W // block
    if h2 == 0 or w2 == 0:
        h2 = max(1, H)
        w2 = max(1, W)
        block = 1
    # mean posterior per block
    small = cv.resize(conf_post_u8, (w2, h2), interpolation=cv.INTER_AREA)
    # upscale back as mosaic (nearest)
    mosaic = cv.resize(small, (W, H), interpolation=cv.INTER_NEAREST)

    if not two_color:
        # grayscale mosaic (0..255)
        Image.fromarray(mosaic).save(out_path)
        return

    # two-color overlay: FG=red, BG=blue, intensity by posterior
    mosaic_f = mosaic.astype(np.float32) / 255.0
    fg = np.stack([np.ones_like(mosaic_f), np.zeros_like(mosaic_f), np.zeros_like(mosaic_f)], axis=-1)  # red
    bg = np.stack([np.zeros_like(mosaic_f), np.zeros_like(mosaic_f), np.ones_like(mosaic_f)], axis=-1)  # blue
    rgb = (mosaic_f[..., None] * fg + (1.0 - mosaic_f[..., None]) * bg) * 255.0
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)


# ---------- Core computations ----------

def _compute_edge_cost_map(
    img_rgb_u8: np.ndarray,
    E: np.ndarray,
    adaptive_edges: bool,
    edge_alpha: float
) -> np.ndarray:
    """
    E_soft = (normalize+blur(E))**0.8
    alpha_map = edge_alpha * (0.35 + 0.65 * local_contrast(gray)) if adaptive else edge_alpha
    C = 1 + alpha_map * E_soft
    """
    E = E.astype(np.float32, copy=False)
    p95 = np.percentile(E, 95.0)
    if p95 > 1e-6:
        E = np.clip(E / p95, 0.0, 1.0)
    E = cv.GaussianBlur(E, (0, 0), 0.8)
    E_soft = np.power(E, 0.8, dtype=np.float32)

    if adaptive_edges:
        gray = cv.cvtColor(img_rgb_u8, cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lc = mgc.local_contrast(gray, r=3)
        alpha_map = float(edge_alpha) * (0.35 + 0.65 * lc)
        alpha_f32 = alpha_map.astype(np.float32, copy=False)
    else:
        alpha_f32 = np.full_like(E_soft, float(edge_alpha), dtype=np.float32)

    cost = 1.0 + alpha_f32 * E_soft
    return cost


def _geodesic_distance(cost_map: np.ndarray, seeds_bool: np.ndarray, eight_connected: bool=True) -> np.ndarray:
    """Compute geodesic distance via the fast C++ op; returns float64 HxW."""
    cost = cost_map.astype(np.float64, copy=False)
    seeds = seeds_bool.astype(np.uint8, copy=False)
    d = fastgeo.geodesic(cost, seeds, bool(eight_connected))
    return np.asarray(d, dtype=np.float64)


def _compute_confidence(img_for_conf: np.ndarray, seeds_fg: np.ndarray, seeds_bg: np.ndarray,
                        tau: float=0.75, return_score: bool=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Confidence gate using Gaussian color models fit on seeds.
    mgc.seeds_confidence_lab expects 3-channel uint8; internally converts to Lab.
    """
    conf_mask, conf_post = mgc.seeds_confidence_lab(
        img_for_conf, seeds_fg, seeds_bg, tau=float(tau), return_score=return_score
    )
    return conf_mask.astype(bool), (conf_post if return_score else None)


# ---------- Per-class processing ----------

def _process_one_class(
    c_sel: int,
    img_rgb_u8: np.ndarray,
    anns: np.ndarray,
    args,
    E: np.ndarray,
    cost: np.ndarray,
    out_root: Path
) -> dict:
    """
    Run the full visualization stack for a single class (one-vs-rest).
    Saves artifacts to out_root / f"class_{c_sel:02d}".
    Returns a dict with keys:
      'class_id', 'seeds_fg', 'seeds_bg', 'df', 'refined_fg', 'refined_bg'
    where arrays are bool or float as appropriate (for later aggregation).
    """
    out_dir = out_root / f"class_{c_sel:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seeds: 1=BG, 2..=classes (others act as BG for one-vs-rest)
    seeds_fg = (anns == c_sel)
    seeds_bg = (anns == 1) | ((anns > 1) & (anns != c_sel))


    # Save initial seeds (binary per-class) with palette logic
    pal = voc_palette()
    save_binary_mask_with_palette(out_dir / "seeds_fg.png", seeds_fg, c_sel, pal, fg_is_class=True)
    save_binary_mask_with_palette(out_dir / "seeds_bg.png", seeds_bg, c_sel, pal, fg_is_class=False)

    # Geodesic distances
    df = _geodesic_distance(cost, seeds_fg, eight_connected=True)
    db = _geodesic_distance(cost, seeds_bg, eight_connected=True)
    _save_u8_gray(out_dir / "df.png", df)
    _save_u8_gray(out_dir / "db.png", db)

    # Confidence image
    if args.conf_color_space.lower() in ("rgb", "srgb"):
        conf_img = img_rgb_u8
    else:
        conf_img = convert_color_space(img_rgb_u8, args.conf_color_space)

    # Confidence (posterior + mask)
    conf_mask, conf_post = _compute_confidence(
        conf_img, seeds_fg, seeds_bg, tau=float(args.conf_tau), return_score=True
    )
    if conf_post is not None:
        # Save grayscale posterior
        _save_u8_gray(out_dir / "conf_posterior.png", conf_post.astype(np.float32))
        # Save posterior mosaic visualizations
        conf_post_u8 = (np.clip(conf_post, 0.0, 1.0) * 255.0).astype(np.uint8)
        _posterior_mosaic(img_rgb_u8, conf_post_u8, out_dir / "conf_posterior_mosaic_gray.png", block=16, two_color=False)
        _posterior_mosaic(img_rgb_u8, conf_post_u8, out_dir / "conf_posterior_mosaic_color.png", block=16, two_color=True)
    
    # For conf_mask, just save as binary (not class-specific)
    Image.fromarray((conf_mask.astype(np.uint8) * 255)).save(out_dir / "conf_mask.png")
    
    # Save scribble color swatches for visual inspection
    _make_scribble_swatches(img_rgb_u8, seeds_fg, seeds_bg, out_dir / "scribble_swatches.png", tiles=16)
    
    # Save Lab a-b scatter plot of scribble colors
    _lab_ab_scatter(img_rgb_u8, seeds_fg, seeds_bg, out_dir / "scribble_lab_scatter.png")

    # Refined seeds (binary)
    seeds_bg2, seeds_fg2 = mgc.expand_seeds(
        img_rgb=conf_img,
        E=E,
        seeds_fg=seeds_fg.astype(bool),
        seeds_bg=seeds_bg.astype(bool),
        r_geo=int(args.geo_radius),
        edge_alpha=float(args.edge_alpha),
        adaptive=bool(args.adaptive_edges),
        conf_tau=float(args.conf_tau),
        dbg=None,
        save_geo=False
    )
    save_binary_mask_with_palette(out_dir / "refined_seeds_fg.png", seeds_fg2, c_sel, pal, fg_is_class=True)
    save_binary_mask_with_palette(out_dir / "refined_seeds_bg.png", seeds_bg2, c_sel, pal, fg_is_class=False)

    return {
        "class_id": int(c_sel),
        "seeds_fg": seeds_fg.astype(bool),
        "seeds_bg": seeds_bg.astype(bool),
        "df": df.astype(np.float32),
        "refined_fg": seeds_fg2.astype(bool),
        "refined_bg": seeds_bg2.astype(bool),
    }


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--anns", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)

    # Class & pipeline knobs
    ap.add_argument("--class_id", type=int, default=-1,
                    help="VOC-style class id to visualize (>1). If <0, picks the first present class >1.")
    ap.add_argument("--all_classes", action="store_true", default=False,
                    help="If set, visualize seed preprocessing for every present class (>1).")
    ap.add_argument("--geo_radius", type=int, default=12, help="Geodesic halo radius (pixels).")

    # Edge/cost config
    ap.add_argument("--edge_backend", default="structured", choices=["structured", "composite"],
                    help="Edge backend for get_edge_map.")
    ap.add_argument("--structured_model", default="./mgc_core/third_party/sed/model.yml.gz",
                    help="Path to Structured Forests model (if using 'structured').")
    ap.add_argument("--adaptive_edges", action="store_true", default=False,
                    help="Use local-contrast scaled alpha map.")
    ap.add_argument("--edge_alpha", type=float, default=3.0,
                    help="Base alpha for edge cost map.")

    # Confidence config
    ap.add_argument("--conf_tau", type=float, default=0.75,
                    help="Posterior threshold for confidence gating.")
    ap.add_argument("--conf_color_space", default="cielab",
                    help="Color space used to compute confidence (Lab-like space recommended).")

    # Feature dumps
    ap.add_argument("--color_spaces", nargs="+", default=["cielab", "oklab", "rgb"],
                    help="Additional color spaces to dump feature PNGs for.")

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load inputs
    img_rgb_u8 = _load_img_rgb_u8(args.image)
    anns = _load_anns(args.anns)
    H, W = img_rgb_u8.shape[:2]
    if anns.shape[:2] != (H, W):
        anns = cv.resize(anns.astype(np.int32), (W, H), interpolation=cv.INTER_NEAREST)

    # Determine present classes (>1 are foreground classes)
    present = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not present:
        raise ValueError("No foreground classes (>1) found in anns.")

    # --- Compute shared artifacts ONCE (edge & cost) ---
    E = mgc.get_edge_map(
        img_rgb_u8,
        edge_backend=args.edge_backend,
        structured_model=args.structured_model,
        use_texture=False,
        dbg=None,
        tag="edge_map.png"
    )
    _save_u8_gray(args.out / "edge_map.png", E)

    cost = _compute_edge_cost_map(
        img_rgb_u8=img_rgb_u8,
        E=E,
        adaptive_edges=bool(args.adaptive_edges),
        edge_alpha=float(args.edge_alpha)
    )
    _save_u8_gray(args.out / "edge_cost_map.png", cost.astype(np.float32))
    np.save(args.out / "edge_cost_map.npy", cost.astype(np.float32))

    # --- Feature dumps (global, not class-specific) ---
    feature_files = []
    for cs in args.color_spaces:
        feats_u8 = convert_color_space(img_rgb_u8, cs)
        filename = f"feats_{cs}.png"
        Image.fromarray(feats_u8).save(args.out / filename)
        feature_files.append(filename)

    # --- Build initial class-indexed scribble maps for reference ---
    seeds_fg_indexed = np.zeros((H, W), dtype=np.int32)
    seeds_bg_indexed = np.zeros((H, W), dtype=np.int32)
    seeds_bg_indexed[anns == 1] = 1
    # Put each class id where the scribble says it's that class (>1)
    for c in present:
        seeds_fg_indexed[anns == c] = c
    _save_indexed(args.out / "seeds_fg_indexed.png", seeds_fg_indexed)
    _save_indexed(args.out / "seeds_bg_indexed.png", seeds_bg_indexed)

    # --- Per-class visualization + collect for aggregation ---
    per_class_outputs: List[str] = []
    results: List[dict] = []

    if args.all_classes:
        classes_to_run = present
    else:
        c_sel = args.class_id if args.class_id > 1 else present[0]
        classes_to_run = [int(c_sel)]

    for c in classes_to_run:
        res = _process_one_class(c, img_rgb_u8, anns, args, E, cost, args.out)
        results.append(res)
        # record files (relative) for meta
        base = f"class_{c:02d}/"
        per_class_outputs.extend([
            base + "seeds_fg.png",
            base + "seeds_bg.png",
            base + "df.png",
            base + "db.png",
            base + "conf_mask.png",
            base + "conf_posterior.png",
            base + "conf_posterior_mosaic_gray.png",
            base + "conf_posterior_mosaic_color.png",
            base + "scribble_swatches.png",
            base + "scribble_lab_scatter.png",
            base + "refined_seeds_fg.png",
            base + "refined_seeds_bg.png",
        ])
        # conf_posterior may or may not exist
        # don't assert; meta stays simple

    # --- Aggregate refined seeds into a single unified class-labeled map ---
    # Start with label 0 (unlabeled/unknown)
    refined_indexed = np.zeros((H, W), dtype=np.int32)
    
    # First pass: mark refined background as label 1
    any_bg = np.zeros((H, W), dtype=bool)
    for res in results:
        any_bg |= res["refined_bg"]
    refined_indexed[any_bg] = 1
    
    # Second pass: assign foreground classes by minimal DF among refined candidates
    # This will overwrite background where foreground is closer
    df_best = np.full((H, W), np.inf, dtype=np.float32)
    for res in results:
        c = res["class_id"]
        fg = res["refined_fg"]        # bool
        df = res["df"]                # float32
        # pixels proposed as refined FG by this class
        cand = fg
        # better if strictly closer than current best
        better = cand & (df < df_best)
        refined_indexed[better] = c
        df_best[better] = df[better]

    # Save as .npy (0=unlabeled, 1=bg, >1=fg classes) + visualization PNG
    _save_indexed_with_viz(args.out / "refined_indexed.npy", refined_indexed)

    # --- Meta for reproducibility ---
    meta = {
        "image": str(args.image),
        "anns": str(args.anns),
        "classes_present": present,
        "classes_visualized": classes_to_run,
        "all_classes": bool(args.all_classes),
        "geo_radius": int(args.geo_radius),
        "edge_backend": args.edge_backend,
        "structured_model": str(args.structured_model),
        "adaptive_edges": bool(args.adaptive_edges),
        "edge_alpha": float(args.edge_alpha),
        "conf_tau": float(args.conf_tau),
        "conf_color_space": args.conf_color_space,
        "color_spaces": args.color_spaces,
        "top_level_outputs": [
            "edge_map.png",
            "edge_cost_map.png",
            "edge_cost_map.npy",
            "seeds_fg_indexed.png",
            "seeds_bg_indexed.png",
            "refined_indexed.npy",
            "refined_indexed.png",
            *feature_files
        ],
        "per_class_outputs": per_class_outputs,
    }
    with open(args.out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.all_classes:
        print(f"[ok] Saved ALL classes (with refined_indexed.npy + visualization) to: {args.out.resolve()}")
    else:
        print(f"[ok] Saved selected class and refined_indexed.npy + visualization to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
