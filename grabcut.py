# Filename: grabcut.py
# -*- coding: utf-8 -*-

"""GrabCut batch CLI for one vs rest segmentation with optional ensemble over three color spaces.
This version moves majority voting from per-class binary masks to voting over final indexed masks.
Indexed masks now follow app/GT convention: 0 background, foreground labels equal class IDs (2..20 -> 2..20).
Minimal flags: --images_dir, --anns_dir, --output_dir. To enable ensemble, add --enable_majority_vote.
For three-way label ties, use --ensemble_label_tie_strategy [first, second, third]."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "mgc_core"))
from mgc_api import mgc_refine_seeds, mgc_post_smooth_mask

# ---------- constants / palette ----------
NUM_VOC_CLASSES = 21
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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

def save_indexed_png(mask_2d: np.ndarray, path: str) -> None:
    img = Image.fromarray(mask_2d.astype(np.uint8))
    img = img.convert("P")
    img.putpalette(voc_palette().ravel())
    img.save(path)

# ---------- I/O ----------

# --- majority vote on indexed masks (2D uint8) ---
def majority_vote_indexed(a, b, c, tie_pref=0):
    """Majority vote over three 2D uint8 indexed masks.
    tie_pref, 0 choose first, 1 choose second, 2 choose third, for three way ties."""
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(f"Shape mismatch in majority_vote_indexed, got {a.shape}, {b.shape}, {c.shape}")
    a = a.astype("uint8", copy=False)
    b = b.astype("uint8", copy=False)
    c = c.astype("uint8", copy=False)
    eq_ab = (a == b)
    eq_ac = (a == c)
    eq_bc = (b == c)
    out = a.copy()
    mask_ab = eq_ab
    out[mask_ab] = a[mask_ab]
    mask_ac = eq_ac & (~mask_ab)
    out[mask_ac] = a[mask_ac]
    mask_bc = eq_bc & (~(mask_ab | mask_ac))
    out[mask_bc] = b[mask_bc]
    mask_tie = ~(mask_ab | mask_ac | mask_bc)
    if mask_tie.any():
        if tie_pref == 0:
            out[mask_tie] = a[mask_tie]
        elif tie_pref == 1:
            out[mask_tie] = b[mask_tie]
        else:
            out[mask_tie] = c[mask_tie]
    return out


def load_img(p: Path) -> np.ndarray:
    """Load RGB uint8 image.
    Uses PIL conversion to RGB then numpy array view.
    """
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def load_anns(p: Path) -> np.ndarray:
    """Load annotations as int32 array.
    Accepts .npy, .png, .bmp, .tif, .tiff. Uses memory-mapped loading for .npy.
    Includes bounds checking for expected range [0, NUM_VOC_CLASSES].
    """
    ext = p.suffix.lower()
    if ext == ".npy":
        # Reads the .npy file as a memory-mapped array to avoid loading the entire file into memory at once.
        a = np.load(p, mmap_mode="r")
        
        # Converts the memory-mapped array to a regular NumPy array of type int32. Not 
        a = np.asarray(a, dtype=np.int32)
    elif ext in (".png", ".bmp", ".tif", ".tiff"):
        a = np.asarray(Image.open(p).convert("P"), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported annotation format: {ext}")

    a_min = int(a.min()) if a.size else 0
    a_max = int(a.max()) if a.size else 0
    if a_min < 0 or a_max > NUM_VOC_CLASSES:
        warnings.warn(
            f"Annotation values out of expected range [0, {NUM_VOC_CLASSES}]: [{a_min}, {a_max}]"
        )
    return a


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


# ---------- color-space helpers moved to color_space.py ----------
from color_space import convert_color_space  # type: ignore


# ---------- OpenCV GrabCut, single call ----------

def opencv_grabcut_once(img_feats_u8: np.ndarray,
                        seeds_bg: np.ndarray,
                        seeds_fg: np.ndarray,
                        iters: int = 2,
                        return_models: bool = False,
                        return_mask_states: bool = False
                        ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Input error handling. Ensures image features are uint8 and have three channels, and that seed masks match image dimensions. If no foreground seeds are provided, returns an empty mask immediately.
    """
    if img_feats_u8.dtype != np.uint8:
        raise TypeError(
            f"GrabCut requires uint8 features. Got {img_feats_u8.dtype}. "
            "Ensure the color converter scales data to [0, 255] before calling."
        )

    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 topology, got shape {img_feats_u8.shape}")

    H, W, _ = img_feats_u8.shape

    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError(
            f"Seed masks must match image size, got {seeds_bg.shape} and {seeds_fg.shape}, expected {(H, W)}"
        )

    if not np.any(seeds_fg):
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty

    """
    Initialize annotation mask with OpenCV's expected labels: GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3. In this case, we fill the annotation mask with 2 (probable background) and then set the firm background seeds to 0 and firm foreground seeds to 1.

    Follows the same step by step process as seen in the demo below with GC_INIT_WITH_MASK:
    Source: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_grabcut.html
    """
    mask = np.full((H, W), cv.GC_PR_BGD, dtype=np.uint8)
    mask[seeds_bg] = cv.GC_BGD
    mask[seeds_fg] = cv.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv.grabCut(img_feats_u8, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

    """
    Create the binary mask by asking whether the resulting mask is either GC_FGD (1) or GC_PR_FGD (3), which we treat as foreground, and everything else as background.
    """
    bin_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)
    return bin_mask


# ---------- multi class wrapper, one vs rest ----------

def run_one_vs_rest(img_feats_u8: np.ndarray,
                    img_rgb_u8: np.ndarray,
                    anns: np.ndarray,
                    gc_iters: int = 5,
                    tie_mode: str = "nearest-scribble",
                    collect_models: bool = False,
                    collect_binary_masks: bool = False,
                    collect_pre_refinement_masks: bool = False,
                    collect_superpixels: bool = False,
                    collect_guided_soft: bool = False):
    """
    For each present class c > 1:
      FG seeds = anns == c
      BG seeds = anns == 1 or anns > 1 and not equal to c
    Combine binary masks into a single index map where:
      output 0 = background, output 1..20 = foreground classes, map c -> c - 1.

    New options:
      collect_models=True returns models_by_class where models_by_class[c] = {'bgdModel': ..., 'fgdModel': ...}
      collect_binary_masks=True returns fg_masks (post-refinement binary masks)
      collect_pre_refinement_masks=True returns fg_masks_pre (pre-refinement binary masks, raw GrabCut output)
      collect_superpixels=True returns superpixel_segs (superpixel segmentation maps per class)
      collect_guided_soft=True returns guided_soft_masks (soft masks from guided filter per class)
      
    Returns based on flags:
      - Just final: final_mask
      - With models: (final_mask, models_by_class)
      - With binary_masks: (final_mask, fg_masks)
      - With pre_refinement: (final_mask, fg_masks_pre)
      - With superpixels: (final_mask, superpixel_segs)
      - With guided_soft: (final_mask, guided_soft_masks)
      - Combinations return in order: (final_mask, [models_by_class], [fg_masks], [fg_masks_pre], [superpixel_segs], [guided_soft_masks])
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        empty = np.zeros((H, W), dtype=np.uint8)
        returns = [empty]
        if collect_models:
            returns.append({})
        if collect_binary_masks:
            returns.append({})
        if collect_pre_refinement_masks:
            returns.append({})
        if collect_superpixels:
            returns.append({})
        if collect_guided_soft:
            returns.append({})
        return tuple(returns) if len(returns) > 1 else empty

    fg_masks: Dict[int, np.ndarray] = {}
    fg_masks_pre: Dict[int, np.ndarray] = {}
    models_by_class: Dict[int, Dict[str, np.ndarray]] = {}
    superpixel_segs: Dict[int, np.ndarray] = {}
    guided_soft_masks: Dict[int, np.ndarray] = {}

    for c in classes:
        """
        Select the seeds for the current class as the only foreground annotation and
        the original background seeds plus the other foreground seeds as the background annotation
        """
        seeds_fg = (anns == c)
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))


        """
        Performs seed expansio via geodesic distance calculation and simplified GMM LAB confidence
        """
        seeds_fg, seeds_bg = mgc_refine_seeds(img_rgb_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, conf_img=img_feats_u8)
        
        """
        Passes color converted image and expanded seeds for GrabCut run
        """
        y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters)  # type: ignore
        
        """
        Performs post-refinement using Guided Image Filtering
        """
        y = mgc_post_smooth_mask(img_rgb_u8, y)
        fg_masks[c] = y # binary 0 or 1

    final = _combine_fg_masks_to_final(fg_masks, anns, tie_mode)
    
    # Build return tuple based on what was requested
    returns = [final]
    if collect_models:
        returns.append(models_by_class)
    if collect_binary_masks:
        returns.append(fg_masks)
    if collect_pre_refinement_masks:
        returns.append(fg_masks_pre)
    if collect_superpixels:
        returns.append(superpixel_segs)
    if collect_guided_soft:
        returns.append(guided_soft_masks)
    
    return tuple(returns) if len(returns) > 1 else final


def _combine_fg_masks_to_final(fg_masks: Dict[int, np.ndarray],
                               anns: np.ndarray,
                               tie_mode: str = "nearest-scribble") -> np.ndarray:
    """
    Combine per class binary masks into a final VOC index map, with tie handling.
    """
    classes = sorted(fg_masks.keys())
    if not classes:
        H, W = anns.shape
        return np.zeros((H, W), dtype=np.uint8)

    H, W = anns.shape
    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    overlap_count = stack.sum(axis=2)
    any_overlap = (overlap_count > 1).any()

    final = np.zeros((H, W), dtype=np.uint8)

    if not any_overlap or tie_mode != "nearest-scribble":
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1
        return final

    # nearest scribble for overlapped pixels
    overlap_mask = (overlap_count > 1)

    dist_to_scrib: Dict[int, np.ndarray] = {}
    classes_for_dt: List[int] = []
    for c in classes:
        if np.any(fg_masks[c] & overlap_mask):
            s = (anns == c).astype(np.uint8)
            if np.any(s):
                ones = np.ones_like(s, dtype=np.uint8)
                ones[s > 0] = 0
                d = cv.distanceTransform(ones, cv.DIST_L2, 3).astype(np.float32)
            else:
                d = np.full(s.shape, 1e6, dtype=np.float32)
            dist_to_scrib[c] = d
            classes_for_dt.append(c)

    if classes_for_dt:
        INF = 1e9
        dstack = np.stack(
            [np.where(fg_masks[c] > 0, dist_to_scrib[c], INF) for c in classes_for_dt],
            axis=2
        )
        arg = np.argmin(dstack, axis=2)

        for c in classes:
            m = (fg_masks[c] > 0) & (~overlap_mask)
            final[m] = c - 1

        for idx, c in enumerate(classes_for_dt):
            m = overlap_mask & (arg == idx)
            final[m] = c - 1
    else:
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1

    return final


# ---------- majority voting ensemble, one vs rest ----------

def run_one_vs_rest_majority_ensemble(img_rgb_u8: np.ndarray,
                                      anns: np.ndarray,
                                      trio: List[str],
                                      gc_iters: int = 5,
                                      tie_mode: str = "nearest-scribble",
                                      trio_parallel: bool = False,
                                      trio_workers: int = 0,
                                      label_tie_pref: int = 0) -> np.ndarray:
    """
    Majority voting ensemble over generated indexed masks, one per color space.
    For each color space, compute a full indexed mask via run_one_vs_rest, then vote on labels.
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((H, W), dtype=np.uint8)
    
    if len(trio) != 3:
        raise ValueError("Ensemble trio must have exactly three color spaces")
    
    workers = trio_workers if trio_workers and trio_workers > 0 else len(trio)

    def _predict_for_space(cs: str) -> np.ndarray:
        feats = convert_color_space(img_rgb_u8, cs)
        pred = run_one_vs_rest(feats, img_rgb_u8, anns, gc_iters=int(gc_iters), tie_mode=tie_mode)  # type: ignore
        return pred.astype(np.uint8, copy=False)
    
    """
    The entry point for the parallel processing of three color space branches
    """
    if trio_parallel:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_predict_for_space, cs) for cs in trio]
            preds = [f.result() for f in futures]

    else:
        preds = [_predict_for_space(cs) for cs in trio]
    out = majority_vote_indexed(preds[0], preds[1], preds[2], tie_pref=int(label_tie_pref))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_models_npz(base: str,
                    color_space: str,
                    out_dir: Path,
                    models_by_class: Dict[int, Dict[str, np.ndarray]],
                    meta: Optional[Dict[str, object]] = None) -> List[str]:
    """
    Save per-class models to individual NPZ files.

    File name pattern: {base}__{color_space}__c{class_id:02d}__models.npz
    Contents:
      - bgdModel, shape 1 by 65 float64
      - fgdModel, shape 1 by 65 float64
      - meta, dict stored as JSON string under key meta_json
    Returns list of filenames written.
    """
    written: List[str] = []
    _ensure_dir(out_dir)

    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    for c, d in models_by_class.items():
        bgm = np.asarray(d.get("bgdModel"))
        fgm = np.asarray(d.get("fgdModel"))

        if bgm.shape != (1, 65) or fgm.shape != (1, 65):
            continue

        fname = f"{base}__{color_space}__c{c:02d}__models.npz"
        fpath = out_dir / fname
        np.savez(fpath, bgdModel=bgm, fgdModel=fgm, meta_json=meta_json)
        written.append(fname)
    return written


def save_binary_mask_indexed(base: str,
                             color_space: str,
                             class_id: int,
                             binary_mask: np.ndarray,
                             out_dir: Path,
                             suffix: str = "binary") -> str:
    """
        Save a binary mask as an indexed PNG.
        The binary mask (0 or 1) is converted to indexed format where:
            - 0 stays as 0 (background)
            - 1 is mapped to the class label equal to `class_id` (app/GT convention)
    
    File name pattern: {base}__{color_space}__c{class_id:02d}__{suffix}.png
    suffix: "binary" for post-refinement, "pre_refine" for pre-refinement, etc.
    Returns the filename written.
    """
    _ensure_dir(out_dir)
    
    # Convert binary mask to indexed: 0 -> 0, 1 -> class_id
    indexed = np.where(binary_mask > 0, class_id, 0).astype(np.uint8)
    
    fname = f"{base}__{color_space}__c{class_id:02d}__{suffix}.png"
    fpath = out_dir / fname
    save_indexed_png(indexed, str(fpath))
    
    return fname


def save_superpixel_segmentation(base: str,
                                 color_space: str,
                                 class_id: int,
                                 superpixel_seg: np.ndarray,
                                 out_dir: Path,
                                 img_rgb: Optional[np.ndarray] = None) -> str:
    """
    Save superpixel segmentation as boundary visualization PNG and raw NPY.
    If img_rgb is provided, overlays superpixel boundaries on the original image.
    Otherwise creates a grayscale visualization with boundaries.
    
    File name pattern: 
      - {base}__{color_space}__c{class_id:02d}__superpixels.png (boundary visualization)
      - {base}__{color_space}__c{class_id:02d}__superpixels.npy (raw data)
    Returns the base filename written.
    """
    _ensure_dir(out_dir)
    
    # Save raw superpixel labels as NPY
    fname_npy = f"{base}__{color_space}__c{class_id:02d}__superpixels.npy"
    fpath_npy = out_dir / fname_npy
    np.save(fpath_npy, superpixel_seg.astype(np.int32))
    
    H, W = superpixel_seg.shape
    
    # Create boundary map by detecting edges between different superpixels
    sp_padded = np.pad(superpixel_seg, 1, mode='edge')
    boundaries = np.zeros((H, W), dtype=bool)
    boundaries |= (superpixel_seg != sp_padded[:-2, 1:-1])  # top
    boundaries |= (superpixel_seg != sp_padded[2:, 1:-1])   # bottom
    boundaries |= (superpixel_seg != sp_padded[1:-1, :-2])  # left
    boundaries |= (superpixel_seg != sp_padded[1:-1, 2:])   # right
    
    if img_rgb is not None and img_rgb.shape[:2] == (H, W):
        # Overlay boundaries on original RGB image
        sp_vis = img_rgb.copy()
        # Draw boundaries in bright cyan (easy to see on most images)
        sp_vis[boundaries] = [0, 255, 255]  # BGR format: cyan
    else:
        # Fallback: modulo-based grayscale coloring with white boundaries
        sp_vis_mod = ((superpixel_seg % 85) * 3).astype(np.uint8)
        sp_vis = cv.cvtColor(sp_vis_mod, cv.COLOR_GRAY2BGR)
        sp_vis[boundaries] = [255, 255, 255]  # White boundaries
    
    fname_png = f"{base}__{color_space}__c{class_id:02d}__superpixels.png"
    fpath_png = out_dir / fname_png
    cv.imwrite(str(fpath_png), sp_vis)
    
    return fname_png


def save_guided_soft_mask(base: str,
                         color_space: str,
                         class_id: int,
                         soft_mask: np.ndarray,
                         out_dir: Path) -> str:
    """
    Save the soft mask from guided filtering.
    The soft mask is a float32 array in range [0, 1], saved as both NPY and visualized as PNG.
    
    File name pattern:
      - {base}__{color_space}__c{class_id:02d}__guided_soft.png (visualization, 0-255)
      - {base}__{color_space}__c{class_id:02d}__guided_soft.npy (raw float32 data)
    Returns the base filename written.
    """
    _ensure_dir(out_dir)
    
    # Save raw float32 soft mask as NPY
    fname_npy = f"{base}__{color_space}__c{class_id:02d}__guided_soft.npy"
    fpath_npy = out_dir / fname_npy
    np.save(fpath_npy, soft_mask.astype(np.float32))
    
    # Create visualization: scale to 0-255
    soft_vis = np.clip(soft_mask * 255.0, 0, 255).astype(np.uint8)
    
    fname_png = f"{base}__{color_space}__c{class_id:02d}__guided_soft.png"
    fpath_png = out_dir / fname_png
    cv.imwrite(str(fpath_png), soft_vis)
    
    return fname_png


# ---------- worker for parallel batch ----------

def _process_single_image(ann_path: str,
                          images_dir: str,
                          output_dir: str,
                          color_space: str,
                          gc_iters: int,
                          tie_mode: str,
                          emit_models: bool,
                          models_dir: Optional[str],
                          emit_binary_masks: bool,
                          binary_masks_dir: Optional[str],
                          emit_pre_refinement_masks: bool,
                          pre_refinement_masks_dir: Optional[str],
                          emit_superpixels: bool,
                          superpixels_dir: Optional[str],
                          emit_guided_soft: bool,
                          guided_soft_dir: Optional[str],
                          enable_majority_vote: bool,
                          ensemble_trio: str,
                          trio_parallel: bool,
                          trio_workers: int,
                          ensemble_label_tie_strategy: str) -> Dict[str, object]:
    """Worker function to process a single image, safe for ProcessPoolExecutor."""
    ann_p = Path(ann_path)
    images_dir_p = Path(images_dir)
    out_dir_p = Path(output_dir)

    base = base_from_ann_name(ann_p.stem)
    img_path = find_image(base, images_dir_p)
    if img_path is None:
        return {"ok": False, "base": base, "reason": "image not found"}

    t0 = perf_counter()
    img_rgb = load_img(img_path)
    anns = load_anns(ann_p)

    # resize anns to match image if needed
    if anns.shape[:2] != img_rgb.shape[:2]:
        anns = cv.resize(anns.astype(np.int32),
                         (img_rgb.shape[1], img_rgb.shape[0]), # cv.resize expects (width, height)
                         interpolation=cv.INTER_NEAREST)

    # run either majority ensemble or single space
    if enable_majority_vote:
        trio = [s.strip() for s in (ensemble_trio if ensemble_trio else "ruderman_lab,oklab,jzczhz").split(",")]

        if len(trio) != 3:
            return {"ok": False, "base": base, "reason": "ensemble_trio must have exactly 3 entries"}
        
        """
        Get tie preference for three-way ties in majority voting, otherwise opt for first model.
        """
        tie_map = {"first": 0, "second": 1, "third": 2}
        label_tie_pref = tie_map.get(ensemble_label_tie_strategy, 0)

        pred = run_one_vs_rest_majority_ensemble(
            img_rgb_u8=img_rgb,
            anns=anns,
            trio=trio,
            gc_iters=int(gc_iters),
            tie_mode=tie_mode,
            trio_parallel=bool(trio_parallel),
            trio_workers=int(trio_workers) if trio_workers else 0,
            label_tie_pref=label_tie_pref
        )

        written = []
        binary_written = []
        pre_refinement_written = []
        superpixels_written = []
        guided_soft_written = []
    else:
        img_feats = convert_color_space(img_rgb, color_space)
        
        # Determine what to collect based on flags
        result = run_one_vs_rest(
            img_feats, img_rgb, anns,
            gc_iters=int(gc_iters),
            tie_mode=tie_mode,
            collect_models=emit_models,
            collect_binary_masks=emit_binary_masks,
            collect_pre_refinement_masks=emit_pre_refinement_masks,
            collect_superpixels=emit_superpixels,
            collect_guided_soft=emit_guided_soft
        )
        
        # Unpack result based on what was collected
        if isinstance(result, tuple):
            pred = result[0]
            idx = 1
            models_by_class = result[idx] if emit_models else {}
            if emit_models:
                idx += 1
            fg_masks = result[idx] if emit_binary_masks else {}
            if emit_binary_masks:
                idx += 1
            fg_masks_pre = result[idx] if emit_pre_refinement_masks else {}
            if emit_pre_refinement_masks:
                idx += 1
            superpixel_segs = result[idx] if emit_superpixels else {}
            if emit_superpixels:
                idx += 1
            guided_soft_masks = result[idx] if emit_guided_soft else {}
        else:
            pred = result
            models_by_class = {}
            fg_masks = {}
            fg_masks_pre = {}
            superpixel_segs = {}
            guided_soft_masks = {}
        
        # Save models if requested
        if emit_models:
            models_out_dir = Path(models_dir) if models_dir else (out_dir_p / "models")
            written = save_models_npz(
                base=base,
                color_space=color_space,
                out_dir=models_out_dir,
                models_by_class=models_by_class,
                meta={
                    "base": base,
                    "color_space": color_space,
                    "gc_iters": int(gc_iters),
                    "tie_mode": tie_mode,
                },
            )
        else:
            written = []
        
        # Save binary masks (post-refinement) if requested
        if emit_binary_masks:
            binary_out_dir = Path(binary_masks_dir) if binary_masks_dir else (out_dir_p / "binary_masks")
            binary_written = []
            for class_id, binary_mask in fg_masks.items():
                fname = save_binary_mask_indexed(
                    base=base,
                    color_space=color_space,
                    class_id=class_id,
                    binary_mask=binary_mask,
                    out_dir=binary_out_dir,
                    suffix="binary"
                )
                binary_written.append(fname)
        else:
            binary_written = []
        
        # Save pre-refinement masks if requested
        if emit_pre_refinement_masks:
            pre_refine_out_dir = Path(pre_refinement_masks_dir) if pre_refinement_masks_dir else (out_dir_p / "pre_refinement_masks")
            pre_refinement_written = []
            for class_id, binary_mask in fg_masks_pre.items():
                fname = save_binary_mask_indexed(
                    base=base,
                    color_space=color_space,
                    class_id=class_id,
                    binary_mask=binary_mask,
                    out_dir=pre_refine_out_dir,
                    suffix="pre_refine"
                )
                pre_refinement_written.append(fname)
        else:
            pre_refinement_written = []
        
        # Save superpixel segmentations if requested
        if emit_superpixels:
            superpixels_out_dir = Path(superpixels_dir) if superpixels_dir else (out_dir_p / "superpixels")
            superpixels_written = []
            for class_id, sp_seg in superpixel_segs.items():
                fname = save_superpixel_segmentation(
                    base=base,
                    color_space=color_space,
                    class_id=class_id,
                    superpixel_seg=sp_seg,
                    out_dir=superpixels_out_dir,
                    img_rgb=img_rgb  # Pass RGB image for overlay
                )
                superpixels_written.append(fname)
        else:
            superpixels_written = []
        
        # Save guided soft masks if requested
        if emit_guided_soft:
            guided_soft_out_dir = Path(guided_soft_dir) if guided_soft_dir else (out_dir_p / "guided_soft")
            guided_soft_written = []
            for class_id, soft_mask in guided_soft_masks.items():
                fname = save_guided_soft_mask(
                    base=base,
                    color_space=color_space,
                    class_id=class_id,
                    soft_mask=soft_mask,
                    out_dir=guided_soft_out_dir
                )
                guided_soft_written.append(fname)
        else:
            guided_soft_written = []

    out_path = out_dir_p / f"{base}_index.png"
    save_indexed_png(pred, str(out_path))

    dt = (perf_counter() - t0) * 1000.0
    return {"ok": True, "base": base, "ms": dt, "out": out_path.name, "models_written": written, "binary_masks_written": binary_written, "pre_refinement_masks_written": pre_refinement_written, "superpixels_written": superpixels_written, "guided_soft_written": guided_soft_written}


# ---------- CLI ----------

def parse_args(argv=None):
    ap = argparse.ArgumentParser("GrabCut batch CLI, OpenCV backend, one vs rest")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--anns_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=0, help="0 means all")
    ap.add_argument("--start_one", type=int, default=1, help="1 based index of first file")

    ap.add_argument("--gc_iters", type=int, default=5, help="Iterations for cv2.GrabCut, typical 1 to 5")
    ap.add_argument("--tie_mode", type=str, default="nearest-scribble",
                    choices=["nearest-scribble", "first-wins"],
                    help="How to resolve multi class overlaps")

    ap.add_argument("--color_space", type=str, default="rgb",
                    choices=[
                        "rgb", "hsv_conic", "cielab", "c02_scd", "c16_scd",
                        "oklab", "oklch", "jzazbz", "jzczhz",
                        "ictcp_pq", "xyz", "ycbcr_bt709", "srgb_linear" ,"ruderman_lab", "opponent"
                    ],
                    help="Input feature color space for the single space baseline path. Default is rgb.")

    # majority voting ensemble controls
    ap.add_argument("--enable_majority_vote", action="store_true",
                    help="Enable majority voting ensemble across a trio of color spaces for binary masks prior to class assignment.")
    ap.add_argument("--ensemble_trio", type=str, default="ruderman_lab,oklab,jzczhz",
                    help="Comma separated trio for majority voting, default ruderman_lab,oklab,jzczhz.")
    ap.add_argument("--ensemble_trio_parallel", type=str, default="auto", choices=["auto", "on", "off"],
                    help="Intra image trio parallelization. auto, parallelize trio when not running batch parallel, off in batch. on, always parallelize. off, never parallelize.")
    ap.add_argument("--ensemble_trio_workers", type=int, default=0,
                    help="Workers for intra image trio parallelization with threads, 0 means len(trio)")
    ap.add_argument("--ensemble_label_tie_strategy", type=str, default="first",
                    choices=["first", "second", "third"],
                    help="When indexed labels from three color spaces all disagree at a pixel, choose first, second, or third.")

    ap.add_argument("--emit_models", action="store_true",
                    help="When set, save per class bgdModel and fgdModel NPZ files for the single space path.")
    ap.add_argument("--models_dir", type=str, default="",
                    help="Optional output directory for NPZ model files, defaults to output_dir slash models")

    ap.add_argument("--emit_binary_masks", action="store_true",
                    help="When set, save per class binary masks (post-refinement) as indexed PNG files for the single space path.")
    ap.add_argument("--binary_masks_dir", type=str, default="",
                    help="Optional output directory for binary mask PNG files, defaults to output_dir slash binary_masks")

    ap.add_argument("--emit_pre_refinement_masks", action="store_true",
                    help="When set, save per class pre-refinement masks (raw GrabCut output before superpixel refinement) for comparison.")
    ap.add_argument("--pre_refinement_masks_dir", type=str, default="",
                    help="Optional output directory for pre-refinement mask PNG files, defaults to output_dir slash pre_refinement_masks")

    ap.add_argument("--emit_superpixels", action="store_true",
                    help="When set, save per class superpixel segmentations from the superpixel boundary snapping phase.")
    ap.add_argument("--superpixels_dir", type=str, default="",
                    help="Optional output directory for superpixel segmentation files, defaults to output_dir slash superpixels")

    ap.add_argument("--emit_guided_soft", action="store_true",
                    help="When set, save per class soft masks from the guided filtering phase before thresholding.")
    ap.add_argument("--guided_soft_dir", type=str, default="",
                    help="Optional output directory for guided soft mask files, defaults to output_dir slash guided_soft")

    ap.add_argument("--parallel", action="store_true", help="Enable parallel processing of images")
    ap.add_argument("--max_workers", type=int, default=0, help="Workers for parallel mode, 0 picks os.cpu_count()")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    label_tie_strategy = args.ensemble_label_tie_strategy
    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_out_dir = Path(args.models_dir) if args.models_dir else (out_dir / "models")
    if args.emit_models and not args.enable_majority_vote:
        models_out_dir.mkdir(parents=True, exist_ok=True)

    binary_masks_out_dir = Path(args.binary_masks_dir) if args.binary_masks_dir else (out_dir / "binary_masks")
    if args.emit_binary_masks and not args.enable_majority_vote:
        binary_masks_out_dir.mkdir(parents=True, exist_ok=True)

    pre_refinement_masks_out_dir = Path(args.pre_refinement_masks_dir) if args.pre_refinement_masks_dir else (out_dir / "pre_refinement_masks")
    if args.emit_pre_refinement_masks and not args.enable_majority_vote:
        pre_refinement_masks_out_dir.mkdir(parents=True, exist_ok=True)

    superpixels_out_dir = Path(args.superpixels_dir) if args.superpixels_dir else (out_dir / "superpixels")
    if args.emit_superpixels and not args.enable_majority_vote:
        superpixels_out_dir.mkdir(parents=True, exist_ok=True)

    guided_soft_out_dir = Path(args.guided_soft_dir) if args.guided_soft_dir else (out_dir / "guided_soft")
    if args.emit_guided_soft and not args.enable_majority_vote:
        guided_soft_out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted([p for p in anns_dir.iterdir()
                        if p.suffix.lower() in (".npy", ".png", ".bmp", ".tif", ".tiff")])
    total = len(ann_files)
    if total == 0:
        print(json.dumps({"error": "no annotations found", "anns_dir": str(anns_dir)}))
        return

    if args.start_one is not None and args.start_one > 1:
        ann_files = ann_files[args.start_one - 1:]
    if args.num_images and args.num_images > 0:
        ann_files = ann_files[: args.num_images]

    processed, skipped = 0, 0
    times_ms: List[float] = []

    # Decide intra image trio parallelization for this run
    if args.ensemble_trio_parallel == "on":
        trio_parallel_flag = True
    elif args.ensemble_trio_parallel == "off":
        trio_parallel_flag = False
    else:
        # auto
        trio_parallel_flag = not bool(args.parallel)

    if args.parallel:
        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process_single_image,
                    str(ann_path), str(images_dir), str(out_dir),
                    str(args.color_space), int(args.gc_iters), str(args.tie_mode),
                    bool(args.emit_models), str(models_out_dir) if args.emit_models and not args.enable_majority_vote else None,
                    bool(args.emit_binary_masks), str(binary_masks_out_dir) if args.emit_binary_masks and not args.enable_majority_vote else None,
                    bool(args.emit_pre_refinement_masks), str(pre_refinement_masks_out_dir) if args.emit_pre_refinement_masks and not args.enable_majority_vote else None,
                    bool(args.emit_superpixels), str(superpixels_out_dir) if args.emit_superpixels and not args.enable_majority_vote else None,
                    bool(args.emit_guided_soft), str(guided_soft_out_dir) if args.emit_guided_soft and not args.enable_majority_vote else None,
                    bool(args.enable_majority_vote), str(args.ensemble_trio),
                    bool(trio_parallel_flag) and bool(args.enable_majority_vote),
                    int(args.ensemble_trio_workers),
                    str(label_tie_strategy)
                ): ann_path for ann_path in ann_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), unit="img", desc="GrabCut[par]"):
                try:
                    res = fut.result()
                    if res.get("ok"):
                        processed += 1
                        times_ms.append(float(res.get("ms", 0.0)))
                        msg = f"[OK] {res.get('base')} ({res.get('ms'):.1f} ms) -> {res.get('out')}"
                        written = res.get("models_written")
                        if written:
                            msg += f", models: {len(written)} files"
                        binary_written = res.get("binary_masks_written")
                        if binary_written:
                            msg += f", binary_masks: {len(binary_written)} files"
                        pre_refine_written = res.get("pre_refinement_masks_written")
                        if pre_refine_written:
                            msg += f", pre_refine: {len(pre_refine_written)} files"
                        superpixels_written = res.get("superpixels_written")
                        if superpixels_written:
                            msg += f", superpixels: {len(superpixels_written)} files"
                        guided_soft_written = res.get("guided_soft_written")
                        if guided_soft_written:
                            msg += f", guided_soft: {len(guided_soft_written)} files"
                        tqdm.write(msg)
                    else:
                        skipped += 1
                        tqdm.write(f"[SKIP] {Path(futures[fut]).name} {res.get('reason')}")
                except Exception as e:
                    skipped += 1
                    ann_path = futures[fut]
                    tqdm.write(f"[SKIP] {ann_path.name} {e}")
    else:
        iterator = tqdm(ann_files, unit="img", desc="GrabCut")
        for ann_path in iterator:
            base = base_from_ann_name(ann_path.stem)
            img_path = find_image(base, images_dir)
            if img_path is None:
                tqdm.write(f"[SKIP] {ann_path.name} image not found")
                skipped += 1
                continue

            try:
                t0 = perf_counter()
                img_rgb = load_img(img_path)
                anns = load_anns(ann_path)

                if anns.shape[:2] != img_rgb.shape[:2]:
                    anns = cv.resize(anns.astype(np.int32),
                                     (img_rgb.shape[1], img_rgb.shape[0]),
                                     interpolation=cv.INTER_NEAREST)

                written = []
                binary_written = []
                pre_refinement_written = []
                superpixels_written = []
                guided_soft_written = []
                if args.enable_majority_vote:
                    trio = [s.strip() for s in (args.ensemble_trio if args.ensemble_trio else "jzazbz,jzczhz,rgb").split(",")]
                    if len(trio) != 3:
                        raise ValueError("ensemble_trio must have exactly three comma separated color spaces")
                    tie_map = {"first": 0, "second": 1, "third": 2}
                    label_tie_pref = tie_map.get(args.ensemble_label_tie_strategy, 0)
                    pred = run_one_vs_rest_majority_ensemble(
                        img_rgb_u8=img_rgb,
                        anns=anns,
                        trio=trio,
                        gc_iters=int(args.gc_iters),
                        tie_mode=args.tie_mode,
                        trio_parallel=bool(trio_parallel_flag),
                        trio_workers=int(args.ensemble_trio_workers),
                        label_tie_pref=label_tie_pref
                    )
                else:
                    img_feats = convert_color_space(img_rgb, args.color_space)
                    
                    # Call run_one_vs_rest with appropriate flags
                    result = run_one_vs_rest(
                        img_feats, img_rgb, anns,
                        gc_iters=int(args.gc_iters),
                        tie_mode=args.tie_mode,
                        collect_models=args.emit_models,
                        collect_binary_masks=args.emit_binary_masks,
                        collect_pre_refinement_masks=args.emit_pre_refinement_masks,
                        collect_superpixels=args.emit_superpixels,
                        collect_guided_soft=args.emit_guided_soft
                    )
                    
                    # Unpack result based on what was collected
                    if isinstance(result, tuple):
                        pred = result[0]
                        idx = 1
                        models_by_class = result[idx] if args.emit_models else {}
                        if args.emit_models:
                            idx += 1
                        fg_masks = result[idx] if args.emit_binary_masks else {}
                        if args.emit_binary_masks:
                            idx += 1
                        fg_masks_pre = result[idx] if args.emit_pre_refinement_masks else {}
                        if args.emit_pre_refinement_masks:
                            idx += 1
                        superpixel_segs = result[idx] if args.emit_superpixels else {}
                        if args.emit_superpixels:
                            idx += 1
                        guided_soft_masks = result[idx] if args.emit_guided_soft else {}
                    else:
                        pred = result
                        models_by_class = {}
                        fg_masks = {}
                        fg_masks_pre = {}
                        superpixel_segs = {}
                        guided_soft_masks = {}
                    
                    # Save models if requested
                    if args.emit_models:
                        written = save_models_npz(
                            base=base,
                            color_space=args.color_space,
                            out_dir=models_out_dir,
                            models_by_class=models_by_class,
                            meta={
                                "base": base,
                                "color_space": args.color_space,
                                "gc_iters": int(args.gc_iters),
                                "tie_mode": args.tie_mode,
                            },
                        )
                    
                    # Save binary masks (post-refinement) if requested
                    if args.emit_binary_masks:
                        for class_id, binary_mask in fg_masks.items():
                            fname = save_binary_mask_indexed(
                                base=base,
                                color_space=args.color_space,
                                class_id=class_id,
                                binary_mask=binary_mask,
                                out_dir=binary_masks_out_dir,
                                suffix="binary"
                            )
                            binary_written.append(fname)
                    
                    # Save pre-refinement masks if requested
                    if args.emit_pre_refinement_masks:
                        for class_id, binary_mask in fg_masks_pre.items():
                            fname = save_binary_mask_indexed(
                                base=base,
                                color_space=args.color_space,
                                class_id=class_id,
                                binary_mask=binary_mask,
                                out_dir=pre_refinement_masks_out_dir,
                                suffix="pre_refine"
                            )
                            pre_refinement_written.append(fname)
                    
                    # Save superpixel segmentations if requested
                    if args.emit_superpixels:
                        for class_id, sp_seg in superpixel_segs.items():
                            fname = save_superpixel_segmentation(
                                base=base,
                                color_space=args.color_space,
                                class_id=class_id,
                                superpixel_seg=sp_seg,
                                out_dir=superpixels_out_dir,
                                img_rgb=img_rgb  # Pass RGB image for overlay
                            )
                            superpixels_written.append(fname)
                    
                    # Save guided soft masks if requested
                    if args.emit_guided_soft:
                        for class_id, soft_mask in guided_soft_masks.items():
                            fname = save_guided_soft_mask(
                                base=base,
                                color_space=args.color_space,
                                class_id=class_id,
                                soft_mask=soft_mask,
                                out_dir=guided_soft_out_dir
                            )
                            guided_soft_written.append(fname)

                out_path = out_dir / f"{base}_index.png"
                save_indexed_png(pred, str(out_path))

                dt = (perf_counter() - t0) * 1000.0
                times_ms.append(dt)
                processed += 1
                msg = f"[OK] {base} ({dt:.1f} ms) -> {out_path.name}"
                if written:
                    msg += f", models: {len(written)} files"
                if binary_written:
                    msg += f", binary_masks: {len(binary_written)} files"
                if pre_refinement_written:
                    msg += f", pre_refine: {len(pre_refinement_written)} files"
                if superpixels_written:
                    msg += f", superpixels: {len(superpixels_written)} files"
                if guided_soft_written:
                    msg += f", guided_soft: {len(guided_soft_written)} files"
                tqdm.write(msg)

            except FileNotFoundError:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} image file not found, expected at {img_path}")
            except cv.error as e:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} OpenCV GrabCut failed: {e}")
            except Exception as e:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} {e}")

    summary = {
        "mode": "batch",
        "core": "opencv_only",
        "images_dir": str(images_dir),
        "anns_dir": str(anns_dir),
        "output_dir": str(out_dir),
        "processed": processed,
        "skipped": skipped,
        "params": {
            "gc_iters": int(args.gc_iters),
            "tie_mode": args.tie_mode,
            "color_space": args.color_space,
            "enable_majority_vote": bool(args.enable_majority_vote),
            "ensemble_trio": str(args.ensemble_trio),
            "ensemble_trio_parallel": str(args.ensemble_trio_parallel),
            "ensemble_trio_workers": int(args.ensemble_trio_workers),

            "ensemble_label_tie_strategy": str(args.ensemble_label_tie_strategy),
            "parallel": bool(args.parallel),
            "emit_models": bool(args.emit_models) and not bool(args.enable_majority_vote),
            "models_dir": str(models_out_dir) if args.emit_models and not args.enable_majority_vote else None,
            "emit_binary_masks": bool(args.emit_binary_masks) and not bool(args.enable_majority_vote),
            "binary_masks_dir": str(binary_masks_out_dir) if args.emit_binary_masks and not args.enable_majority_vote else None,
            "emit_pre_refinement_masks": bool(args.emit_pre_refinement_masks) and not bool(args.enable_majority_vote),
            "pre_refinement_masks_dir": str(pre_refinement_masks_out_dir) if args.emit_pre_refinement_masks and not args.enable_majority_vote else None,
            "emit_superpixels": bool(args.emit_superpixels) and not bool(args.enable_majority_vote),
            "superpixels_dir": str(superpixels_out_dir) if args.emit_superpixels and not args.enable_majority_vote else None,
            "emit_guided_soft": bool(args.emit_guided_soft) and not bool(args.enable_majority_vote),
            "guided_soft_dir": str(guided_soft_out_dir) if args.emit_guided_soft and not args.enable_majority_vote else None,
            "max_workers": int(args.max_workers if args.max_workers else (os.cpu_count() or 4) if args.parallel else 0),
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()