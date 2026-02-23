# Filename: grabcut.py
# -*- coding: utf-8 -*-

"""GrabCut batch CLI for one vs rest segmentation with optional ensemble over three color spaces.
This version moves majority voting from per-class binary masks to voting over final indexed masks.
Indexed masks now follow app/GT convention: 0 background, foreground labels equal class IDs (2..20 -> 2..20).
Minimal flags: --images_dir, --anns_dir, --output_dir. To enable ensemble, add --enable_majority_vote."""

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
    """
    Algorithm 1: Final Ensemble Majority Voting.
    Fuses three multi-class segmentation results by finding the consensus 
    label for each pixel.
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(f"Shape mismatch in majority_vote_indexed, got {a.shape}, {b.shape}, {c.shape}")
    
    # Ensure all inputs are in the correct unsigned 8-bit integer format for VOC labels
    a = a.astype("uint8", copy=False)
    b = b.astype("uint8", copy=False)
    c = c.astype("uint8", copy=False)
    
    """
    Identify consensus pairs.
    A label wins the 'Majority Vote' if at least two out of three masks agree on it.
    """
    eq_ab = (a == b) # Class assignment matches between Mask A and Mask B
    eq_ac = (a == c) # Class assignment matches between Mask A and Mask C
    eq_bc = (b == c) # Class assignment matches between Mask B and Mask C
    
    # Initialize output array with the first mask's values as a baseline
    out = a.copy()
    
    # If A and B agree, their shared label is the majority winner
    mask_ab = eq_ab
    out[mask_ab] = a[mask_ab]
    
    # If A and C agree (and it wasn't already settled by A and B), A/C is the winner
    mask_ac = eq_ac & (~mask_ab)
    out[mask_ac] = a[mask_ac]
    
    # If B and C agree (and it wasn't settled by A), B/C is the winner
    mask_bc = eq_bc & (~(mask_ab | mask_ac))
    out[mask_bc] = b[mask_bc]
    
    """
    Algorithm 1, Line 28: Handling 'Three-way Ties'.
    If all three masks predict a different label, a majority cannot be found.
    We resolve this using the 'tie_pref' (Tie Preference) parameter.
    """
    mask_tie = ~(mask_ab | mask_ac | mask_bc)
    
    if mask_tie.any():
        # Fallback to the preferred source mask when no consensus exists
        if tie_pref == 0:
            out[mask_tie] = a[mask_tie] # Prefer Mask A
        elif tie_pref == 1:
            out[mask_tie] = b[mask_tie] # Prefer Mask B
        else:
            out[mask_tie] = c[mask_tie] # Prefer Mask C
            
    # Return the final consensus-based multiclass segmentation map
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
                    anns: np.ndarray) -> np.ndarray:
    """
    For each present class c > 1:
      FG seeds = anns == c
      BG seeds = anns == 1 or anns > 1 and not equal to c
    Combine binary masks into a single index map where:
      output 0 = background, output 1..20 = foreground classes, map c -> c - 1.

    Returns the final indexed mask.
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        return np.zeros((H, W), dtype=np.uint8)

    fg_masks: Dict[int, np.ndarray] = {}

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
        y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg)  # type: ignore
        
        """
        Performs post-refinement using Guided Image Filtering
        """
        y = mgc_post_smooth_mask(img_rgb_u8, y)
        fg_masks[c] = y # binary 0 or 1


    """
    Performs multiclass assembly
    """
    final = _combine_fg_masks_to_final(fg_masks, anns)
    return final


def _combine_fg_masks_to_final(fg_masks: Dict[int, np.ndarray],
                               anns: np.ndarray) -> np.ndarray:
    """
    Algorithm 1: Final Label Fusion via Majority Voting.
    This function aggregates multiple binary s-t cuts into a single multiclass 
    segmentation map, maintaining the O(K) linear complexity described in the paper.

    Source: Hu YC, Mageras G, Grossberg M. Multi-class medical image segmentation using one-vs-rest graph cuts and majority voting. J Med Imaging (Bellingham). 2021;8(3):034003. doi:10.1117/1.JMI.8.3.034003
    """
    
    # Algorithm 1, Line 1: Iterate through the set of labels L
    classes = sorted(fg_masks.keys())
    if not classes:
        H, W = anns.shape
        return np.zeros((H, W), dtype=np.uint8)

    H, W = anns.shape
    
    """
    Algorithm 1, Line 22: Majority votes.
    Collect individual binary segmentation results into a 3D volume to evaluate 
    class assignments per pixel.
    """
    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    
    # Algorithm 1, Line 22: Count the number of labels assigned to each pixel (Ta)
    overlap_count = stack.sum(axis=2)
    
    # Check if any pixel has been claimed by more than one binary s-t cut
    any_overlap = (overlap_count > 1).any()

    # Algorithm 1, Line 21: Initialize the final multi-class segmentation map S
    final = np.zeros((H, W), dtype=np.uint8)

    """
    Algorithm 1, Lines 25-27: Case where a unique majority exists (|I_max| == 1).
    If only one binary s-t cut returns 'foreground' for a pixel, that specific 
    label is assigned to the final segmentation map.
    """
    if not any_overlap:
        # Iterate through each class to apply the binary foreground mask to the final result
        for c in classes:
            # Create a boolean mask where the current class was predicted as foreground
            m = fg_masks[c] > 0
            # Assign the class index (zero-indexed) to the final multiclass map
            final[m] = c - 1
        # Return the resulting map if no overlaps need to be resolved via tie-breaking
        return final

    """
    EXTENSION/DEVIATION from Algorithm 1, Line 28:
    The paper suggests using a regional classifier (Random Forest) to 
    provide a probability p(alpha|xi) as a tie-breaker. 
    
    Our architecture extends this logic for Interactive GrabCut by using 
    spatial distance to user scribbles as the confidence metric instead 
    of a secondary classifier.
    """
    overlap_mask = (overlap_count > 1)

    dist_to_scrib: Dict[int, np.ndarray] = {}
    classes_for_dt: List[int] = []
    
    # Calculate spatial confidence (Distance Transform) for each class
    for c in classes:
        # Optimization: Only calculate distances for classes actually involved in a conflict
        if np.any(fg_masks[c] & overlap_mask):
            
            # Extract only the user-drawn seeds for this specific class
            s = (anns == c).astype(np.uint8)
            
            if np.any(s):
                """
                OpenCV's distanceTransform measures distance to the nearest ZERO pixel.
                To find distance to our seeds, we create a map where seeds are 0 
                and background is 1.
                """
                ones = np.ones_like(s, dtype=np.uint8)
                ones[s > 0] = 0 # Set seed locations to 0 (the targets)
                
                # Compute Euclidean distance map from every pixel to the nearest seed
                d = cv.distanceTransform(ones, cv.DIST_L2, 3).astype(np.float32)
            else:
                """
                If a class was predicted by the GMM/GraphCut but has no seeds 
                (e.g., from a previous frame's propagation), we assign a massive 
                penalty distance (1e6) so it loses almost any tie-break.
                """
                d = np.full(s.shape, 1e6, dtype=np.float32)
            
            # Map the distance array to the class ID for lookup during the argmin phase
            dist_to_scrib[c] = d
            # Keep track of which specific classes are competing in the stack
            classes_for_dt.append(c)

    if classes_for_dt:
        """
        Create a 3D volume where each layer is a distance map.
        If a class didn't even claim a pixel in the binary mask (fg_masks[c] == 0), 
        we set its distance to INF (1e9) so it is excluded from the competition.
        """
        INF = 1e9

        """
        If current class c predicted a pixel as foreground (fg_masks[c] > 0), we take the distance to scribble for that class; otherwise, we set it to INF to exclude it from winning the tie-break for that pixel.
        """
        dstack = np.stack(
            [np.where(fg_masks[c] > 0, dist_to_scrib[c], INF) for c in classes_for_dt],
            axis=2
        )
        
        # arg contains the integer index of the "winning" class (the one with the minimum distance)
        arg = np.argmin(dstack, axis=2)

        # First, fill in labels for pixels where only one class was predicted
        for c in classes:
            # Mask for pixels that are foreground for class 'c' AND have no overlap conflict
            m = (fg_masks[c] > 0) & (~overlap_mask)
            final[m] = c - 1

        """
        Now resolve conflicts: For each class that participated in the distance stack, 
        identify pixels where that class had the minimum distance (arg == idx) 
        and an overlap actually existed.
        """
        for idx, c in enumerate(classes_for_dt):
            # idx is the position in the stack; if arg == idx, this class 'c' is the closest
            m = overlap_mask & (arg == idx)
            final[m] = c - 1
    else:
        # Fallback to standard majority assignment if no spatial data is available
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1

    # Algorithm 1, Line 32: Return the final multi-class segmentation S
    return final


# ---------- majority voting ensemble, one vs rest ----------

def run_one_vs_rest_majority_ensemble(img_rgb_u8: np.ndarray,
                                      anns: np.ndarray,
                                      trio: List[str],
                                      trio_parallel: bool = False) -> np.ndarray:
    """
    Majority voting ensemble over generated indexed masks, one per color space.
    For each color space, compute a full indexed mask via run_one_vs_rest, then vote on labels.
    Three-way ties default to the first color space in the trio.
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((H, W), dtype=np.uint8)
    
    if len(trio) != 3:
        raise ValueError("Ensemble trio must have exactly three color spaces")
    
    workers = len(trio)

    def _predict_for_space(cs: str) -> np.ndarray:
        feats = convert_color_space(img_rgb_u8, cs)
        pred = run_one_vs_rest(feats, img_rgb_u8, anns)  # type: ignore
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
    out = majority_vote_indexed(preds[0], preds[1], preds[2], tie_pref=0)
    return out


# ---------- worker for parallel batch ----------

def _process_single_image(ann_path: str,
                          images_dir: str,
                          output_dir: str,
                          color_space: str,
                          enable_majority_vote: bool,
                          ensemble_trio: str,
                          trio_parallel: bool) -> Dict[str, object]:
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
        Runs the majority voting ensemble across the specified trio of color spaces
        """
        pred = run_one_vs_rest_majority_ensemble(
            img_rgb_u8=img_rgb,
            anns=anns,
            trio=trio,
            trio_parallel=bool(trio_parallel)
        )
    else:
        img_feats = convert_color_space(img_rgb, color_space)
        pred = run_one_vs_rest(
            img_feats, img_rgb, anns
        )

    out_path = out_dir_p / f"{base}_index.png"
    save_indexed_png(pred, str(out_path))

    dt = (perf_counter() - t0) * 1000.0
    return {"ok": True, "base": base, "ms": dt, "out": out_path.name}


# ---------- CLI ----------

def parse_args(argv=None):
    ap = argparse.ArgumentParser("GrabCut batch CLI, OpenCV backend, one vs rest")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--anns_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=0, help="0 means all")
    ap.add_argument("--start_one", type=int, default=1, help="1 based index of first file")

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
    ap.add_argument("--parallel", action="store_true", help="Enable parallel processing of images")

    ### Should be default
    ap.add_argument("--max_workers", type=int, default=0, help="Workers for parallel mode, 0 picks os.cpu_count()")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Intra-image trio parallelization: parallelize when not running batch parallel
    trio_parallel_flag = not bool(args.parallel)

    if args.parallel:
        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process_single_image,
                    str(ann_path), str(images_dir), str(out_dir),
                    str(args.color_space),
                    bool(args.enable_majority_vote), str(args.ensemble_trio),
                    bool(trio_parallel_flag) and bool(args.enable_majority_vote)
                ): ann_path for ann_path in ann_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), unit="img", desc="GrabCut[par]"):
                try:
                    res = fut.result()
                    if res.get("ok"):
                        processed += 1
                        times_ms.append(float(res.get("ms", 0.0)))
                        tqdm.write(f"[OK] {res.get('base')} ({res.get('ms'):.1f} ms) -> {res.get('out')}")
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

                if args.enable_majority_vote:
                    trio = [s.strip() for s in (args.ensemble_trio if args.ensemble_trio else "ruderman_lab,oklab,jzczhz").split(",")]
                    if len(trio) != 3:
                        raise ValueError("ensemble_trio must have exactly three comma separated color spaces")
                    pred = run_one_vs_rest_majority_ensemble(
                        img_rgb_u8=img_rgb,
                        anns=anns,
                        trio=trio,
                        trio_parallel=bool(trio_parallel_flag)
                    )
                else:
                    img_feats = convert_color_space(img_rgb, args.color_space)
                    pred = run_one_vs_rest(
                        img_feats, img_rgb, anns
                    )

                out_path = out_dir / f"{base}_index.png"
                save_indexed_png(pred, str(out_path))

                dt = (perf_counter() - t0) * 1000.0
                times_ms.append(dt)
                processed += 1
                tqdm.write(f"[OK] {base} ({dt:.1f} ms) -> {out_path.name}")

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
            "gc_iters": 5,
            "tie_mode": "nearest-scribble",
            "color_space": args.color_space,
            "enable_majority_vote": bool(args.enable_majority_vote),
            "ensemble_trio": str(args.ensemble_trio),
            "parallel": bool(args.parallel),
            "max_workers": int(args.max_workers if args.max_workers else (os.cpu_count() or 4) if args.parallel else 0),
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()