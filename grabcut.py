# Filename: grabcut.py
# -*- coding: utf-8 -*-
"""
Batch CLI for one-vs-rest GrabCut segmentation.
Fuses class-specific binary masks into indexed PNGs.
Supports majority-vote ensembles across color spaces.
"""
from __future__ import annotations

# Standard library imports
import argparse
import json
import os
import pathlib
import sys
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Dict
from typing import List

# Third-party imports
import cv2 as cv
import numpy as np
from tqdm import tqdm

# Local application imports
# NOTE: Extend sys.path to include mgc_core subpackage before local imports.
sys.path.append(str(pathlib.Path(__file__).parent / "mgc_core"))

from color_space import convert_color_space  # type: ignore
from io_utils import base_from_ann_name
from io_utils import find_image
from io_utils import load_anns
from io_utils import load_img
from io_utils import NUM_VOC_CLASSES
from io_utils import save_indexed_png
from mgc_api import _apply_guided_filter
from mgc_api import _expand_seeds

# Number of graph-cut iterations per class
GRABCUT_ITERATIONS: int = 5

# High distance penalty for pixels unclaimed by any class during tie-breaking
TIE_BREAK_INFINITY: float = 1e9

# High distance penalty for classes lacking user seeds
NO_SEED_PENALTY: float = 1e6

def majority_vote_indexed(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tie_pref: int = 0,
) -> np.ndarray:
    """
    Fuses three multi-class masks via pixel-wise majority voting.
    At least two masks must agree for a label to win; otherwise, takes tie_pref.
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(
            f"Shape mismatch in majority_vote_indexed: "
            f"{a.shape}, {b.shape}, {c.shape}"
        )

    # Cast to uint8 for consistent label handling
    a = a.astype("uint8", copy=False)
    b = b.astype("uint8", copy=False)
    c = c.astype("uint8", copy=False)

    # Find where any two masks agree
    eq_ab = a == b
    eq_ac = a == c
    eq_bc = b == c

    # Preference: A=B matches prioritized over A=C, then B=C
    out = a.copy()
    mask_ab = eq_ab
    out[mask_ab] = a[mask_ab]

    mask_ac = eq_ac & (~mask_ab)
    out[mask_ac] = a[mask_ac]

    mask_bc = eq_bc & (~(mask_ab | mask_ac))
    out[mask_bc] = b[mask_bc]

    # Resolve three-way ties where A, B, and C all differ
    mask_tie = ~(mask_ab | mask_ac | mask_bc)

    if mask_tie.any():
        if tie_pref == 0:
            out[mask_tie] = a[mask_tie]
        elif tie_pref == 1:
            out[mask_tie] = b[mask_tie]
        else:
            out[mask_tie] = c[mask_tie]

    return out

def opencv_grabcut_once(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
) -> np.ndarray:
    """
    Runs a single GrabCut pass using binary seeds.
    Initializes the mask with user-defined seeds and iterates to optimize foreground/background.
    """
    if img_feats_u8.dtype != np.uint8:
        raise TypeError(
            f"GrabCut requires uint8 features. Got {img_feats_u8.dtype}."
        )

    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 topology, got shape {img_feats_u8.shape}"
        )

    h, w, _ = img_feats_u8.shape

    if seeds_bg.shape != (h, w) or seeds_fg.shape != (h, w):
        raise ValueError(
            f"Seed masks must match image size. "
            f"Got {seeds_bg.shape} and {seeds_fg.shape}, expected {(h, w)}."
        )

    # Return empty mask if no foreground seeds exist
    if not np.any(seeds_fg):
        return np.zeros((h, w), dtype=np.uint8)

    # Map user seeds to OpenCV GrabCut labels (0:BGD, 1:FGD, 2:PR_BGD, 3:PR_FGD)
    mask = np.full((h, w), cv.GC_PR_BGD, dtype=np.uint8)
    mask[seeds_bg] = cv.GC_BGD
    mask[seeds_fg] = cv.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv.grabCut(
        img_feats_u8,
        mask,
        None,
        bgd_model,
        fgd_model,
        GRABCUT_ITERATIONS,
        cv.GC_INIT_WITH_MASK,
    )

    # Segment as foreground if pixel is definite or probable foreground
    bin_mask = np.where(
        (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    return bin_mask

def run_one_vs_rest(
    img_feats_u8: np.ndarray,
    img_rgb_u8: np.ndarray,
    anns: np.ndarray,
) -> np.ndarray:
    """
    Orchestrates one-vs-rest segmentation for all active classes.
    For each class, expands user seeds and runs GrabCut independently.
    Fuses result binary masks into a single indexed map.
    """
    h, w = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((h, w), dtype=np.uint8)

    fg_masks: Dict[int, np.ndarray] = {}

    for c in classes:
        # Define current class as FG, all others (including background) as BG
        seeds_fg = anns == c
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))

        # Propagate user seeds using geodesic distance and color confidence
        seeds_fg, seeds_bg = _expand_seeds(
            img_rgb_u8,
            seeds_bg=seeds_bg,
            seeds_fg=seeds_fg,
            conf_img=img_feats_u8,
        )

        # Compute binary foreground mask for current class
        y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg)

        # Smooth boundaries with guided image filtering
        y = _apply_guided_filter(img_rgb_u8, y)
        fg_masks[c] = y  # 0: background, 1: class-specific foreground

    return _combine_fg_masks_to_final(fg_masks, anns)


def _combine_fg_masks_to_final(
    fg_masks: Dict[int, np.ndarray],
    anns: np.ndarray,
) -> np.ndarray:
    """
    Fuses class-specific binary masks into a unified multi-class map. 
    If pixels are claimed by multiple classes, resolve via spatial distance to user seeds.
    """
    classes = sorted(fg_masks.keys())
    if not classes:
        h, w = anns.shape
        return np.zeros((h, w), dtype=np.uint8)

    h, w = anns.shape

    # Determine pixel-wise label assignment counts
    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    overlap_count = stack.sum(axis=2)
    any_overlap = (overlap_count > 1).any()

    final = np.zeros((h, w), dtype=np.uint8)

    # Simplified case: each pixel belongs to at most one class
    if not any_overlap:
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1
        return final

    # Complex case: resolve class overlaps via spatial proximity to user scribbles
    overlap_mask = overlap_count > 1
    dist_to_scrib: Dict[int, np.ndarray] = {}
    classes_for_dt: List[int] = []

    for c in classes:
        if not np.any(fg_masks[c] & overlap_mask):
            continue

        s = (anns == c).astype(np.uint8)

        if np.any(s):
            # Distance transform to nearest zero (seed location)
            ones = np.ones_like(s, dtype=np.uint8)
            ones[s > 0] = 0
            d = cv.distanceTransform(ones, cv.DIST_L2, 3).astype(np.float32)
        else:
            # Penalize classes without seeds
            d = np.full(s.shape, NO_SEED_PENALTY, dtype=np.float32)

        dist_to_scrib[c] = d
        classes_for_dt.append(c)

    if classes_for_dt:
        # Select winning class based on minimum distance to seeds
        dstack = np.stack(
            [
                np.where(fg_masks[c] > 0, dist_to_scrib[c], TIE_BREAK_INFINITY)
                for c in classes_for_dt
            ],
            axis=2,
        )

        arg = np.argmin(dstack, axis=2)

        # First, fill pixels without overlaps
        for c in classes:
            m = (fg_masks[c] > 0) & (~overlap_mask)
            final[m] = c - 1

        # Then, fill pixels with overlaps using distance winners
        for idx, c in enumerate(classes_for_dt):
            m = overlap_mask & (arg == idx)
            final[m] = c - 1
    else:
        # Fallback to simple majority if distance resolution fails
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1

    return final

def run_one_vs_rest_majority_ensemble(
    img_rgb_u8: np.ndarray,
    anns: np.ndarray,
    trio: List[str],
) -> np.ndarray:
    """
    Computes a consensus segmentation by voting over three color space predictions.
    Parallelizes the computation for each color space in the trio.
    """
    h, w = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((h, w), dtype=np.uint8)

    if len(trio) != 3:
        raise ValueError("Ensemble trio must have exactly three color spaces.")

    def _predict_for_space(cs: str) -> np.ndarray:
        # Convert image to target color space and predict multi-class map
        feats = convert_color_space(img_rgb_u8, cs)
        pred = run_one_vs_rest(feats, img_rgb_u8, anns)
        return pred.astype(np.uint8, copy=False)

    # Launch parallel threads for each color space branch
    with ThreadPoolExecutor(max_workers=len(trio)) as executor:
        futures = [executor.submit(_predict_for_space, cs) for cs in trio]
        preds = [f.result() for f in futures]

    return majority_vote_indexed(preds[0], preds[1], preds[2], tie_pref=0)

def _process_single_image(
    ann_path: str,
    images_dir: str,
    output_dir: str,
    color_space: str,
) -> Dict[str, object]:
    """
    Worker function to process one image in single color space mode.
    Loads data, runs segmentation, and saves the indexed PNG result.
    """
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

    # Align annotation resolution with source image
    if anns.shape[:2] != img_rgb.shape[:2]:
        anns = cv.resize(
            anns.astype(np.int32),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv.INTER_NEAREST,
        )

    img_feats = convert_color_space(img_rgb, color_space)
    pred = run_one_vs_rest(img_feats, img_rgb, anns)

    out_path = out_dir_p / f"{base}_index.png"
    save_indexed_png(pred, str(out_path))

    dt = (perf_counter() - t0) * 1000.0
    return {"ok": True, "base": base, "ms": dt, "out": out_path.name}

# Supported color spaces for single-space segmentation.
_COLOR_SPACE_CHOICES = [
    "rgb",
    "hsv_conic",
    "cielab",
    "c02_scd",
    "c16_scd",
    "oklab",
    "oklch",
    "jzazbz",
    "jzczhz",
    "ictcp_pq",
    "xyz",
    "ycbcr_bt709",
    "srgb_linear",
    "ruderman_lab",
    "opponent",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments for batch segmentation.
    Defines input/output paths, batch controls, color space settings, and parallelization.
    """
    ap = argparse.ArgumentParser(
        description="GrabCut batch CLI, OpenCV backend, one-vs-rest."
    )

    # I/O paths
    ap.add_argument("--images_dir", type=str, required=True,
                    help="Directory containing source images.")
    ap.add_argument("--anns_dir", type=str, required=True,
                    help="Directory containing annotation masks.")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory for output indexed PNG masks.")

    # Slicing and batch control
    ap.add_argument("--num_images", type=int, default=0,
                    help="Number of images to process. 0 means all.")
    ap.add_argument("--start_one", type=int, default=1,
                    help="1-based index of first file to process.")

    # Color space and ensemble settings
    ap.add_argument(
        "--color_space",
        type=str,
        default="rgb",
        choices=_COLOR_SPACE_CHOICES,
        help="Color space for feature extraction. Default: rgb.",
    )

    ap.add_argument(
        "--enable_majority_vote",
        action="store_true",
        help="Enable ensemble over a trio of color spaces (parallelized).",
    )
    ap.add_argument(
        "--ensemble_trio",
        type=str,
        default="ruderman_lab,jzazbz,oklch",
        help="Comma-separated trio for majority voting.",
    )

    # Execution control
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Enable batch parallel processing (single color space mode only).",
    )

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Executes the batch segmentation pipeline.
    Discovers annotation files and processes them either sequentially or in parallel.
    Outputs a summary of processed images and average timing.
    """
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid annotation files from source directory
    valid_suffixes = (".npy", ".png", ".bmp", ".tif", ".tiff")
    ann_files = sorted([
        p for p in anns_dir.iterdir()
        if p.suffix.lower() in valid_suffixes
    ])

    if not ann_files:
        print(json.dumps({
            "error": "no annotations found",
            "anns_dir": str(anns_dir),
        }))
        return

    # Filter files by start index and count
    if args.start_one is not None and args.start_one > 1:
        ann_files = ann_files[args.start_one - 1:]
    if args.num_images and args.num_images > 0:
        ann_files = ann_files[:args.num_images]

    processed, skipped = 0, 0
    times_ms: List[float] = []

    if args.parallel and args.enable_majority_vote:
        print(
            "Warning: --parallel is ignored when --enable_majority_vote "
            "is enabled (ensemble mode already parallelizes internally)."
        )

    if args.parallel and not args.enable_majority_vote:
        # Execute batch processing across multiple processes
        max_workers = os.cpu_count() or 4
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_image,
                    str(ann_path),
                    str(images_dir),
                    str(out_dir),
                    str(args.color_space),
                ): ann_path
                for ann_path in ann_files
            }

            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                unit="img",
                desc="GrabCut[par]",
            )
            for fut in progress:
                try:
                    res = fut.result()
                    if res.get("ok"):
                        processed += 1
                        times_ms.append(float(res.get("ms", 0.0)))
                        tqdm.write(
                            f"[OK] {res.get('base')} "
                            f"({res.get('ms'):.1f} ms) -> {res.get('out')}"
                        )
                    else:
                        skipped += 1
                        tqdm.write(
                            f"[SKIP] {Path(futures[fut]).name} {res.get('reason')}"
                        )
                except Exception as exc:
                    skipped += 1
                    ann_path = futures[fut]
                    tqdm.write(f"[SKIP] {ann_path.name} {exc}")
    else:
        # Sequential execution loop
        desc = "GrabCut[ensemble]" if args.enable_majority_vote else "GrabCut"
        progress = tqdm(ann_files, unit="img", desc=desc)

        for ann_path in progress:
            base = base_from_ann_name(ann_path.stem)
            img_path = find_image(base, images_dir)

            if img_path is None:
                tqdm.write(f"[SKIP] {ann_path.name} image not found")
                skipped += 1
                continue

            try:
                t0 = perf_counter()
                img_rgb = load_img(img_path)
                anns_data = load_anns(ann_path)

                # Resize user scribbles to match image resolution
                if anns_data.shape[:2] != img_rgb.shape[:2]:
                    anns_data = cv.resize(
                        anns_data.astype(np.int32),
                        (img_rgb.shape[1], img_rgb.shape[0]),
                        interpolation=cv.INTER_NEAREST,
                    )

                if args.enable_majority_vote:
                    trio_str = args.ensemble_trio or "ruderman_lab,oklab,jzczhz"
                    trio = [s.strip() for s in trio_str.split(",")]
                    if len(trio) != 3:
                        raise ValueError(
                            "ensemble_trio must have exactly three "
                            "comma-separated color spaces."
                        )
                    pred = run_one_vs_rest_majority_ensemble(
                        img_rgb_u8=img_rgb,
                        anns=anns_data,
                        trio=trio,
                    )
                else:
                    img_feats = convert_color_space(img_rgb, args.color_space)
                    pred = run_one_vs_rest(img_feats, img_rgb, anns_data)

                out_path = out_dir / f"{base}_index.png"
                save_indexed_png(pred, str(out_path))

                dt = (perf_counter() - t0) * 1000.0
                times_ms.append(dt)
                processed += 1
                tqdm.write(f"[OK] {base} ({dt:.1f} ms) -> {out_path.name}")

            except FileNotFoundError:
                skipped += 1
                tqdm.write(
                    f"[SKIP] {ann_path.name} "
                    f"image file not found, expected at {img_path}"
                )
            except cv.error as exc:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} OpenCV GrabCut failed: {exc}")
            except Exception as exc:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} {exc}")

    # Determine execution concurrency for summary
    if args.enable_majority_vote:
        effective_workers = 3
    elif args.parallel:
        effective_workers = os.cpu_count() or 4
    else:
        effective_workers = 1

    summary = {
        "mode": "batch",
        "core": "opencv_only",
        "images_dir": str(images_dir),
        "anns_dir": str(anns_dir),
        "output_dir": str(out_dir),
        "processed": processed,
        "skipped": skipped,
        "params": {
            "gc_iters": GRABCUT_ITERATIONS,
            "tie_mode": "nearest-scribble",
            "color_space": args.color_space,
            "enable_majority_vote": bool(args.enable_majority_vote),
            "ensemble_trio": str(args.ensemble_trio),
            "parallel": bool(args.parallel),
            "max_workers": effective_workers,
        },
        "timing_ms_avg": float(np.mean(times_ms)) if times_ms else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()