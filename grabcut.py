# Filename: grabcut.py
# -*- coding: utf-8 -*-
"""GrabCut batch CLI for one-vs-rest segmentation with optional ensemble.

This version moves majority voting from per-class binary masks to voting over
final indexed masks. Indexed masks follow app/GT convention: 0 background,
foreground labels equal class IDs (2..20 -> 2..20).

Minimal flags:
    --images_dir, --anns_dir, --output_dir

To enable ensemble:
    --enable_majority_vote
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

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
GRABCUT_ITERATIONS: int = 5
"""Number of GrabCut graph-cut iterations per class."""

TIE_BREAK_INFINITY: float = 1e9
"""Large penalty distance for pixels not claimed by a class during tie-breaking."""

NO_SEED_PENALTY: float = 1e6
"""Penalty distance for classes with no user-drawn seeds."""

def majority_vote_indexed(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tie_pref: int = 0,
) -> np.ndarray:
    """Fuse three multi-class segmentation results via pixel-wise majority voting.

    Implements Algorithm 1: Final Ensemble Majority Voting. For each pixel, the
    label agreed upon by at least two of the three input masks is selected.

    Args:
        a: First indexed segmentation mask, shape (H, W), dtype uint8.
        b: Second indexed segmentation mask, shape (H, W), dtype uint8.
        c: Third indexed segmentation mask, shape (H, W), dtype uint8.
        tie_pref: Index of the preferred mask (0, 1, or 2) used when all three
            masks disagree (three-way tie). Defaults to 0 (prefer mask ``a``).

    Returns:
        Consensus segmentation mask with the same shape as inputs, dtype uint8.

    Raises:
        ValueError: If input mask shapes do not match.
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(
            f"Shape mismatch in majority_vote_indexed: "
            f"{a.shape}, {b.shape}, {c.shape}"
        )

    # Ensure all inputs are uint8 for VOC-style labels.
    a = a.astype("uint8", copy=False)
    b = b.astype("uint8", copy=False)
    c = c.astype("uint8", copy=False)

    # Identify consensus pairs. A label wins if at least two masks agree.
    eq_ab = a == b  # Mask A matches Mask B
    eq_ac = a == c  # Mask A matches Mask C
    eq_bc = b == c  # Mask B matches Mask C

    # Initialize output with first mask values as baseline.
    out = a.copy()

    # If A and B agree, their shared label is the majority winner.
    mask_ab = eq_ab
    out[mask_ab] = a[mask_ab]

    # If A and C agree (not already settled by A-B), A/C wins.
    mask_ac = eq_ac & (~mask_ab)
    out[mask_ac] = a[mask_ac]

    # If B and C agree (not settled by A), B/C wins.
    mask_bc = eq_bc & (~(mask_ab | mask_ac))
    out[mask_bc] = b[mask_bc]

    # Algorithm 1, Line 28: Handle three-way ties.
    # If all three masks predict different labels, fall back to tie_pref.
    mask_tie = ~(mask_ab | mask_ac | mask_bc)

    if mask_tie.any():
        if tie_pref == 0:
            out[mask_tie] = a[mask_tie]
        elif tie_pref == 1:
            out[mask_tie] = b[mask_tie]
        else:
            out[mask_tie] = c[mask_tie]

    return out


# ---------------------------------------------------------------------------
# OpenCV GrabCut, single call
# ---------------------------------------------------------------------------


def opencv_grabcut_once(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
) -> np.ndarray:
    """Execute a single OpenCV GrabCut segmentation pass.

    Performs graph-cut based foreground/background segmentation using user-
    provided seed masks. Follows the GC_INIT_WITH_MASK workflow.

    Source:
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/
        page_tutorial_py_grabcut.html

    Args:
        img_feats_u8: Input image features, shape (H, W, 3), dtype uint8.
            Typically an RGB or color-converted image.
        seeds_bg: Boolean mask indicating definite background pixels.
        seeds_fg: Boolean mask indicating definite foreground pixels.

    Returns:
        Binary segmentation mask, shape (H, W), dtype uint8. Pixels labeled
        1 are foreground, 0 are background.

    Raises:
        TypeError: If ``img_feats_u8`` is not dtype uint8.
        ValueError: If image is not HxWx3 or seed shapes do not match image.
    """
    if img_feats_u8.dtype != np.uint8:
        raise TypeError(
            f"GrabCut requires uint8 features. Got {img_feats_u8.dtype}. "
            "Ensure the color converter scales data to [0, 255] before calling."
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

    # Return empty mask if no foreground seeds provided.
    if not np.any(seeds_fg):
        return np.zeros((h, w), dtype=np.uint8)

    # Initialize annotation mask with OpenCV's expected labels:
    # GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3.
    # Fill with probable background, then apply firm seeds.
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

    # Extract binary mask: foreground = GC_FGD or GC_PR_FGD.
    bin_mask = np.where(
        (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    return bin_mask


# ---------------------------------------------------------------------------
# Multi-class wrapper: one-vs-rest segmentation
# ---------------------------------------------------------------------------


def run_one_vs_rest(
    img_feats_u8: np.ndarray,
    img_rgb_u8: np.ndarray,
    anns: np.ndarray,
) -> np.ndarray:
    """Perform one-vs-rest GrabCut segmentation for all annotated classes.

    For each foreground class c > 1:
        - FG seeds = pixels where ``anns == c``
        - BG seeds = pixels where ``anns == 1`` OR other foreground classes

    Binary masks are combined into a single indexed segmentation map.

    Args:
        img_feats_u8: Color-space converted image features, shape (H, W, 3),
            dtype uint8.
        img_rgb_u8: Original RGB image, shape (H, W, 3), dtype uint8.
            Used for seed refinement.
        anns: Annotation mask with class labels. 0 = unknown, 1 = background,
            2..K = foreground classes.

    Returns:
        Indexed segmentation mask, shape (H, W), dtype uint8. Background is 0,
        foreground classes are mapped to ``class_id - 1``.
    """
    h, w = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((h, w), dtype=np.uint8)

    fg_masks: Dict[int, np.ndarray] = {}

    for c in classes:
        # Select seeds: current class as FG, background + other FG classes as BG.
        seeds_fg = anns == c
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))

        # Refine seeds via geodesic distance and GMM LAB confidence.
        seeds_fg, seeds_bg = _expand_seeds(
            img_rgb_u8,
            seeds_bg=seeds_bg,
            seeds_fg=seeds_fg,
            conf_img=img_feats_u8,
        )

        # Run GrabCut with expanded seeds.
        y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg)

        # Post-refinement using Guided Image Filtering.
        y = _apply_guided_filter(img_rgb_u8, y)
        fg_masks[c] = y  # Binary mask: 0 or 1

    # Assemble final multiclass map.
    return _combine_fg_masks_to_final(fg_masks, anns)


def _combine_fg_masks_to_final(
    fg_masks: Dict[int, np.ndarray],
    anns: np.ndarray,
) -> np.ndarray:
    """Aggregate binary one-vs-rest masks into a single multiclass segmentation map.

    Implements Algorithm 1: Final Label Fusion via Majority Voting. Maintains
    O(K) linear complexity where K is the number of classes.

    When multiple classes claim the same pixel (overlap), ties are resolved using
    spatial distance to user-drawn scribbles as a confidence metric.

    Source:
        Hu YC, Mageras G, Grossberg M. Multi-class medical image segmentation
        using one-vs-rest graph cuts and majority voting. J Med Imaging
        (Bellingham). 2021;8(3):034003. doi:10.1117/1.JMI.8.3.034003

    EXTENSION/DEVIATION from Algorithm 1, Line 28:
        The paper suggests using a regional classifier (Random Forest) to provide
        a probability p(alpha|xi) as a tie-breaker. Our architecture extends this
        logic for Interactive GrabCut by using spatial distance to user scribbles
        as the confidence metric instead of a secondary classifier.

    Args:
        fg_masks: Dictionary mapping class IDs to binary foreground masks.
            Each mask is shape (H, W) with values 0 or 1.
        anns: Original annotation mask used to extract user-drawn seed locations
            for distance-based tie-breaking.

    Returns:
        Indexed segmentation mask, shape (H, W), dtype uint8. Labels are
        zero-indexed (class c -> c - 1).
    """
    # Algorithm 1, Line 1: Iterate through the set of labels L.
    classes = sorted(fg_masks.keys())
    if not classes:
        h, w = anns.shape
        return np.zeros((h, w), dtype=np.uint8)

    h, w = anns.shape

    # Algorithm 1, Line 22: Majority votes. Stack binary masks into 3D volume.
    stack = np.stack([fg_masks[c] for c in classes], axis=2)

    # Count labels assigned per pixel (Ta).
    overlap_count = stack.sum(axis=2)

    # Check if any pixel claimed by more than one class.
    any_overlap = (overlap_count > 1).any()

    # Algorithm 1, Line 21: Initialize final multiclass segmentation map S.
    final = np.zeros((h, w), dtype=np.uint8)

    # Algorithm 1, Lines 25-27: No overlap case - unique majority exists.
    if not any_overlap:
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1
        return final

    # Overlap detected - resolve via distance transform.
    overlap_mask = overlap_count > 1

    dist_to_scrib: Dict[int, np.ndarray] = {}
    classes_for_dt: List[int] = []

    # Calculate spatial confidence (Distance Transform) for conflicting classes.
    for c in classes:
        if not np.any(fg_masks[c] & overlap_mask):
            continue

        # Extract user-drawn seeds for this class.
        s = (anns == c).astype(np.uint8)

        if np.any(s):
            # distanceTransform measures distance to nearest ZERO pixel.
            # Create map where seeds are 0 (targets) and background is 1.
            ones = np.ones_like(s, dtype=np.uint8)
            ones[s > 0] = 0
            d = cv.distanceTransform(ones, cv.DIST_L2, 3).astype(np.float32)
        else:
            # No seeds: assign massive penalty so class loses tie-breaks.
            d = np.full(s.shape, NO_SEED_PENALTY, dtype=np.float32)

        dist_to_scrib[c] = d
        classes_for_dt.append(c)

    if classes_for_dt:
        # Build distance stack. Non-claimed pixels get INF to exclude from argmin.
        dstack = np.stack(
            [
                np.where(fg_masks[c] > 0, dist_to_scrib[c], TIE_BREAK_INFINITY)
                for c in classes_for_dt
            ],
            axis=2,
        )

        # Winning class has minimum distance.
        arg = np.argmin(dstack, axis=2)

        # Fill non-conflicting pixels first.
        for c in classes:
            m = (fg_masks[c] > 0) & (~overlap_mask)
            final[m] = c - 1

        # Resolve conflicts using distance-based winner.
        for idx, c in enumerate(classes_for_dt):
            m = overlap_mask & (arg == idx)
            final[m] = c - 1
    else:
        # Fallback: standard majority assignment.
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1

    # Algorithm 1, Line 32: Return final multiclass segmentation S.
    return final


# ---------------------------------------------------------------------------
# Majority voting ensemble: one-vs-rest with three color spaces
# ---------------------------------------------------------------------------


def run_one_vs_rest_majority_ensemble(
    img_rgb_u8: np.ndarray,
    anns: np.ndarray,
    trio: List[str],
) -> np.ndarray:
    """Perform ensemble segmentation by voting over three color space predictions.

    For each color space in the trio, computes a full indexed mask via
    ``run_one_vs_rest``, then fuses results using pixel-wise majority voting.
    Three-way ties default to the first color space in the trio.

    Processing of the three color space branches is parallelized.

    Args:
        img_rgb_u8: Input RGB image, shape (H, W, 3), dtype uint8.
        anns: Annotation mask with class labels.
        trio: List of exactly three color space names for the ensemble.

    Returns:
        Consensus segmentation mask, shape (H, W), dtype uint8.

    Raises:
        ValueError: If ``trio`` does not contain exactly three color spaces.
    """
    h, w = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])

    if not classes:
        return np.zeros((h, w), dtype=np.uint8)

    if len(trio) != 3:
        raise ValueError("Ensemble trio must have exactly three color spaces.")

    def _predict_for_space(cs: str) -> np.ndarray:
        """Convert to color space and predict."""
        feats = convert_color_space(img_rgb_u8, cs)
        pred = run_one_vs_rest(feats, img_rgb_u8, anns)
        return pred.astype(np.uint8, copy=False)

    # Parallel processing of three color space branches.
    with ThreadPoolExecutor(max_workers=len(trio)) as executor:
        futures = [executor.submit(_predict_for_space, cs) for cs in trio]
        preds = [f.result() for f in futures]

    return majority_vote_indexed(preds[0], preds[1], preds[2], tie_pref=0)


# ---------------------------------------------------------------------------
# Worker for parallel batch processing
# ---------------------------------------------------------------------------


def _process_single_image(
    ann_path: str,
    images_dir: str,
    output_dir: str,
    color_space: str,
) -> Dict[str, object]:
    """Process a single image for batch parallel execution.

    Worker function used by ``ProcessPoolExecutor`` in single color space mode.

    Args:
        ann_path: Absolute path to the annotation file.
        images_dir: Directory containing source images.
        output_dir: Directory for output indexed PNG masks.
        color_space: Color space identifier for feature conversion.

    Returns:
        Dictionary with processing results:
            - ``ok``: True if successful, False otherwise.
            - ``base``: Base filename of the processed image.
            - ``ms``: Processing time in milliseconds (if successful).
            - ``out``: Output filename (if successful).
            - ``reason``: Failure reason (if unsuccessful).
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

    # Resize annotations to match image dimensions if needed.
    if anns.shape[:2] != img_rgb.shape[:2]:
        anns = cv.resize(
            anns.astype(np.int32),
            (img_rgb.shape[1], img_rgb.shape[0]),  # cv.resize expects (width, height)
            interpolation=cv.INTER_NEAREST,
        )

    img_feats = convert_color_space(img_rgb, color_space)
    pred = run_one_vs_rest(img_feats, img_rgb, anns)

    out_path = out_dir_p / f"{base}_index.png"
    save_indexed_png(pred, str(out_path))

    dt = (perf_counter() - t0) * 1000.0
    return {"ok": True, "base": base, "ms": dt, "out": out_path.name}


# ---------------------------------------------------------------------------
# CLI: Argument parsing and entry point
# ---------------------------------------------------------------------------

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
    """Parse command-line arguments for the GrabCut batch CLI.

    Args:
        argv: Command-line arguments. If None, uses ``sys.argv``.

    Returns:
        Parsed argument namespace.
    """
    ap = argparse.ArgumentParser(
        description="GrabCut batch CLI, OpenCV backend, one-vs-rest."
    )

    # Required paths.
    ap.add_argument("--images_dir", type=str, required=True,
                    help="Directory containing source images.")
    ap.add_argument("--anns_dir", type=str, required=True,
                    help="Directory containing annotation masks.")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory for output indexed PNG masks.")

    # Batch control.
    ap.add_argument("--num_images", type=int, default=0,
                    help="Number of images to process. 0 means all.")
    ap.add_argument("--start_one", type=int, default=1,
                    help="1-based index of first file to process.")

    # Single color space mode.
    ap.add_argument(
        "--color_space",
        type=str,
        default="rgb",
        choices=_COLOR_SPACE_CHOICES,
        help="Color space for feature extraction. Default: rgb.",
    )

    # Ensemble mode.
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

    # Parallelization.
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Enable batch parallel processing (single color space mode only).",
    )

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the GrabCut batch segmentation pipeline.

    Processes all annotation files in the specified directory, performing
    one-vs-rest segmentation with optional ensemble voting.

    Args:
        argv: Command-line arguments. If None, uses ``sys.argv``.
    """
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather annotation files.
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

    # Apply slicing based on start index and count.
    if args.start_one is not None and args.start_one > 1:
        ann_files = ann_files[args.start_one - 1:]
    if args.num_images and args.num_images > 0:
        ann_files = ann_files[:args.num_images]

    processed, skipped = 0, 0
    times_ms: List[float] = []

    # Warn if conflicting parallel options.
    if args.parallel and args.enable_majority_vote:
        print(
            "Warning: --parallel is ignored when --enable_majority_vote "
            "is enabled (ensemble mode already parallelizes internally)."
        )

    if args.parallel and not args.enable_majority_vote:
        # Parallel batch processing in single color space mode.
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
        # Sequential processing.
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

                # Resize annotations to match image dimensions.
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

    # Compute effective max_workers for summary.
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