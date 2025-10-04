# Filename: grabcut.py
# -*- coding: utf-8 -*-

"""
GrabCut batch CLI, OpenCV backend, one-vs-rest wrapper.

Update: adds --color_space to run GrabCut on alternative input spaces.
Supported: rgb, hsv_conic, cielab, c02_scd, c16_scd,
           oklab, oklch, jzazbz, jzczhz, ictcp_pq, xyz, ycbcr_bt709, srgb_linear.

Labeling scheme project wide:
  0 = unlabeled, 1 = background, >1 = foreground classes.

For each foreground class c > 1:
  FG seeds = anns == c
  BG seeds = anns == 1 union anns in other foreground classes

Output mapping when saving:
  background -> 0, class c > 1 -> c - 1, which matches PASCAL VOC indices 0..20 when using the VOC palette.

2025-09-30 Ensemble change:
- Removed fused cut ensemble path and helpers, no PyMaxflow required.
- Added majority voting ensemble over a trio of color spaces, default trio is jzazbz, jzczhz, rgb.
- In this ensemble, we run GrabCut for each color space per class to get three binary masks, then take majority vote per pixel,
  then perform tie resolution between classes as usual.
- Ensemble is enabled with --enable_majority_vote, optional --ensemble_trio sets the trio.

2025-09-30 Intra image parallelization change:
- Added intra image parallelization for the trio color spaces when using the majority voting ensemble.
- New flags:
    --ensemble_trio_parallel, choices auto, on, off. Default auto.
      auto, when args.parallel is False, parallelize the three color spaces for a single image, when args.parallel is True, do them sequentially to avoid oversubscription.
    --ensemble_trio_workers, integer, default 0 which means use len(trio) workers.
- Implementation uses ThreadPoolExecutor to avoid nested process spawning when batch mode is also using processes.
"""

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

from ao import ao_refine_seeds

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
        a = np.load(p, mmap_mode="r")
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
from color_space import convert_color_space, get_color_converter  # type: ignore


# ---------- OpenCV GrabCut, single call ----------

def opencv_grabcut_once(img_feats_u8: np.ndarray,
                        seeds_bg: np.ndarray,
                        seeds_fg: np.ndarray,
                        iters: int = 2,
                        return_models: bool = False,
                        return_mask_states: bool = False
                        ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run cv2.grabCut once with firm seeds and return a binary mask, 1 FG, 0 BG.
    Works on any 3 channel 8 bit image of per pixel features.

    New options:
      return_models, when True, returns (bin_mask, bgdModel, fgdModel)
      return_mask_states, when True with return_models, returns (bin_mask, bgdModel, fgdModel, raw_mask_states)

    Notes:
      - bgdModel, fgdModel are OpenCV's 1x65 buffers, and the raw GrabCut mask uses labels {0=BGD,1=FGD,2=PR_BGD,3=PR_FGD}.
    """
    if img_feats_u8.dtype != np.uint8:
        img_feats_u8 = np.clip(img_feats_u8, 0, 255).astype(np.uint8)
    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got shape {img_feats_u8.shape}")

    H, W, _ = img_feats_u8.shape

    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError(
            f"Seed masks must match image size, got {seeds_bg.shape} and {seeds_fg.shape}, expected {(H, W)}"
        )

    if not np.any(seeds_fg):
        empty = np.zeros((H, W), dtype=np.uint8)
        if return_models:
            if return_mask_states:
                return empty, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), empty.copy()
            else:
                return empty, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        return empty

    mask = np.full((H, W), cv.GC_PR_BGD, dtype=np.uint8)
    mask[seeds_bg] = cv.GC_BGD
    mask[seeds_fg] = cv.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv.grabCut(img_feats_u8, mask, None, bgdModel, fgdModel, int(iters), cv.GC_INIT_WITH_MASK)
    except cv.error as e:
        raise RuntimeError(f"OpenCV GrabCut failed: {e}") from e

    bin_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)

    if return_models:
        if return_mask_states:
            return bin_mask, bgdModel.copy(), fgdModel.copy(), mask.copy()
        return bin_mask, bgdModel.copy(), fgdModel.copy()
    return bin_mask


# ---------- multi class wrapper, one vs rest ----------

def run_one_vs_rest(img_feats_u8: np.ndarray,
                    anns: np.ndarray,
                    gc_iters: int = 5,
                    tie_mode: str = "nearest-scribble",
                    collect_models: bool = False
                    ) -> np.ndarray | Tuple[np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
    """
    For each present class c > 1:
      FG seeds = anns == c
      BG seeds = anns == 1 or anns > 1 and not equal to c
    Combine binary masks into a single VOC index map where:
      output 0 = background, output 1..20 = foreground classes, map c -> c - 1.

    New option:
      collect_models=True returns (final_mask, models_by_class) where models_by_class[c] = {'bgdModel': ..., 'fgdModel': ...}
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        empty = np.zeros((H, W), dtype=np.uint8)
        if collect_models:
            return empty, {}
        return empty

    fg_masks: Dict[int, np.ndarray] = {}
    models_by_class: Dict[int, Dict[str, np.ndarray]] = {}

    for c in classes:
        seeds_fg = (anns == c)
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))
        seeds_fg, seeds_bg = ao_refine_seeds(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg)

        if collect_models:
            y, bgm, fgm = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters, return_models=True)  # type: ignore
            models_by_class[c] = {"bgdModel": bgm, "fgdModel": fgm}
        else:
            y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters)  # type: ignore
        fg_masks[c] = y  # binary 0 or 1

    final = _combine_fg_masks_to_final(fg_masks, anns, tie_mode)
    return (final, models_by_class) if collect_models else final


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
                                      trio_workers: int = 0) -> np.ndarray:
    """
    Majority voting over a trio of color spaces on the binary masks, done before class label assignment.

    For each present class c > 1:
      1. Build FG and BG seeds from anns.
      2. Convert the RGB image to each color space in the trio.
      3. Run GrabCut per color space to get a binary mask.
      4. Take per pixel majority vote over the three masks, threshold sum >= 2 to 1 else 0.

    When trio_parallel is True, the three color space runs are executed in parallel threads per class.
    trio_workers, if 0, defaults to len(trio).
    After per class majority masks are obtained, combine across classes with tie handling.
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        return np.zeros((H, W), dtype=np.uint8)

    if len(trio) != 3:
        raise ValueError("Ensemble trio must have exactly three color spaces")

    workers = trio_workers if trio_workers and trio_workers > 0 else len(trio)

    fg_masks: Dict[int, np.ndarray] = {}

    for c in classes:
        seeds_fg = (anns == c)
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))

        if trio_parallel:
            def _run(cs: str) -> np.ndarray:
                feats_cs = convert_color_space(img_rgb_u8, cs)
                sfg, sbg = ao_refine_seeds(feats_cs, seeds_bg=seeds_bg, seeds_fg=seeds_fg)
                y_bin = opencv_grabcut_once(feats_cs, seeds_bg=sbg, seeds_fg=sfg, iters=gc_iters)
                return y_bin.astype(np.uint8)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_run, cs) for cs in trio]
                votes = [f.result() for f in futures]
        else:
            votes = []
            for cs in trio:
                feats_cs = convert_color_space(img_rgb_u8, cs)
                sfg, sbg = ao_refine_seeds(feats_cs, seeds_bg=seeds_bg, seeds_fg=seeds_fg)
                y_bin = opencv_grabcut_once(feats_cs, seeds_bg=sbg, seeds_fg=sfg, iters=gc_iters)
                votes.append(y_bin.astype(np.uint8))

        stack = np.stack(votes, axis=2)
        y_majority = (stack.sum(axis=2) >= 2).astype(np.uint8)
        fg_masks[c] = y_majority

    final = _combine_fg_masks_to_final(fg_masks, anns, tie_mode)
    return final


# ---------- model I O helpers ----------

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


# ---------- worker for parallel batch ----------

def _process_single_image(ann_path: str,
                          images_dir: str,
                          output_dir: str,
                          color_space: str,
                          gc_iters: int,
                          tie_mode: str,
                          emit_models: bool,
                          models_dir: Optional[str],
                          enable_majority_vote: bool,
                          ensemble_trio: str,
                          trio_parallel: bool,
                          trio_workers: int) -> Dict[str, object]:
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
                         (img_rgb.shape[1], img_rgb.shape[0]),
                         interpolation=cv.INTER_NEAREST)

    # run either majority ensemble or single space
    if enable_majority_vote:
        trio = [s.strip() for s in (ensemble_trio if ensemble_trio else "jzazbz,jzczhz,rgb").split(",")]
        if len(trio) != 3:
            return {"ok": False, "base": base, "reason": "ensemble_trio must have exactly 3 entries"}
        pred = run_one_vs_rest_majority_ensemble(
            img_rgb_u8=img_rgb,
            anns=anns,
            trio=trio,
            gc_iters=int(gc_iters),
            tie_mode=tie_mode,
            trio_parallel=bool(trio_parallel),
            trio_workers=int(trio_workers) if trio_workers else 0
        )
        written = []  # no model export in ensemble path
    else:
        img_feats = convert_color_space(img_rgb, color_space)
        if emit_models:
            pred, models_by_class = run_one_vs_rest(img_feats, anns, gc_iters=int(gc_iters), tie_mode=tie_mode, collect_models=True)  # type: ignore
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
            pred = run_one_vs_rest(img_feats, anns, gc_iters=int(gc_iters), tie_mode=tie_mode)  # type: ignore
            written = []

    out_path = out_dir_p / f"{base}_index.png"
    save_indexed_png(pred, str(out_path))

    dt = (perf_counter() - t0) * 1000.0
    return {"ok": True, "base": base, "ms": dt, "out": out_path.name, "models_written": written}


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
                        "ictcp_pq", "xyz", "ycbcr_bt709", "srgb_linear"
                    ],
                    help="Input feature color space for the single space baseline path. Default is rgb.")

    # majority voting ensemble controls
    ap.add_argument("--enable_majority_vote", action="store_true",
                    help="Enable majority voting ensemble across a trio of color spaces for binary masks prior to class assignment.")
    ap.add_argument("--ensemble_trio", type=str, default="jzazbz,jzczhz,rgb",
                    help="Comma separated trio for majority voting, default jzazbz,jzczhz,rgb.")
    ap.add_argument("--ensemble_trio_parallel", type=str, default="auto", choices=["auto", "on", "off"],
                    help="Intra image trio parallelization. auto, parallelize trio when not running batch parallel, off in batch. on, always parallelize. off, never parallelize.")
    ap.add_argument("--ensemble_trio_workers", type=int, default=0,
                    help="Workers for intra image trio parallelization with threads, 0 means len(trio)")

    ap.add_argument("--emit_models", action="store_true",
                    help="When set, save per class bgdModel and fgdModel NPZ files for the single space path.")
    ap.add_argument("--models_dir", type=str, default="",
                    help="Optional output directory for NPZ model files, defaults to output_dir slash models")

    ap.add_argument("--parallel", action="store_true", help="Enable parallel processing of images")
    ap.add_argument("--max_workers", type=int, default=0, help="Workers for parallel mode, 0 picks os.cpu_count()")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    anns_dir = Path(args.anns_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_out_dir = Path(args.models_dir) if args.models_dir else (out_dir / "models")
    if args.emit_models and not args.enable_majority_vote:
        models_out_dir.mkdir(parents=True, exist_ok=True)

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
                    bool(args.emit_models), str(models_out_dir) if args.emit_models and not args.enable_majority_vote else "",
                    bool(args.enable_majority_vote), str(args.ensemble_trio),
                    bool(trio_parallel_flag) and bool(args.enable_majority_vote),  # only matters for ensemble path
                    int(args.ensemble_trio_workers)
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
                if args.enable_majority_vote:
                    trio = [s.strip() for s in (args.ensemble_trio if args.ensemble_trio else "jzazbz,jzczhz,rgb").split(",")]
                    if len(trio) != 3:
                        raise ValueError("ensemble_trio must have exactly three comma separated color spaces")
                    pred = run_one_vs_rest_majority_ensemble(
                        img_rgb_u8=img_rgb,
                        anns=anns,
                        trio=trio,
                        gc_iters=int(args.gc_iters),
                        tie_mode=args.tie_mode,
                        trio_parallel=bool(trio_parallel_flag),
                        trio_workers=int(args.ensemble_trio_workers)
                    )
                else:
                    img_feats = convert_color_space(img_rgb, args.color_space)
                    if args.emit_models:
                        pred, models_by_class = run_one_vs_rest(img_feats, anns, gc_iters=int(args.gc_iters), tie_mode=args.tie_mode, collect_models=True)  # type: ignore
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
                    else:
                        pred = run_one_vs_rest(img_feats, anns, gc_iters=int(args.gc_iters), tie_mode=args.tie_mode)  # type: ignore

                out_path = out_dir / f"{base}_index.png"
                save_indexed_png(pred, str(out_path))

                dt = (perf_counter() - t0) * 1000.0
                times_ms.append(dt)
                processed += 1
                msg = f"[OK] {base} ({dt:.1f} ms) -> {out_path.name}"
                if written:
                    msg += f", models: {len(written)} files"
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
            "parallel": bool(args.parallel),
            "emit_models": bool(args.emit_models) and not bool(args.enable_majority_vote),
            "models_dir": str(models_out_dir) if args.emit_models and not args.enable_majority_vote else None,
            "max_workers": int(args.max_workers if args.max_workers else (os.cpu_count() or 4) if args.parallel else 0),
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
