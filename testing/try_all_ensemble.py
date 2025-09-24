#!/usr/bin/env python3
"""
Try all three way ensembles across gc_masks_* folders, with GT in NPY, NPZ, MAT, or indexed PNG.

Adds:
- Bootstrap CIs for dataset-level ensemble mIoU and ensemble gain vs best single.
- Optional classwise bootstrap CIs written to a JSON sidecar.
- tqdm progress bar over combos.

Assumptions
1) Project root contains gc_masks_<base>/ folders with predicted indexed masks, labels 0..20.
2) Ground truth folder contains masks with labels 0..20, ignore label 255.
3) Filenames may differ by suffixes like "_index", "_pred", "_prediction", "_mask", this script normalizes stems.

Outputs
- Prints a ranked top table by dataset level ensemble mIoU.
- Writes CSV if you pass --out_csv or --out (now robust if given a directory).
- Optional JSON per-class CI file via --per_class_json.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
import itertools
import numpy as np
import sys
import csv
import json
import os
import concurrent.futures as cf

# tqdm is optional; progress degrades gracefully if unavailable
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import scipy.io as sio
except Exception:
    sio = None

N_CLASSES = 21
DEFAULT_PRED_EXTS = [".png", ".bmp", ".tif", ".tiff", ".mat", ".npy", ".npz"]
DEFAULT_GT_EXTS   = [".npy", ".npz", ".mat", ".png", ".bmp", ".tif", ".tiff"]

DEFAULT_STRIP_SUFFIXES = [
    "_index",
    "_pred",
    "_prediction",
    "_mask",
    "_masks",
    "_label",
    "_labels",
]


# ---------- parallel worker plumbing ----------
_WORKER_CTX = None  # set once per process

def _init_worker(ctx):
    """Initializer runs once per worker process; keeps large structures in globals."""
    global _WORKER_CTX
    _WORKER_CTX = ctx

def _run_combo_task(combo_idx: tuple[int, int, int]):
    """Worker entrypoint for one trio."""
    a, b, c = combo_idx
    ctx = _WORKER_CTX
    labels = ctx["labels"]
    idx_gt = ctx["idx_gt"]
    pred_indices = ctx["pred_indices"]
    args = ctx["args"]

    lab1, lab2, lab3 = labels[a], labels[b], labels[c]
    idx1, idx2, idx3 = pred_indices[a], pred_indices[b], pred_indices[c]

    # Per-trio RNG seed so bootstrap runs are reproducible but independent
    rng = None
    if args["bootstrap_reps"] > 0:
        seed = (args["bootstrap_seed"]
                + a * 1_000_003
                + b * 1_000_033
                + c * 1_000_211) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

    metrics = evaluate_combo(
        idx1, idx2, idx3, idx_gt,
        subset=args["subset"],
        gt_mat_key=args["gt_mat_key"],
        pred_mat_key=args["pred_mat_key"],
        bootstrap_reps=args["bootstrap_reps"],
        ci=args["ci"],
        rng=rng,
        collect_classwise=args["collect_classwise"],
        progress_bar=None,   # child processes don't emit per-image bars
    )
    return (lab1, lab2, lab3, metrics)

# ---------- I/O helpers ----------

def parse_exts(s: str, fallback: List[str]) -> List[str]:
    if not s:
        return list(fallback)
    exts = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        exts.append(part.lower())
    return exts or list(fallback)

def normalize_stem(stem: str, strip_suffixes: Iterable[str]) -> str:
    s = stem
    changed = True
    while changed:
        changed = False
        for suf in strip_suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s

def read_indexed_png(p: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required to read PNG files")
    img = Image.open(p)
    if img.mode not in ("P", "L"):
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D mask in {p}, got shape {arr.shape}")
        return arr.astype(np.uint8)
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask in {p}, got shape {arr.shape}")
    return arr

def load_mat_mask(filepath: Path, key: str = "mtx") -> np.ndarray:
    if sio is None:
        raise RuntimeError("scipy is required to read .mat files")
    mat = sio.loadmat(str(filepath))
    if key not in mat:
        if "arr" in mat:
            arr = mat["arr"]
        elif "data" in mat:
            arr = mat["data"]
        else:
            raise KeyError(f"Key '{key}' not found in {filepath}")
    else:
        arr = mat[key]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask in {filepath}, got shape {arr.shape}")
    return arr.astype(np.uint8)

def load_npy_mask(filepath: Path) -> np.ndarray:
    arr = np.load(str(filepath), allow_pickle=False)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask in {filepath}, got shape {arr.shape}")
    return arr.astype(np.uint8)

def load_npz_mask(filepath: Path) -> np.ndarray:
    with np.load(str(filepath)) as z:
        for k in ("mtx", "mask", "arr_0"):
            if k in z:
                arr = z[k]
                break
        else:
            keys = list(z.keys())
            if not keys:
                raise ValueError(f"No arrays in {filepath}")
            arr = z[keys[0]]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask in {filepath}, got shape {arr.shape}")
    return arr.astype(np.uint8)

def load_mask_auto(p: Path, mat_key: str = "mtx") -> np.ndarray:
    ext = p.suffix.lower()
    if ext in [".png", ".bmp", ".tif", ".tiff"]:
        return read_indexed_png(p)
    if ext == ".mat":
        return load_mat_mask(p, key=mat_key)
    if ext == ".npy":
        return load_npy_mask(p)
    if ext == ".npz":
        return load_npz_mask(p)
    raise ValueError(f"Unsupported mask format for '{p}', use png, bmp, tif, tiff, npy, npz, mat")

def resize_mask_if_needed(src: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if src.shape == target_shape:
        return src
    sh, sw = src.shape
    th, tw = target_shape
    if th % sh == 0 and tw % sw == 0:
        ry = th // sh
        rx = tw // sw
        return np.repeat(np.repeat(src, ry, axis=0), rx, axis=1)
    if Image is None:
        raise RuntimeError("Pillow required to resize non integer scale masks")
    im = Image.fromarray(src, mode="L")
    im = im.resize((tw, th), resample=Image.NEAREST)
    return np.array(im, dtype=np.uint8)

def build_index_recursive(d: Path, strip_suffixes: Iterable[str], exts: List[str]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    files = sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts],
                   key=lambda q: (q.suffix.lower() != ".png", q.as_posix()))
    for p in files:
        ns = normalize_stem(p.stem, strip_suffixes)
        if ns not in idx:
            idx[ns] = p
    return idx

def intersection_keys(*dicts: Dict[str, Path]) -> List[str]:
    if not dicts:
        return []
    keys = set(dicts[0].keys())
    for d in dicts[1:]:
        keys &= set(d.keys())
    return sorted(keys)

# ---------- metrics ----------

def compute_confusion(pred: np.ndarray, gt: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    mask = (gt < 255)
    x = pred[mask].astype(np.int64)
    y = gt[mask].astype(np.int64)
    x = np.clip(x, 0, n_classes - 1)
    y = np.clip(y, 0, n_classes - 1)
    idx = y * n_classes + x
    hist = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return hist

def iou_from_confusion(hist: np.ndarray):
    tp = np.diag(hist).astype(np.float64)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, 0.0)
    miou = float(iou.mean())
    return iou, miou

def tp_union_from_confusion(hist: np.ndarray):
    tp = np.diag(hist).astype(np.float64)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    union = tp + fp + fn
    return tp, union

def majority_vote_first_wins(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    assert p1.shape == p2.shape == p3.shape
    s0, s1, s2 = p1, p2, p3
    out = np.zeros_like(s0, dtype=np.uint8)

    all_eq = (s0 == s1) & (s1 == s2)
    out[all_eq] = s0[all_eq]

    mask = ~all_eq
    two_01 = (s0 == s1) & mask
    out[two_01] = s0[two_01]
    mask = mask & (~two_01)

    two_02 = (s0 == s2) & mask
    out[two_02] = s0[two_02]
    mask = mask & (~two_02)

    two_12 = (s1 == s2) & mask
    out[two_12] = s1[two_12]
    mask = mask & (~two_12)

    out[mask] = s0[mask]
    return out

# ---------- bootstrap ----------

def bootstrap_ci_from_tp_union(tp_stack: np.ndarray,
                               union_stack: np.ndarray,
                               reps: int,
                               ci: float,
                               rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    tp_stack: (N_images, C)
    union_stack: (N_images, C)
    Returns dict with keys:
      - 'miou_dist': (reps,) mIoU distribution (0..100 scale)
      - 'iou_dist': (reps, C) per-class IoU distribution (0..100 scale)
    """
    n, c = tp_stack.shape
    alpha = 1.0 - ci
    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    # sample indices (reps, n)
    idx = rng.integers(0, n, size=(reps, n), endpoint=False)
    # sum TP/union across sampled images
    tp_sum = tp_stack[idx].sum(axis=1)           # (reps, C)
    union_sum = union_stack[idx].sum(axis=1)     # (reps, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union_sum > 0, tp_sum / union_sum, 0.0)  # (reps, C)
    miou = iou.mean(axis=1) * 100.0
    iou = iou * 100.0
    return {
        "miou_dist": miou,
        "iou_dist": iou,
        "q_lo": q_lo,
        "q_hi": q_hi,
    }

# ---------- evaluation ----------

from typing import Tuple

def evaluate_combo(idx1: Dict[str, Path],
                   idx2: Dict[str, Path],
                   idx3: Dict[str, Path],
                   idx_gt: Dict[str, Path],
                   subset: int,
                   gt_mat_key: str,
                   pred_mat_key: str,
                   bootstrap_reps: int = 0,
                   ci: float = 0.95,
                   rng: np.random.Generator | None = None,
                   collect_classwise: bool = False,
                   progress_bar=None) -> Dict[str, float]:
    keys = intersection_keys(idx1, idx2, idx3, idx_gt)
    if subset and subset > 0:
        keys = keys[:subset]
    if not keys:
        base = {
            "n_images": 0,
            "agree_all_frac_mean": 0.0,
            "agree12_frac_mean": 0.0,
            "agree13_frac_mean": 0.0,
            "agree23_frac_mean": 0.0,
            "miou1_imgmean": 0.0,
            "miou2_imgmean": 0.0,
            "miou3_imgmean": 0.0,
            "miouE_imgmean": 0.0,
            "miou1_dataset": 0.0,
            "miou2_dataset": 0.0,
            "miou3_dataset": 0.0,
            "miouE_dataset": 0.0,
        }
        if bootstrap_reps > 0:
            base.update({
                "miouE_ci_lo": 0.0, "miouE_ci_hi": 0.0,
                "gain_ci_lo": 0.0, "gain_ci_hi": 0.0,
            })
        return base

    agree_all = []
    agree12 = []
    agree13 = []
    agree23 = []
    miou1_list = []
    miou2_list = []
    miou3_list = []
    miouE_list = []

    hist1 = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    hist2 = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    hist3 = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    histE = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    # For bootstrap: per-image TP and Union for each variant
    tp1_list, un1_list = [], []
    tp2_list, un2_list = [], []
    tp3_list, un3_list = [], []
    tpE_list, unE_list = [], []

    for k in keys:
        p1 = load_mask_auto(idx1[k], mat_key=pred_mat_key)
        p2 = load_mask_auto(idx2[k], mat_key=pred_mat_key)
        p3 = load_mask_auto(idx3[k], mat_key=pred_mat_key)
        gt = load_mask_auto(idx_gt[k], mat_key=gt_mat_key)

        if p1.shape != gt.shape:
            p1 = resize_mask_if_needed(p1, gt.shape)
        if p2.shape != gt.shape:
            p2 = resize_mask_if_needed(p2, gt.shape)
        if p3.shape != gt.shape:
            p3 = resize_mask_if_needed(p3, gt.shape)

        total = gt.size
        all_agree = int(np.sum((p1 == p2) & (p2 == p3))) / total
        a12 = int(np.sum(p1 == p2)) / total
        a13 = int(np.sum(p1 == p3)) / total
        a23 = int(np.sum(p2 == p3)) / total

        agree_all.append(all_agree)
        agree12.append(a12)
        agree13.append(a13)
        agree23.append(a23)

        h1 = compute_confusion(p1, gt)
        h2 = compute_confusion(p2, gt)
        h3 = compute_confusion(p3, gt)

        _, m1 = iou_from_confusion(h1)
        _, m2 = iou_from_confusion(h2)
        _, m3 = iou_from_confusion(h3)

        miou1_list.append(m1 * 100.0)
        miou2_list.append(m2 * 100.0)
        miou3_list.append(m3 * 100.0)

        pe = majority_vote_first_wins(p1, p2, p3)
        he = compute_confusion(pe, gt)
        _, me = iou_from_confusion(he)
        miouE_list.append(me * 100.0)

        hist1 += h1
        hist2 += h2
        hist3 += h3
        histE += he

        if bootstrap_reps > 0:
            tp1, un1 = tp_union_from_confusion(h1)
            tp2, un2 = tp_union_from_confusion(h2)
            tp3, un3 = tp_union_from_confusion(h3)
            tpE, unE = tp_union_from_confusion(he)
            tp1_list.append(tp1); un1_list.append(un1)
            tp2_list.append(tp2); un2_list.append(un2)
            tp3_list.append(tp3); un3_list.append(un3)
            tpE_list.append(tpE); unE_list.append(unE)

        if progress_bar is not None:
            progress_bar.update(1)

    _, miou1_dataset = iou_from_confusion(hist1)
    _, miou2_dataset = iou_from_confusion(hist2)
    _, miou3_dataset = iou_from_confusion(hist3)
    _, miouE_dataset = iou_from_confusion(histE)

    out = {
        "n_images": len(keys),
        "agree_all_frac_mean": float(np.mean(agree_all)),
        "agree12_frac_mean": float(np.mean(agree12)),
        "agree13_frac_mean": float(np.mean(agree13)),
        "agree23_frac_mean": float(np.mean(agree23)),
        "miou1_imgmean": float(np.mean(miou1_list)),
        "miou2_imgmean": float(np.mean(miou2_list)),
        "miou3_imgmean": float(np.mean(miou3_list)),
        "miouE_imgmean": float(np.mean(miouE_list)),
        "miou1_dataset": float(miou1_dataset * 100.0),
        "miou2_dataset": float(miou2_dataset * 100.0),
        "miou3_dataset": float(miou3_dataset * 100.0),
        "miouE_dataset": float(miouE_dataset * 100.0),
    }

    # Bootstrap (dataset-level, and optional classwise)
    if bootstrap_reps > 0:
        if rng is None:
            rng = np.random.default_rng(12345)
        tp1_stack = np.vstack(tp1_list)  # (N, C)
        un1_stack = np.vstack(un1_list)
        tp2_stack = np.vstack(tp2_list)
        un2_stack = np.vstack(un2_list)
        tp3_stack = np.vstack(tp3_list)
        un3_stack = np.vstack(un3_list)
        tpE_stack = np.vstack(tpE_list)
        unE_stack = np.vstack(unE_list)

        dist1 = bootstrap_ci_from_tp_union(tp1_stack, un1_stack, bootstrap_reps, ci, rng)
        dist2 = bootstrap_ci_from_tp_union(tp2_stack, un2_stack, bootstrap_reps, ci, rng)
        dist3 = bootstrap_ci_from_tp_union(tp3_stack, un3_stack, bootstrap_reps, ci, rng)
        distE = bootstrap_ci_from_tp_union(tpE_stack, unE_stack, bootstrap_reps, ci, rng)

        # replicate-wise best single
        best_single_miou = np.maximum.reduce([dist1["miou_dist"], dist2["miou_dist"], dist3["miou_dist"]])
        gain_dist = distE["miou_dist"] - best_single_miou

        miouE_ci_lo = float(np.percentile(distE["miou_dist"], distE["q_lo"]))
        miouE_ci_hi = float(np.percentile(distE["miou_dist"], distE["q_hi"]))
        gain_ci_lo = float(np.percentile(gain_dist, distE["q_lo"]))
        gain_ci_hi = float(np.percentile(gain_dist, distE["q_hi"]))

        out.update({
            "miouE_ci_lo": miouE_ci_lo,
            "miouE_ci_hi": miouE_ci_hi,
            "gain_ci_lo": gain_ci_lo,
            "gain_ci_hi": gain_ci_hi,
        })

        # classwise details (optional)
        if collect_classwise:
            # replicate-wise best single per class (take mean across classes AFTER picking best per-class or per-model?)
            # Here we offer both: ensemble IoU class CIs, and gain per class vs best single (per class).
            best_single_iou_per_class = np.maximum.reduce([dist1["iou_dist"], dist2["iou_dist"], dist3["iou_dist"]])  # (reps, C)
            gain_per_class_dist = distE["iou_dist"] - best_single_iou_per_class

            out["__classwise__"] = {
                "iouE_ci_lo": np.percentile(distE["iou_dist"], distE["q_lo"], axis=0).astype(float).tolist(),
                "iouE_ci_hi": np.percentile(distE["iou_dist"], distE["q_hi"], axis=0).astype(float).tolist(),
                "gain_ci_lo": np.percentile(gain_per_class_dist, distE["q_lo"], axis=0).astype(float).tolist(),
                "gain_ci_hi": np.percentile(gain_per_class_dist, distE["q_hi"], axis=0).astype(float).tolist(),
            }

    return out

# ---------- utilities ----------

def resolve_out_csv_path(user_arg: str) -> str:
    """
    If user passed a directory or '.', create a default file inside it.
    """
    if not user_arg:
        return ""
    p = Path(user_arg)
    if p.is_dir() or str(user_arg).endswith(os.sep) or user_arg in (".", "./"):
        return str((p / "ensemble_results.csv").resolve())
    # If parent dir exists or is creatable, OK; otherwise leave as is and let open() raise a clear error
    return user_arg

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Try all 3 way ensembles across gc_masks_* folders, GT in npy, npz, mat, png")
    ap.add_argument("--root", type=str, default=".", help="Project root, where gc_masks_* folders live")
    ap.add_argument("--gt_dir", type=str, required=True, help="Ground truth folder")
    ap.add_argument("--pred_glob", type=str, default="gc_masks_*", help="Glob to find prediction folders")
    ap.add_argument("--subset", type=int, default=0, help="If positive, evaluate only first N images per combo")
    ap.add_argument("--out_csv", type=str, default="", help="Optional CSV path (file or directory)")
    ap.add_argument("--out", type=str, default="", help="Alias for --out_csv (file or directory)")
    ap.add_argument("--per_class_json", type=str, default="", help="Optional JSON path for classwise CI details")
    ap.add_argument("--strip_pred", type=str, default=",".join(DEFAULT_STRIP_SUFFIXES),
                    help="Comma separated suffixes to strip from prediction stems")
    ap.add_argument("--strip_gt", type=str, default=",".join(DEFAULT_STRIP_SUFFIXES),
                    help="Comma separated suffixes to strip from GT stems")
    ap.add_argument("--pred_exts", type=str, default=",".join(DEFAULT_PRED_EXTS),
                    help="Comma separated list of allowed prediction extensions")
    ap.add_argument("--gt_exts", type=str, default=",".join(DEFAULT_GT_EXTS),
                    help="Comma separated list of allowed GT extensions")
    ap.add_argument("--pred_mat_key", type=str, default="mtx", help="MAT key for predictions when using .mat")
    ap.add_argument("--gt_mat_key", type=str, default="mtx", help="MAT key for GT when using .mat")
    # Bootstrap
    ap.add_argument("--bootstrap_reps", type=int, default=0, help="Number of bootstrap replicates (0 disables)")
    ap.add_argument("--ci", type=float, default=0.95, help="Confidence level (e.g., 0.95)")
    ap.add_argument("--bootstrap_seed", type=int, default=12345, help="Seed for bootstrap RNG")
    ap.add_argument("--workers", type=int, default=0,
                help="Number of parallel workers (0=auto cpu_count, 1=sequential with per-image bars)")
    args = ap.parse_args()

    root = Path(args.root)
    gt_dir = Path(args.gt_dir)
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    if not gt_dir.exists():
        print(f"Ground truth folder not found: {gt_dir}")
        sys.exit(2)

    pred_dirs = sorted([p for p in root.glob(args.pred_glob) if p.is_dir()])
    if len(pred_dirs) < 3:
        print(f"Found fewer than 3 prediction folders under {root}, glob used: {args.pred_glob}")
        sys.exit(3)

    labels = [p.name.replace("gc_masks_", "") for p in pred_dirs]
    print(f"Found {len(pred_dirs)} prediction folders")
    for p, lab in zip(pred_dirs, labels):
        print(f"  {lab}: {p}")

    strip_pred = [s for s in args.strip_pred.split(",") if s]
    strip_gt = [s for s in args.strip_gt.split(",") if s]
    pred_exts = parse_exts(args.pred_exts, DEFAULT_PRED_EXTS)
    gt_exts = parse_exts(args.gt_exts, DEFAULT_GT_EXTS)

    # indices
    pred_indices = []
    for p, lab in zip(pred_dirs, labels):
        idx = build_index_recursive(p, strip_pred, pred_exts)
        pred_indices.append(idx)
        print(f"Indexed {lab}, files {len(idx)} after normalization")

    idx_gt = build_index_recursive(gt_dir, strip_gt, gt_exts)
    print(f"Indexed GT, files {len(idx_gt)} after normalization")

    if len(idx_gt) == 0:
        print("No GT files found with the given extensions, check --gt_dir and --gt_exts")
        print("Example, if your GT are .npy only, run with --gt_exts npy")
        sys.exit(4)

    # quick shared diagnostic
    shared_all = set(idx_gt.keys())
    for idx in pred_indices:
        shared_all &= set(idx.keys())
    print(f"Common keys across ALL prediction folders and GT, count {len(shared_all)}")

    combos = list(itertools.combinations(range(len(pred_dirs)), 3))
    print(f"Will evaluate {len(combos)} three way combinations")

    results = []
    per_class_payload = []  # for optional JSON

    if workers == 1:
        # ----- SEQUENTIAL (keeps your nested per-image tqdm) -----
        rng = np.random.default_rng(args.bootstrap_seed) if args.bootstrap_reps > 0 else None
        outer = tqdm(total=len(combos), unit="trio", dynamic_ncols=True) if tqdm is not None else None

        for a, b, c in combos:
            lab1, lab2, lab3 = labels[a], labels[b], labels[c]
            trio_name = f"{lab1},{lab2},{lab3}"
            if outer is not None:
                outer.set_description_str(trio_name)

            idx1, idx2, idx3 = pred_indices[a], pred_indices[b], pred_indices[c]
            keys = intersection_keys(idx1, idx2, idx3, idx_gt)
            if args.subset and args.subset > 0:
                keys = keys[:args.subset]
            if not keys:
                print(f"Skip combo {trio_name}, no common files after normalization")
                if outer is not None:
                    outer.update(1)
                continue

            inner = tqdm(total=len(keys), desc=trio_name, unit="img",
                        leave=False, dynamic_ncols=True) if tqdm is not None else None
            try:
                metrics = evaluate_combo(
                    idx1, idx2, idx3, idx_gt,
                    subset=args.subset,
                    gt_mat_key=args.gt_mat_key,
                    pred_mat_key=args.pred_mat_key,
                    bootstrap_reps=args.bootstrap_reps,
                    ci=args.ci,
                    rng=rng,
                    collect_classwise=bool(args.per_class_json),
                    progress_bar=inner
                )
            finally:
                if inner is not None:
                    inner.close()

            row = {
                "cs1": lab1, "cs2": lab2, "cs3": lab3, **metrics,
                "best_single_dataset": max(metrics["miou1_dataset"], metrics["miou2_dataset"], metrics["miou3_dataset"]),
                "ensemble_gain_over_best_single": metrics["miouE_dataset"] - max(metrics["miou1_dataset"], metrics["miou2_dataset"], metrics["miou3_dataset"]),
            }
            results.append(row)

            if "__classwise__" in metrics and args.per_class_json:
                per_class_payload.append({
                    "cs1": lab1, "cs2": lab2, "cs3": lab3,
                    "n_images": metrics["n_images"],
                    "iouE_ci_lo": metrics["__classwise__"]["iouE_ci_lo"],
                    "iouE_ci_hi": metrics["__classwise__"]["iouE_ci_hi"],
                    "gain_ci_lo": metrics["__classwise__"]["gain_ci_lo"],
                    "gain_ci_hi": metrics["__classwise__"]["gain_ci_hi"],
                })

            print(f"Done combo {trio_name}, images {row['n_images']}, "
                f"ensemble mIoU dataset {row['miouE_dataset']:.3f}, "
                f"gain {row['ensemble_gain_over_best_single']:.3f}"
                + ("" if args.bootstrap_reps <= 0 else
                    f" | miouE CI [{row.get('miouE_ci_lo', 0.0):.3f}, {row.get('miouE_ci_hi', 0.0):.3f}], "
                    f"gain CI [{row.get('gain_ci_lo', 0.0):.3f}, {row.get('gain_ci_hi', 0.0):.3f}]"))

            if outer is not None:
                outer.update(1)

        if outer is not None:
            outer.close()

    else:
        # ----- PARALLEL (one trio per process) -----
        print(f"Parallel mode: {workers} workers")
        # Build a compact context that’s sent ONCE to each worker
        worker_ctx = {
            "labels": labels,
            "pred_indices": pred_indices,   # large, but sent once per worker
            "idx_gt": idx_gt,
            "args": {
                "subset": args.subset,
                "gt_mat_key": args.gt_mat_key,
                "pred_mat_key": args.pred_mat_key,
                "bootstrap_reps": args.bootstrap_reps,
                "ci": args.ci,
                "bootstrap_seed": args.bootstrap_seed,
                "collect_classwise": bool(args.per_class_json),
            }
        }

        outer = tqdm(total=len(combos), unit="trio", dynamic_ncols=True) if tqdm is not None else None
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    initializer=_init_worker,
                                    initargs=(worker_ctx,)) as ex:
            futs = [ex.submit(_run_combo_task, combo) for combo in combos]
            for fut in cf.as_completed(futs):
                lab1, lab2, lab3, metrics = fut.result()

                row = {
                    "cs1": lab1, "cs2": lab2, "cs3": lab3, **metrics,
                    "best_single_dataset": max(metrics["miou1_dataset"], metrics["miou2_dataset"], metrics["miou3_dataset"]),
                    "ensemble_gain_over_best_single": metrics["miouE_dataset"] - max(metrics["miou1_dataset"], metrics["miou2_dataset"], metrics["miou3_dataset"]),
                }
                results.append(row)

                if "__classwise__" in metrics and args.per_class_json:
                    per_class_payload.append({
                        "cs1": lab1, "cs2": lab2, "cs3": lab3,
                        "n_images": metrics["n_images"],
                        "iouE_ci_lo": metrics["__classwise__"]["iouE_ci_lo"],
                        "iouE_ci_hi": metrics["__classwise__"]["iouE_ci_hi"],
                        "gain_ci_lo": metrics["__classwise__"]["gain_ci_lo"],
                        "gain_ci_hi": metrics["__classwise__"]["gain_ci_hi"],
                    })

                trio_name = f"{lab1},{lab2},{lab3}"
                print(f"Done combo {trio_name}, images {row['n_images']}, "
                    f"ensemble mIoU dataset {row['miouE_dataset']:.3f}, "
                    f"gain {row['ensemble_gain_over_best_single']:.3f}"
                    + ("" if args.bootstrap_reps <= 0 else
                        f" | miouE CI [{row.get('miouE_ci_lo', 0.0):.3f}, {row.get('miouE_ci_hi', 0.0):.3f}], "
                        f"gain CI [{row.get('gain_ci_lo', 0.0):.3f}, {row.get('gain_ci_hi', 0.0):.3f}]"))

                if outer is not None:
                    outer.update(1)

        if outer is not None:
            outer.close()

    # sort
    results.sort(key=lambda r: r["miouE_dataset"], reverse=True)

    # print top 20
    print("\nTop 20 by ensemble mIoU, dataset level")
    header = ["rank", "cs1", "cs2", "cs3", "n_images", "miouE_dataset", "best_single_dataset", "gain"]
    if args.bootstrap_reps > 0:
        header += ["miouE_ci_lo", "miouE_ci_hi", "gain_ci_lo", "gain_ci_hi"]
    print("\t".join(header))
    for idx, r in enumerate(results[:20], start=1):
        row = [
            str(idx),
            r["cs1"], r["cs2"], r["cs3"],
            str(r["n_images"]),
            f"{r['miouE_dataset']:.3f}",
            f"{r['best_single_dataset']:.3f}",
            f"{r['ensemble_gain_over_best_single']:.3f}",
        ]
        if args.bootstrap_reps > 0:
            row += [
                f"{r.get('miouE_ci_lo', 0.0):.3f}",
                f"{r.get('miouE_ci_hi', 0.0):.3f}",
                f"{r.get('gain_ci_lo', 0.0):.3f}",
                f"{r.get('gain_ci_hi', 0.0):.3f}",
            ]
        print("\t".join(row))

    # write CSV
    out_path = resolve_out_csv_path(args.out_csv or args.out)
    if out_path:
        cols = [
            "cs1", "cs2", "cs3", "n_images",
            "agree_all_frac_mean", "agree12_frac_mean", "agree13_frac_mean", "agree23_frac_mean",
            "miou1_imgmean", "miou2_imgmean", "miou3_imgmean", "miouE_imgmean",
            "miou1_dataset", "miou2_dataset", "miou3_dataset", "miouE_dataset",
            "best_single_dataset", "ensemble_gain_over_best_single",
        ]
        if args.bootstrap_reps > 0:
            cols += ["miouE_ci_lo", "miouE_ci_hi", "gain_ci_lo", "gain_ci_hi"]

        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for r in results:
                writer.writerow([r.get(k, "") for k in cols])
        print(f"Wrote CSV to {out_path}")

    # optional classwise JSON
    if args.per_class_json and per_class_payload:
        pc_path = Path(args.per_class_json)
        pc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pc_path, "w") as jf:
            json.dump({
                "n_classes": N_CLASSES,
                "classes": list(range(N_CLASSES)),
                "combos": per_class_payload
            }, jf, indent=2)
        print(f"Wrote classwise CI JSON to {pc_path}")

if __name__ == "__main__":
    main()
