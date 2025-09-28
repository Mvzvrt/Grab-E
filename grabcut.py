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

2025-09-28 Option A prep:
- opencv_grabcut_once can now return bgdModel and fgdModel, OpenCV's 1x65 buffers, and the raw GrabCut mask if requested.
- run_one_vs_rest can collect per-class models for export.
- CLI supports --emit_models and --models_dir to write models to NPZ for downstream ensemble fusion.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

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
      - bgdModel, fgdModel are OpenCV's 1x65 float64 buffers that store 5-comp GMMs.
      - raw_mask_states uses the GrabCut labels {0=BGD,1=FGD,2=PR_BGD,3=PR_FGD}.
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
        if collect_models:
            y, bgm, fgm = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters, return_models=True)  # type: ignore
            models_by_class[c] = {"bgdModel": bgm, "fgdModel": fgm}
        else:
            y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters)  # type: ignore
        fg_masks[c] = y  # binary 0 or 1

    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    overlap_count = stack.sum(axis=2)
    any_overlap = (overlap_count > 1).any()

    final = np.zeros((H, W), dtype=np.uint8)

    if not any_overlap or tie_mode != "nearest-scribble":
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1
        return (final, models_by_class) if collect_models else final

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

    return (final, models_by_class) if collect_models else final


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
                          models_dir: Optional[str]) -> Dict[str, object]:
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
    img_feats = convert_color_space(img_rgb, color_space)

    anns = load_anns(ann_p)
    if anns.shape[:2] != img_feats.shape[:2]:
        anns = cv.resize(anns.astype(np.int32),
                         (img_feats.shape[1], img_feats.shape[0]),
                         interpolation=cv.INTER_NEAREST)

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


# ---------- Ensemble Option A, fused unaries and single cut ----------

def _parse_opencv_gmm_5_from_buf(buf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [Inference] Parse OpenCV bgdModel and fgdModel 1 by 65 buffer into
      weights, shape 5
      means, shape 5 by 3
      covariances, shape 5 by 3 by 3
    Assumes layout, [w, m0, m1, m2, 3 by 3 covariance flattened row major] per component.
    Returns weights clipped and normalized.
    """
    b = np.asarray(buf, dtype=np.float64).reshape(-1)
    if b.size != 65:
        raise ValueError(f"Unexpected model buffer length, expected 65, got {b.size}")
    comps = []
    off = 0
    for _ in range(5):
        w = b[off]; off += 1
        m = b[off:off+3]; off += 3
        C = b[off:off+9].reshape(3, 3); off += 9
        comps.append((w, m, C))
    weights = np.array([c[0] for c in comps], dtype=np.float64)
    means = np.stack([c[1] for c in comps], axis=0).astype(np.float64)
    covs = np.stack([c[2] for c in comps], axis=0).astype(np.float64)

    for k in range(5):
        covs[k].flat[::4] += 1e-6

    weights = np.clip(weights, 0.0, np.inf)
    s = weights.sum()
    if s <= 0:
        weights = np.ones(5, dtype=np.float64) / 5.0
    else:
        weights /= s
    return weights, means, covs


def _logpdf_gmm_full(X: np.ndarray,
                     w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute log sum over i of w_i times N(x | mu_i, Sigma_i) for 3D features.
    Input X shape H, W, 3, returns H, W float64.
    """
    Xf = X.astype(np.float64)
    H, W, _ = Xf.shape
    X2 = Xf.reshape(-1, 3)

    K = w.shape[0]
    logps = np.empty((X2.shape[0], K), dtype=np.float64)

    for k in range(K):
        Sk = Sigma[k]
        muk = mu[k]
        try:
            L = np.linalg.cholesky(Sk)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(Sk)
            vals = np.clip(vals, 1e-6, None)
            L = vecs @ np.diag(np.sqrt(vals))
        diff = X2 - muk
        y = np.linalg.solve(L, diff.T)
        maha2 = np.sum(y * y, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        logN = -0.5 * (3 * np.log(2.0 * np.pi) + logdet + maha2)
        logps[:, k] = np.log(max(w[k], 1e-12)) + logN

    m = np.max(logps, axis=1, keepdims=True)
    ll = m + np.log(np.sum(np.exp(logps - m), axis=1, keepdims=True))
    return ll.reshape(H, W)


def _compute_beta_gamma_rgb(img_rgb_u8: np.ndarray, gamma: float = 50.0) -> Tuple[float, float]:
    """
    Compute beta following GrabCut idea, beta equals 1 divided by (2 times mean squared color diff) over 8 neighbors.
    Returns beta and gamma.
    """
    I = img_rgb_u8.astype(np.float64)
    H, W, _ = I.shape
    diffs = []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
        J = I[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx), :]
        K = I[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx), :]
        d = (J - K)
        diffs.append((d * d).sum(axis=2))
    diffs_all = np.concatenate([d.ravel() for d in diffs])
    m = float(np.mean(diffs_all)) if diffs_all.size else 1.0
    beta = 1.0 / (2.0 * max(m, 1e-6))
    return beta, float(gamma)


def _build_pairwise_edges(img_rgb_u8: np.ndarray,
                          beta: float, gamma: float) -> List[Tuple[int, int, float, float]]:
    """
    Build symmetric pairwise Potts edges over 8 neighbors.
    Returns list of (p, q, w, w) edges for pymaxflow add_edge.
    """
    I = img_rgb_u8.astype(np.float64)
    H, W, _ = I.shape

    def idx(y, x):
        return y * W + x

    edges = []
    for y in range(H):
        for x in range(W):
            p = idx(y, x)
            if x + 1 < W:
                d = I[y, x] - I[y, x + 1]
                w = gamma * np.exp(-beta * float(d @ d)) / 1.0
                edges.append((p, idx(y, x + 1), w, w))
            if y + 1 < H:
                d = I[y, x] - I[y + 1, x]
                w = gamma * np.exp(-beta * float(d @ d)) / 1.0
                edges.append((p, idx(y + 1, x), w, w))
            if x + 1 < W and y + 1 < H:
                d = I[y, x] - I[y + 1, x + 1]
                w = gamma * np.exp(-beta * float(d @ d)) / np.sqrt(2.0)
                edges.append((p, idx(y + 1, x + 1), w, w))
            if x - 1 >= 0 and y + 1 < H:
                d = I[y, x] - I[y + 1, x - 1]
                w = gamma * np.exp(-beta * float(d @ d)) / np.sqrt(2.0)
                edges.append((p, idx(y + 1, x - 1), w, w))
    return edges


def _calibrate_margin_on_seeds(margin: np.ndarray,
                               seeds_bg: np.ndarray,
                               seeds_fg: np.ndarray) -> np.ndarray:
    """
    Simple z score calibration using seed pixels.
    """
    S = seeds_bg | seeds_fg
    if np.any(S):
        m = float(np.mean(margin[S]))
        s = float(np.std(margin[S])) or 1.0
        return (margin - m) / s
    m = float(np.mean(margin))
    s = float(np.std(margin)) or 1.0
    return (margin - m) / s


def _fused_unaries_for_class(img_rgb_u8: np.ndarray,
                             trio: List[str],
                             models_fg: List[np.ndarray],
                             models_bg: List[np.ndarray],
                             seeds_bg: np.ndarray,
                             seeds_fg: np.ndarray,
                             alpha_mode: str = "equal",
                             alpha_weights: Optional[List[float]] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fused D_F and D_B from per space log likelihood margins.
    Returns D_F and D_B as float64 arrays H by W.
    """
    H, W, _ = img_rgb_u8.shape
    assert len(trio) == len(models_fg) == len(models_bg) == 3

    margins = []
    for k, cs in enumerate(trio):
        feats = convert_color_space(img_rgb_u8, cs)
        w_f, mu_f, S_f = _parse_opencv_gmm_5_from_buf(models_fg[k])
        w_b, mu_b, S_b = _parse_opencv_gmm_5_from_buf(models_bg[k])
        ll_f = _logpdf_gmm_full(feats, w_f, mu_f, S_f)
        ll_b = _logpdf_gmm_full(feats, w_b, mu_b, S_b)
        m = ll_f - ll_b
        m = _calibrate_margin_on_seeds(m, seeds_bg, seeds_fg)
        margins.append(m)

    margins = np.stack(margins, axis=2)

    if alpha_mode == "equal" or alpha_weights is None:
        alphas = np.ones(3, dtype=np.float64) / 3.0
    else:
        aw = np.array(alpha_weights, dtype=np.float64)
        if aw.shape != (3,):
            raise ValueError("alpha_weights must have length 3")
        s = float(aw.sum())
        alphas = aw / s if s > 0 else np.ones(3, dtype=np.float64) / 3.0

    M = np.tensordot(margins, alphas, axes=([2], [0]))
    DF = np.clip(-M, 0.0, None)
    DB = np.clip(M, 0.0, None)

    scale = float(np.mean(DF + DB)) or 1.0
    DF = DF / scale
    DB = DB / scale

    BIG = 1e6
    DB = DB + (seeds_fg.astype(np.float64) * BIG)
    DF = DF + (seeds_bg.astype(np.float64) * BIG)

    return DF, DB


def _run_fused_cut_for_class(img_rgb_u8: np.ndarray,
                             DF: np.ndarray, DB: np.ndarray,
                             lambda_smooth: float = 1.0,
                             pairwise_gamma: float = 50.0) -> np.ndarray:
    """
    Build an s t graph and run a single cut using PyMaxflow.
    Returns binary mask 0 or 1 for BG or FG.
    """
    try:
        import maxflow
    except Exception as e:
        raise RuntimeError("PyMaxflow is required for fused cut. Please install PyMaxflow.") from e

    H, W, _ = img_rgb_u8.shape
    N = H * W

    g = maxflow.Graph[float](N, N * 4)
    nodes = g.add_grid_nodes((H, W))

    g.add_grid_tedges(nodes, DF, DB)

    beta, gamma = _compute_beta_gamma_rgb(img_rgb_u8, gamma=pairwise_gamma)
    edges = _build_pairwise_edges(img_rgb_u8, beta=beta, gamma=lambda_smooth * gamma)
    for p, q, w1, w2 in edges:
        g.add_edge(p, q, w1, w2)

    g.maxflow()
    seg = g.get_grid_segments(nodes)
    y = (~seg).astype(np.uint8)
    return y


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
                    help="Input feature color space. Modern options include oklab, jzazbz, ictcp_pq. "
                         "Legacy include rgb, cielab. Default is rgb.")

    ap.add_argument("--ensemble_trio", type=str, default="",
                    help="Comma separated trio for fused unaries, example jzazbz,jzczhz,rgb. Enables ensemble fused cut when set.")
    ap.add_argument("--ensemble_mode", type=str, default="",
                    choices=["", "fused-cut"],
                    help="Set to fused cut to run Option A ensemble using learned models per space, then one final cut.")
    ap.add_argument("--gc_iters_models", type=int, default=5,
                    help="Iterations per space to learn models for ensemble.")
    ap.add_argument("--alpha_mode", type=str, default="equal",
                    choices=["equal", "weights"],
                    help="How to set trio weights, equal or provide --alpha_weights.")
    ap.add_argument("--alpha_weights", type=str, default="",
                    help="Comma separated weights for the trio, example 0.4,0.3,0.3, used when --alpha_mode equals weights.")
    ap.add_argument("--lambda_smooth", type=float, default=1.0,
                    help="Pairwise strength multiplier for fused cut.")
    ap.add_argument("--pairwise_gamma", type=float, default=50.0,
                    help="Gamma parameter for pairwise edge weighting.")

    ap.add_argument("--emit_models", action="store_true",
                    help="When set, save per class bgdModel and fgdModel NPZ files for downstream ensemble fusion.")
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
    if args.emit_models:
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

    if args.parallel:
        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process_single_image,
                    str(ann_path), str(images_dir), str(out_dir),
                    str(args.color_space), int(args.gc_iters), str(args.tie_mode),
                    bool(args.emit_models), str(models_out_dir) if args.emit_models else ""
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
                img_feats = convert_color_space(img_rgb, args.color_space)

                anns = load_anns(ann_path)
                if anns.shape[:2] != img_feats.shape[:2]:
                    anns = cv.resize(anns.astype(np.int32),
                                     (img_feats.shape[1], img_feats.shape[0]),
                                     interpolation=cv.INTER_NEAREST)

                written = []
                if args.ensemble_mode == "fused-cut" and args.ensemble_trio:
                    trio = [s.strip() for s in args.ensemble_trio.split(",")]
                    if len(trio) != 3:
                        raise ValueError("ensemble_trio must have exactly three comma separated color spaces")
                    classes = sorted([int(x) for x in np.unique(anns) if int(x) > 1])
                    H, W, _ = img_rgb.shape
                    fg_masks: Dict[int, np.ndarray] = {}
                    for c in classes:
                        seeds_fg = (anns == c)
                        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))

                        models_fg: List[np.ndarray] = []
                        models_bg: List[np.ndarray] = []
                        for cs in trio:
                            feats_cs = convert_color_space(img_rgb, cs)
                            _, bgm, fgm = opencv_grabcut_once(
                                feats_cs, seeds_bg=seeds_bg, seeds_fg=seeds_fg,
                                iters=int(args.gc_iters_models), return_models=True
                            )
                            models_fg.append(fgm)
                            models_bg.append(bgm)

                        if args.alpha_mode == "weights" and args.alpha_weights:
                            alphas = [float(x) for x in args.alpha_weights.split(",")]
                        else:
                            alphas = None

                        DF, DB = _fused_unaries_for_class(
                            img_rgb_u8=img_rgb,
                            trio=trio,
                            models_fg=models_fg,
                            models_bg=models_bg,
                            seeds_bg=seeds_bg,
                            seeds_fg=seeds_fg,
                            alpha_mode=args.alpha_mode,
                            alpha_weights=alphas
                        )
                        y_bin = _run_fused_cut_for_class(
                            img_rgb_u8=img_rgb,
                            DF=DF, DB=DB,
                            lambda_smooth=float(args.lambda_smooth),
                            pairwise_gamma=float(args.pairwise_gamma)
                        )
                        fg_masks[c] = y_bin

                    stack = np.stack([fg_masks[c] for c in classes], axis=2)
                    overlap_count = stack.sum(axis=2)
                    final = np.zeros((H, W), dtype=np.uint8)

                    if not (overlap_count > 1).any() or args.tie_mode != "nearest-scribble":
                        for c in classes:
                            m = fg_masks[c] > 0
                            final[m] = c - 1
                    else:
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
                    pred = final
                else:
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
            "parallel": bool(args.parallel),
            "emit_models": bool(args.emit_models),
            "models_dir": str(models_out_dir) if args.emit_models else None,
            "max_workers": int(args.max_workers if args.max_workers else (os.cpu_count() or 4) if args.parallel else 0),
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()