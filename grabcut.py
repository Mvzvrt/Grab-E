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
"""

from __future__ import annotations

import argparse
import json
import warnings
import functools
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, List, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

# Optional dependency: Colour Science, used for CAM02 forward model when available.
try:
    import colour  # type: ignore
    _HAS_COLOUR = True
except Exception:
    _HAS_COLOUR = False

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

def _ensure_hwc3(arr: np.ndarray, H: int, W: int, where: str = "converter") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == H * W:
        return a.reshape(H, W, 3)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[1] == H and a.shape[2] == W:
        return a.transpose(1, 2, 0)
    if a.ndim == 1 and a.size == H * W * 3:
        return a.reshape(H, W, 3)
    raise ValueError(f"{where} produced array with shape {a.shape}, expected {(H, W, 3)}")


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
        a = np.load(p, mmap_mode="r")  # memory-mapped read for large arrays
        a = np.asarray(a, dtype=np.int32)
    elif ext in (".png", ".bmp", ".tif", ".tiff"):
        a = np.asarray(Image.open(p).convert("P"), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported annotation format: {ext}")

    # Validate annotation values
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


# ---------- color-space helpers ----------

def _scale_to_uint8_per_channel(x: np.ndarray) -> np.ndarray:
    """Vectorized min max scale per channel to 0..255 uint8."""
    x = x.astype(np.float32, copy=False)
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    x_range = x_max - x_min
    np.putmask(x_range, x_range == 0, 1.0)
    normalized = (x - x_min) / x_range * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _hsv_conic_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    HSV conic form, H prime equals V, S prime equals V times S times sin H, V prime equals V times S times cos H.
    OpenCV HSV has H in [0,180] representing [0,360) degrees.
    """
    img_rgb = img_rgb_u8.astype(np.uint8, copy=False)
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    H_rad = (H * np.pi) / 90.0

    c0 = V
    c1 = V * S * np.sin(H_rad)
    c2 = V * S * np.cos(H_rad)

    C0 = np.clip(c0 * 255.0, 0, 255).astype(np.uint8)
    C1 = np.clip((c1 + 1.0) * 127.5, 0, 255).astype(np.uint8)
    C2 = np.clip((c2 + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return np.stack([C0, C1, C2], axis=2)


def _lab_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """CIELAB, D65, via OpenCV."""
    return cv.cvtColor(img_rgb_u8, cv.COLOR_RGB2LAB)


# ---------- CAM02 SCD using Colour if available, else colorspacious ----------

def _rgb_u8_to_float01(img_rgb_u8: np.ndarray) -> np.ndarray:
    return img_rgb_u8.astype(np.float32, copy=False) / 255.0


def _cam02_scd_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # Prefer colorspacious by default
    H, W = img_rgb_u8.shape[:2]

    try:
        from colorspacious import cspace_convert  # type: ignore
        jab = cspace_convert(
            img_rgb_u8.astype(np.float32) / 255.0,
            "sRGB1",
            "CAM02-SCD"
        )
        jab = np.asarray(jab, dtype=np.float32)
        if jab.ndim == 2 and jab.shape == (H * W, 3):
            jab = jab.reshape(H, W, 3)
        jab = _ensure_hwc3(jab, H, W, where="cam02_scd(colorspacious)")
        return _scale_to_uint8_per_channel(jab)

    except Exception:
        raise RuntimeError("Use colorspacious")

# ---------- CAM16 vectorized implementation, no external deps ----------

_CAT16 = np.array([
    [ 0.401288,  0.650173, -0.051461],
    [-0.250268,  1.204414,  0.045854],
    [-0.002079,  0.048952,  0.953127],
], dtype=np.float32)

_SRGB_TO_XYZ = np.array([
    [0.412456, 0.357576, 0.180438],
    [0.212673, 0.715152, 0.072175],
    [0.019334, 0.119192, 0.950304],
], dtype=np.float32)

# Matrices for BT.2020 conversions used by ICtCp
_XYZ_TO_BT2020 = np.array([
    [ 1.71666343, -0.35567332, -0.25336809],
    [-0.66667384,  1.61645574,  0.01576830],
    [ 0.01764248, -0.04277698,  0.94224328],
], dtype=np.float32)

# ICtCp matrices from BT.2100 PQ form, scaled by 1 4096
_ICTCP_RGB2020_TO_LMS = (1.0 / 4096.0) * np.array([
    [1688, 2146,  262],
    [ 683, 2951,  462],
    [  99,  309, 3688],
], dtype=np.float32)

_ICTCP_LMS_TO_ICTCP_PQ = (1.0 / 4096.0) * np.array([
    [ 2048,   2048,     0],
    [ 6610, -13613,  7003],
    [17933, -17390,  -543],
], dtype=np.float32)

def _whitepoint_D65_XYZ() -> np.ndarray:
    x = 0.31270
    y = 0.32900
    Y = 100.0
    X = Y * x / y
    Z = Y * (1.0 - x - y) / y
    return np.array([X, Y, Z], dtype=np.float32)


def _cam16_nonlinear_response(t: np.ndarray) -> np.ndarray:
    """
    Apply CAM16 nonlinear response to t equals F_L times RGB divided by 100.
    """
    t = t.astype(np.float32, copy=False)
    out = np.empty_like(t, dtype=np.float32)
    pos = t >= 0
    if np.any(pos):
        tp = t[pos]
        tp42 = np.power(tp, 0.42)
        out[pos] = 400.0 * tp42 / (tp42 + 27.13) + 0.1
    neg = ~pos
    if np.any(neg):
        tn = -t[neg]
        tn42 = np.power(tn, 0.42)
        out[neg] = -400.0 * tn42 / (tn42 + 27.13) + 0.1
    return out


@functools.lru_cache(maxsize=None)
def _cam16_setup():
    """
    Precompute context for CAM16 under dim surround, sRGB like viewing.
    """
    XYZ_w = _whitepoint_D65_XYZ()
    F, c, Nc = 0.9, 0.59, 0.9
    E_w = 64.0
    L_w = E_w / np.pi
    Y_b = 20.0
    L_A = (L_w * Y_b) / XYZ_w[1]

    RGB_w = _CAT16 @ XYZ_w
    D = F * (1.0 - (1.0 / 3.6) * np.exp(-(L_A + 42.0) / 92.0))
    D = np.clip(D, 0.0, 1.0)
    D_RGB = D * XYZ_w[1] / RGB_w + 1.0 - D

    k = 1.0 / (5.0 * L_A + 1.0)
    F_L = 0.2 * k**4 * 5.0 * L_A + 0.1 * (1.0 - k**4)**2 * (5.0 * L_A)**(1.0 / 3.0)

    n = Y_b / XYZ_w[1]
    z = 1.48 + n**0.5
    N_bb = 0.725 * (1.0 / n)**0.2
    N_cb = N_bb

    RGB_wc = D_RGB * RGB_w
    t = (F_L * RGB_wc / 100.0)
    RGB_aw = _cam16_nonlinear_response(t)
    A_w = (np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) @ RGB_aw - 0.305) * N_bb

    return {
        "F_L": float(F_L), "c": float(c), "Nc": float(Nc), "n": float(n), "z": float(z),
        "N_bb": float(N_bb), "N_cb": float(N_cb), "A_w": float(A_w),
        "D_RGB": D_RGB.astype(np.float32),
    }


def _srgb_u8_to_linear01(img_rgb_u8: np.ndarray) -> np.ndarray:
    x = img_rgb_u8.astype(np.float32, copy=False) / 255.0
    mask = x <= 0.04045
    y = np.empty_like(x, dtype=np.float32)
    y[mask]  = x[mask] / 12.92
    y[~mask] = ((x[~mask] + 0.055) / 1.055) ** 2.4
    return y


def _cam16_forward_JMh_from_rgb(img_rgb_u8: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CAM16 J, M, h for an HxWx3 RGB uint8 image.
    Returns H, W, and flattened J, M, h vectors.
    """
    ctx = _cam16_setup()
    F_L = ctx["F_L"]
    c = ctx["c"]
    Nc = ctx["Nc"]
    n = ctx["n"]
    z = ctx["z"]
    N_bb = ctx["N_bb"]
    N_cb = ctx["N_cb"]
    A_w = ctx["A_w"]
    D_RGB = ctx["D_RGB"]

    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    H, W, _ = rgb_lin.shape
    rgb_flat = rgb_lin.reshape(-1, 3)
    XYZ = (rgb_flat @ _SRGB_TO_XYZ.T) * 100.0

    RGB = (XYZ @ _CAT16.T) * D_RGB

    t = (F_L * RGB / 100.0)
    RGB_a = _cam16_nonlinear_response(t)

    a = RGB_a @ np.array([1.0, -12.0 / 11.0, 1.0 / 11.0], dtype=np.float32)
    b = RGB_a @ np.array([1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0], dtype=np.float32)

    h = np.degrees(np.arctan2(b, a)).astype(np.float32)
    h[h < 0.0] += 360.0
    h_rad = np.radians(h)

    e = 0.25 * (np.cos(h_rad + 2.0) + 3.8)

    A = (RGB_a @ np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) - 0.305) * N_bb

    A_w_safe = float(np.maximum(A_w, 1e-6))
    J = 100.0 * np.power(np.maximum(A, 0.0) / A_w_safe, c * z)

    p1 = (50000.0 / 13.0) * Nc * N_cb * e * np.sqrt(a * a + b * b)
    p2 = RGB_a @ np.array([1.0, 1.0, 21.0 / 20.0], dtype=np.float32)
    p2 = np.where(np.abs(p2) < 1e-6, 1e-6, p2)
    C = np.power(p1 / p2, 0.9) * np.sqrt(np.maximum(J, 0.0) / 100.0) * np.power(1.64 - 0.29 ** n, 0.73)
    M = C * (ctx["F_L"] ** 0.25)

    return H, W, J.astype(np.float32), M.astype(np.float32), h.astype(np.float32)


def _cam16_ucs_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    H, W, J, M, h = _cam16_forward_JMh_from_rgb(img_rgb_u8)
    Jp = 1.7 * J / (1.0 + 0.007 * J)
    Mp = np.log1p(0.0228 * M) / 0.0228
    h_rad = np.radians(h)
    ap = Mp * np.cos(h_rad)
    bp = Mp * np.sin(h_rad)
    return np.stack([Jp, ap, bp], axis=1).reshape(H, W, 3).astype(np.float32)


def _cam16_scd_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    H, W, J, M, h = _cam16_forward_JMh_from_rgb(img_rgb_u8)
    c1, c2 = 0.007, 0.0363
    Jp = ((1.0 + 100.0 * c1) * J) / (1.0 + c1 * J)
    Mp = np.log1p(c2 * M) / c2
    h_rad = np.radians(h)
    ap = Mp * np.cos(h_rad)
    bp = Mp * np.sin(h_rad)
    jab = np.stack([Jp, ap, bp], axis=1).reshape(H, W, 3).astype(np.float32)
    return _scale_to_uint8_per_channel(jab)


# ---------- Added modern color spaces ----------

# OKLab matrices from Ottosson
_OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
_OKLAB_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)

def _oklab_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    lms = rgb_lin @ _OKLAB_M1.T
    # Guard small negatives before cube root
    lms = np.clip(lms, 0.0, None)
    lms_cbrt = np.cbrt(lms)
    lab = lms_cbrt @ _OKLAB_M2.T
    return _scale_to_uint8_per_channel(lab)

def _oklch_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # L in 0..1, a, b approx in [-, +]
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    lms = rgb_lin @ _OKLAB_M1.T
    lms = np.clip(lms, 0.0, None)
    lms_cbrt = np.cbrt(lms)
    lab = lms_cbrt @ _OKLAB_M2.T
    L = np.clip(lab[:, :, 0], 0.0, 1.0)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    h = np.degrees(np.arctan2(b, a))
    h[h < 0.0] += 360.0
    # Map to uint8, fixed scale for L and h, per image scale for C
    L8 = np.clip(L * 255.0, 0, 255).astype(np.uint8)
    # Per channel scale for C
    C8 = _scale_to_uint8_per_channel(C[:, :, None])[:, :, 0]
    h8 = np.clip((h / 360.0) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([L8, C8, h8], axis=2)

# JzAzBz, PQ based, constants per Safdar
_JZ_c1 = 0.8359375
_JZ_c2 = 18.8515625
_JZ_c3 = 18.6875
_JZ_n  = 0.1593017578125  # 2610 / 16384
_JZ_p  = 78.84375         # 2523 / 32
_JZ_b  = 1.15
_JZ_g  = 0.66
_JZ_d  = -0.56
_JZ_d0 = 1.6295499532821566e-11

_JZ_M1 = np.array([
    [ 0.41478972, 0.57999900, 0.01464800],
    [-0.20151000, 1.12064900, 0.05310080],
    [-0.01660080, 0.26480000, 0.66847990],
], dtype=np.float32)

_JZ_M2 = np.array([
    [0.5,       0.5,       0.0     ],
    [3.524000, -4.066708,  0.542708],
    [0.199076,  1.096799, -1.295875],
], dtype=np.float32)

def _pq_oetf_inverse(x: np.ndarray) -> np.ndarray:
    """Apply ST 2084 inverse EOTF, maps linear relative luminance to PQ signal."""
    x = np.clip(x.astype(np.float32), 0.0, None)
    x_m = np.power(x, _JZ_n)
    num = _JZ_c1 + _JZ_c2 * x_m
    den = 1.0 + _JZ_c3 * x_m
    y = np.power(num / np.maximum(den, 1e-12), _JZ_p)
    return y.astype(np.float32)

def _jzazbz_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # sRGB linear to XYZ
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T  # relative XYZ
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    Xp = _JZ_b * X - (_JZ_b - 1.0) * Z
    Yp = _JZ_g * Y - (_JZ_g - 1.0) * X
    Zp = Z
    XYZp = np.stack([Xp, Yp, Zp], axis=2)
    LMS = XYZp @ _JZ_M1.T
    LMS_p = _pq_oetf_inverse(LMS)  # approximate, using relative units
    IzAzBz = LMS_p @ _JZ_M2.T
    Iz = IzAzBz[:, :, 0]
    az = IzAzBz[:, :, 1]
    bz = IzAzBz[:, :, 2]
    Jz = ((1.0 + _JZ_d) * Iz) / (1.0 + _JZ_d * Iz) - _JZ_d0
    jzazbz = np.stack([Jz, az, bz], axis=2).astype(np.float32)
    return _scale_to_uint8_per_channel(jzazbz)

def _jzczhz_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # Build from JzAzBz then to polar
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    Xp = _JZ_b * X - (_JZ_b - 1.0) * Z
    Yp = _JZ_g * Y - (_JZ_g - 1.0) * X
    Zp = Z
    XYZp = np.stack([Xp, Yp, Zp], axis=2)
    LMS = XYZp @ _JZ_M1.T
    LMS_p = _pq_oetf_inverse(LMS)
    IzAzBz = LMS_p @ _JZ_M2.T
    Iz = IzAzBz[:, :, 0]
    az = IzAzBz[:, :, 1]
    bz = IzAzBz[:, :, 2]
    Jz = ((1.0 + _JZ_d) * Iz) / (1.0 + _JZ_d * Iz) - _JZ_d0
    Cz = np.sqrt(az * az + bz * bz)
    hz = np.degrees(np.arctan2(bz, az))
    hz[hz < 0.0] += 360.0
    # Map to uint8, fixed scale for hue, per image for Cz, scaled Jz
    J8 = _scale_to_uint8_per_channel(Jz[:, :, None])[:, :, 0]
    C8 = _scale_to_uint8_per_channel(Cz[:, :, None])[:, :, 0]
    h8 = np.clip((hz / 360.0) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([J8, C8, h8], axis=2)

def _xyz_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T
    return _scale_to_uint8_per_channel(XYZ)

def _srgb_linear_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    lin = _srgb_u8_to_linear01(img_rgb_u8)
    return np.clip(lin * 255.0, 0, 255).astype(np.uint8)

def _ycbcr_bt709_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # Use gamma encoded R prime, G prime, B prime approximated by sRGB 8 bit normalized
    rp = img_rgb_u8[:, :, 0].astype(np.float32) / 255.0
    gp = img_rgb_u8[:, :, 1].astype(np.float32) / 255.0
    bp = img_rgb_u8[:, :, 2].astype(np.float32) / 255.0
    # BT.709 coefficients
    Yp = 0.2126 * rp + 0.7152 * gp + 0.0722 * bp
    Cb = -0.1146 * rp - 0.3854 * gp + 0.5 * bp
    Cr =  0.5 * rp - 0.4542 * gp - 0.0458 * bp
    # Map to uint8, Y in 0..1, Cb Cr around -0.5..0.5 to 0..1 by offset
    Y8  = np.clip(Yp * 255.0, 0, 255).astype(np.uint8)
    Cb8 = np.clip((Cb + 0.5) * 255.0, 0, 255).astype(np.uint8)
    Cr8 = np.clip((Cr + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([Y8, Cb8, Cr8], axis=2)

def _ictcp_pq_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # sRGB linear -> XYZ -> BT.2020 linear -> LMS -> PQ -> ICtCp
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T
    rgb2020_lin = XYZ @ _XYZ_TO_BT2020.T
    LMS = rgb2020_lin @ _ICTCP_RGB2020_TO_LMS.T
    LMS = np.clip(LMS, 0.0, None)
    LMS_p = _pq_oetf_inverse(LMS)  # approximate, relative units
    ICTCP = LMS_p @ _ICTCP_LMS_TO_ICTCP_PQ.T
    return _scale_to_uint8_per_channel(ICTCP)

# ---------- colorspace router with caching ----------

@functools.lru_cache(maxsize=32)
def get_color_converter(mode: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return a cached converter function for a given mode string."""
    converters = {
        'rgb': lambda x: x,
        'hsv_conic': _hsv_conic_from_rgb,
        'cielab': _lab_from_rgb,
        'c02_scd': _cam02_scd_from_rgb,
        'c16_scd': _cam16_scd_from_rgb,
        # new modern spaces
        'oklab': _oklab_from_rgb,
        'oklch': _oklch_from_rgb,
        'jzazbz': _jzazbz_from_rgb,
        'jzczhz': _jzczhz_from_rgb,
        'ictcp_pq': _ictcp_pq_from_rgb,
        'xyz': _xyz_from_rgb,
        'ycbcr_bt709': _ycbcr_bt709_from_rgb,
        'srgb_linear': _srgb_linear_from_rgb,
    }
    return converters.get(mode.lower())


def convert_color_space(img_rgb_u8: np.ndarray, mode: str) -> np.ndarray:
    fn = get_color_converter(mode)
    if fn is None:
        raise ValueError(f"Unsupported color_space: {mode}")
    H, W = img_rgb_u8.shape[:2]
    out = fn(img_rgb_u8)
    out = _ensure_hwc3(out, H, W, where=f"{mode} converter")
    if out.dtype != np.uint8:
        out = _scale_to_uint8_per_channel(out)
    return out



# ---------- OpenCV GrabCut (single call) ----------

def opencv_grabcut_once(img_feats_u8: np.ndarray,
                        seeds_bg: np.ndarray,
                        seeds_fg: np.ndarray,
                        iters: int = 2) -> np.ndarray:
    """
    Run cv2.grabCut once with firm seeds and return a binary mask, 1 FG, 0 BG.
    Works on any 3 channel 8 bit image of per pixel features.
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
        return np.zeros((H, W), dtype=np.uint8)

    mask = np.full((H, W), cv.GC_PR_BGD, dtype=np.uint8)
    mask[seeds_bg] = cv.GC_BGD
    mask[seeds_fg] = cv.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv.grabCut(img_feats_u8, mask, None, bgdModel, fgdModel, int(iters), cv.GC_INIT_WITH_MASK)
    except cv.error as e:
        raise RuntimeError(f"OpenCV GrabCut failed: {e}") from e

    out = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)
    return out


# ---------- multi class wrapper, one vs rest ----------

def run_one_vs_rest(img_feats_u8: np.ndarray,
                    anns: np.ndarray,
                    gc_iters: int = 5,
                    tie_mode: str = "nearest-scribble") -> np.ndarray:
    """
    For each present class c > 1:
      FG seeds = anns == c
      BG seeds = anns == 1 or anns > 1 and not equal to c
    Combine binary masks into a single VOC index map where:
      output 0 = background, output 1..20 = foreground classes, map c -> c - 1.
    """
    H, W = anns.shape
    classes = sorted([int(x) for x in np.unique(anns) if x > 1])
    if not classes:
        return np.zeros((H, W), dtype=np.uint8)

    fg_masks: Dict[int, np.ndarray] = {}
    for c in classes:
        seeds_fg = (anns == c)
        seeds_bg = (anns == 1) | ((anns > 1) & (anns != c))
        y = opencv_grabcut_once(img_feats_u8, seeds_bg=seeds_bg, seeds_fg=seeds_fg, iters=gc_iters)
        fg_masks[c] = y

    stack = np.stack([fg_masks[c] for c in classes], axis=2)
    overlap_count = stack.sum(axis=2)
    any_overlap = (overlap_count > 1).any()

    final = np.zeros((H, W), dtype=np.uint8)

    if not any_overlap or tie_mode != "nearest-scribble":
        for c in classes:
            m = fg_masks[c] > 0
            final[m] = c - 1
        return final

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


# ---------- worker for parallel batch ----------

def _process_single_image(ann_path: str,
                          images_dir: str,
                          output_dir: str,
                          color_space: str,
                          gc_iters: int,
                          tie_mode: str) -> Dict[str, object]:
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

    pred = run_one_vs_rest(img_feats, anns, gc_iters=int(gc_iters), tie_mode=tie_mode)

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

    # algorithm params
    ap.add_argument("--gc_iters", type=int, default=5, help="Iterations for cv2.grabCut, typical 1 to 5")
    ap.add_argument("--tie_mode", type=str, default="nearest-scribble",
                    choices=["nearest-scribble", "first-wins"],
                    help="How to resolve multi class overlaps")

    # color spaces
    ap.add_argument("--color_space", type=str, default="rgb",
                    choices=[
                        "rgb", "hsv_conic", "cielab", "c02_scd", "c16_scd",
                        "oklab", "oklch", "jzazbz", "jzczhz",
                        "ictcp_pq", "xyz", "ycbcr_bt709", "srgb_linear"
                    ],
                    help="Input feature color space. Modern options include oklab, jzazbz, ictcp_pq. "
                         "Legacy include rgb, cielab. Default is rgb.")

    # parallel processing
    ap.add_argument("--parallel", action="store_true", help="Enable parallel processing of images")
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

    if args.parallel:
        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process_single_image,
                    str(ann_path), str(images_dir), str(out_dir),
                    str(args.color_space), int(args.gc_iters), str(args.tie_mode)
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
                # convert to requested feature space
                img_feats = convert_color_space(img_rgb, args.color_space)

                anns = load_anns(ann_path)
                if anns.shape[:2] != img_feats.shape[:2]:
                    anns = cv.resize(anns.astype(np.int32),
                                     (img_feats.shape[1], img_feats.shape[0]),
                                     interpolation=cv.INTER_NEAREST)

                pred = run_one_vs_rest(img_feats, anns, gc_iters=int(args.gc_iters), tie_mode=args.tie_mode)

                out_path = out_dir / f"{base}_index.png"
                save_indexed_png(pred, str(out_path))

                dt = (perf_counter() - t0) * 1000.0
                times_ms.append(dt)
                processed += 1
                tqdm.write(f"[OK] {base} ({dt:.1f} ms) -> {out_path.name}")

            except FileNotFoundError:
                skipped += 1
                tqdm.write(f"[SKIP] {ann_path.name} image file not found: expected at {img_path}")
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
            "max_workers": int(args.max_workers if args.max_workers else (os.cpu_count() or 4) if args.parallel else 0),
        },
        "timing_ms_avg": (float(np.mean(times_ms)) if times_ms else None)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()