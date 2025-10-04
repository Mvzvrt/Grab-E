"""
tight_roi_pre.py

One-function export to build a tight ROI from scribbles for GrabCut mask mode.

The ROI is a boolean mask with True inside the region of interest.
Outside the ROI, you should force OpenCV GrabCut labels to GC_BGD.

Methods:
  - "union_rect": bounding rectangle of (FG ∪ BG) scribbles, with margin.
  - "components_rect": per connected component rectangle on FG scribbles, unioned, with margin.
  - "union_hull": convex hull on (FG ∪ BG) scribbles, then filled, with margin approximated by dilating the hull mask.

Notes:
  [Unverified] Reasonable defaults are margin=10 pixels and a small dilation on scribbles before measuring boxes,
  this stabilizes the ROI for very thin strokes.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2 as cv

__all__ = ["build_tight_roi"]


def _clip_rect(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W - 1))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _rect_from_mask(m: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    # m is binary uint8
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, m.shape[1] - 1, m.shape[0] - 1  # fallback to full image
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 -= margin
    x1 += margin
    y0 -= margin
    y1 += margin
    return _clip_rect(x0, y0, x1, y1, m.shape[1], m.shape[0])


def build_tight_roi(
    seeds_fg: np.ndarray,
    seeds_bg: np.ndarray,
    margin: int = 10,
    method: str = "union_rect",
    dilate_px: int = 1,
    include_bg: bool = True,
    min_area_px: int = 9,
) -> np.ndarray:
    """
    Build a tight ROI mask from scribbles.

    Parameters
    ----------
    seeds_fg, seeds_bg : np.ndarray of bool or uint8, shape H by W
        Foreground and background scribble maps.
    margin : int
        Padding added around the tight region, in pixels.
    method : str
        "union_rect", "components_rect", or "union_hull".
    dilate_px : int
        Optional dilation radius applied to scribbles before measuring boxes, 0 disables.
    include_bg : bool
        Whether to include BG scribbles when computing the ROI, recommended True.
    min_area_px : int
        Components below this area are ignored in components_rect mode.

    Returns
    -------
    roi : np.ndarray of bool, shape H by W
        True inside ROI, False outside.
    """
    if seeds_fg.shape != seeds_bg.shape:
        raise ValueError("seeds_fg and seeds_bg must have the same shape")
    H, W = seeds_fg.shape[:2]

    fg = (seeds_fg.astype(np.uint8) > 0).astype(np.uint8)
    bg = (seeds_bg.astype(np.uint8) > 0).astype(np.uint8)
    all_scrib = fg | (bg if include_bg else 0)

    if dilate_px and dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        fg = cv.dilate(fg, kernel)
        all_scrib = cv.dilate(all_scrib, kernel)

    roi = np.zeros((H, W), dtype=bool)

    if method == "union_rect":
        x0, y0, x1, y1 = _rect_from_mask(all_scrib, int(margin))
        roi[y0:y1+1, x0:x1+1] = True
        return roi

    if method == "components_rect":
        num, labels = cv.connectedComponents(fg, connectivity=8)
        for lbl in range(1, int(num)):
            comp = (labels == lbl).astype(np.uint8)
            if comp.sum() < int(min_area_px):
                continue
            x0, y0, x1, y1 = _rect_from_mask(comp, int(margin))
            roi[y0:y1+1, x0:x1+1] = True
        if not roi.any():
            # fallback to union_rect to avoid empty ROI
            x0, y0, x1, y1 = _rect_from_mask(all_scrib, int(margin))
            roi[y0:y1+1, x0:x1+1] = True
        return roi

    if method == "union_hull":
        pts = cv.findNonZero(all_scrib)
        if pts is None or len(pts) < 3:
            x0, y0, x1, y1 = _rect_from_mask(all_scrib, int(margin))
            roi[y0:y1+1, x0:x1+1] = True
            return roi
        hull = cv.convexHull(pts)
        hull_mask = np.zeros((H, W), dtype=np.uint8)
        cv.fillConvexPoly(hull_mask, hull, 255)
        if margin and margin > 0:
            k = 2 * int(margin) + 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
            hull_mask = cv.dilate(hull_mask, kernel)
        roi = hull_mask > 0
        return roi

    # unknown method, fallback to union_rect
    x0, y0, x1, y1 = _rect_from_mask(all_scrib, int(margin))
    roi[y0:y1+1, x0:x1+1] = True
    return roi
