"""
modified_grabcut_pre.py

One function export for the paper's change: apply a median filter, then K-means color quantization,
then pass the resulting 3-channel uint8 image to OpenCV GrabCut.

Verified idea from the paper: "the image is smoothed using median filter and the quantized image using
K-means algorithm is used for the normal GrabCut method" [see accompanying citations in the chat].

Notes:
  [Unverified] Default hyperparameters (k=8, median_ksize=3, max_iter=10) are practical starting points,
  the paper does not mandate exact values. Please tune per dataset.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2 as cv

__all__ = ["pre_grabcut_median_kmeans"]


def pre_grabcut_median_kmeans(
    img_u8: np.ndarray,
    k: int = 8,
    median_ksize: int = 3,
    max_iter: int = 10,
    attempts: int = 1,
    sample_stride: int = 1,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply median filtering then K-means quantization to a 3-channel image.

    Parameters
    ----------
    img_u8 : np.ndarray
        H by W by 3, dtype uint8, any 3-channel feature space is acceptable.
    k : int, default 8
        Number of clusters for K-means. [Unverified] pick based on image complexity.
    median_ksize : int, default 3
        Median filter window size, must be odd and >= 1. Use 1 to disable.
    max_iter : int, default 10
        Maximum K-means iterations.
    attempts : int, default 1
        Number of K-means restarts.
    sample_stride : int, default 1
        Subsample pixels for faster K-means, 1 means use all pixels.
    rng_seed : Optional[int], default None
        Seed for K-means initialization.

    Returns
    -------
    np.ndarray
        Quantized image, H by W by 3, dtype uint8.

    Raises
    ------
    ValueError
        If the input is not H by W by 3 or dtype cannot be converted to uint8.
    """
    if img_u8 is None or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {getattr(img_u8, 'shape', None)}")

    # Ensure uint8
    if img_u8.dtype != np.uint8:
        img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

    # Median filter per paper
    if median_ksize and median_ksize > 1:
        if median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd")
        smoothed = cv.medianBlur(img_u8, int(median_ksize))
    else:
        smoothed = img_u8

    H, W, C = smoothed.shape
    assert C == 3

    # Prepare data for K-means
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    data = smoothed.reshape(-1, 3).astype(np.float32)
    if sample_stride > 1:
        # Subsample for speed, then still assign labels for all pixels
        grid = np.arange(0, data.shape[0], sample_stride, dtype=np.int32)
        data_sample = data[grid]
    else:
        data_sample = data

    # K-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, int(max_iter), 1e-4)
    flags = cv.KMEANS_PP_CENTERS
    if rng_seed is not None:
        cv.setRNGSeed(int(rng_seed))

    # Run on the sampled set to get centers
    compactness, labels_sample, centers = cv.kmeans(
        data_sample, int(k), None, criteria, int(attempts), flags
    )
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)

    # Assign all pixels to nearest center (including those not in the sample)
    # Use squared Euclidean distance
    # centers_u8: k by 3, data: N by 3
    # Compute distances in a vectorized way: for each center, squared norm of (data - center)
    data_f = data  # already float32
    centers_f = centers.astype(np.float32)
    # Compute squared distances: N by K
    # d(x, c) = ||x||^2 - 2 x·c + ||c||^2
    x2 = np.sum(data_f * data_f, axis=1, keepdims=True)           # N by 1
    c2 = np.sum(centers_f * centers_f, axis=1, keepdims=True).T   # 1 by K
    xc = data_f @ centers_f.T                                     # N by K
    d2 = x2 - 2.0 * xc + c2                                       # N by K
    labels_all = np.argmin(d2, axis=1).astype(np.int32)           # N

    quantized = centers_u8[labels_all].reshape(H, W, 3)
    return quantized
