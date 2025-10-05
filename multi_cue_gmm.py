# -*- coding: utf-8 -*-
"""
Adaptive, Multi-Cue GMMs for GrabCut seed refinement.

This module implements two cues:
1) Appearance GMM on the provided 3 channel features.
2) Structure GMM on a steerable-like Gabor filter bank magnitude.

It exports:
    mc_refine_seeds(img_feats_u8, seeds_bg, seeds_fg, ...)
which returns refined firm seeds for use with cv2.grabCut.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2 as cv


# -----------------------------
# Steerable-ish filter bank using paired Gabor kernels
# -----------------------------

def _gabor_bank(gray_f32: np.ndarray,
                num_orient: int = 8,
                num_scales: int = 3,
                ksize0: int = 9,
                sigma0: float = 2.5,
                lambda0: float = 6.0,
                gamma: float = 0.5) -> np.ndarray:
    """
    Build magnitude responses of a steerable-like Gabor filter bank.
    Returns HxWx(num_orient*num_scales) float32 array in [0, 1] after per-channel normalization.
    """
    H, W = gray_f32.shape
    gray = gray_f32.astype(np.float32, copy=False)
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        gray = (gray - gmin) / (gmax - gmin)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    feats = []
    for si in range(num_scales):
        # scale parameters
        scale = 2.0 ** si
        ksize = int(round(ksize0 * scale))
        if ksize % 2 == 0:
            ksize += 1
        sigma = sigma0 * scale
        lambd = lambda0 * scale

        for oi in range(num_orient):
            theta = np.pi * oi / num_orient
            # even and odd pairs via phase 0 and pi/2
            k_even = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv.CV_32F)
            k_odd  = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2.0, ktype=cv.CV_32F)

            r_even = cv.filter2D(gray, cv.CV_32F, k_even, borderType=cv.BORDER_REFLECT101)
            r_odd  = cv.filter2D(gray, cv.CV_32F, k_odd,  borderType=cv.BORDER_REFLECT101)
            mag = np.sqrt(r_even * r_even + r_odd * r_odd)

            # robust per-channel normalization to [0, 1] using percentiles
            p1, p99 = np.percentile(mag, [1, 99])
            if p99 > p1:
                mag = np.clip((mag - p1) / (p99 - p1), 0.0, 1.0)
            else:
                mag = np.zeros_like(mag, dtype=np.float32)
            feats.append(mag.astype(np.float32))

    bank = np.stack(feats, axis=2) if feats else np.zeros((H, W, 1), np.float32)
    return bank


# -----------------------------
# Lightweight GMM wrappers
# -----------------------------

class _EMGMM:
    """
    Thin wrapper around OpenCV EM for GMM likelihoods.
    Falls back to single Gaussian if cv.ml.EM is unavailable.
    """
    def __init__(self, n_components: int = 5, covariance_diag: bool = True):
        self.n_components = int(n_components)
        self.cov_diag = bool(covariance_diag)
        self._em = None
        self._mu = None
        self._cov = None

        if hasattr(cv.ml, "EM_create"):
            em = cv.ml.EM_create()
            em.setClustersNumber(self.n_components)
            em.setCovarianceMatrixType(cv.ml.EM_COV_MAT_DIAGONAL if self.cov_diag else cv.ml.EM_COV_MAT_GENERIC)
            em.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3))
            self._em = em

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < max(self.n_components, 2):
            # degenerate, switch to single Gaussian
            self._em = None

        if self._em is not None:
            try:
                self._em.trainEM(X)
                return
            except cv.error:
                self._em = None  # fall through to single Gaussian

        # single Gaussian fallback
        mu = X.mean(axis=0, keepdims=True)
        # rowvar False gives DxD
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov).astype(np.float32)
        # add jitter for stability
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * 1e-6
        self._mu = mu.astype(np.float32)
        self._cov = cov

    def predict_loglik(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self._em is not None:
            ll, _ = self._em.predict2(X)  # ll shape [N,1]
            return ll.reshape(-1).astype(np.float32)

        # single Gaussian log likelihood
        mu = self._mu.reshape(1, -1)
        cov = self._cov
        D = X.shape[1]
        try:
            inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
            eig = np.linalg.eigvalsh((cov + cov.T) * 0.5)
            eig = np.clip(eig, 1e-12, None)
            logdet = float(np.log(eig).sum())
        diff = X - mu
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        ll = -0.5 * (D * np.log(2.0 * np.pi) + logdet + maha)
        return ll.astype(np.float32)

    def predict_peakedness(self, X: np.ndarray) -> np.ndarray:
        """
        Returns per sample confidence in [0,1], approximated as the max posterior across components.
        For the single Gaussian fallback, returns 0.5.
        """
        X = np.asarray(X, dtype=np.float32)
        if self._em is not None:
            _, post = self._em.predict2(X)  # [N,K]
            if post is None:
                return np.full((X.shape[0],), 0.5, dtype=np.float32)
            return post.max(axis=1).astype(np.float32)
        return np.full((X.shape[0],), 0.5, dtype=np.float32)


# -----------------------------
# Adaptive, Multi-Cue refinement
# -----------------------------

def _percentile_normalize(x: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def mc_refine_seeds(img_feats_u8: np.ndarray,
                    seeds_bg: np.ndarray,
                    seeds_fg: np.ndarray,
                    *,
                    K_app: int = 5,
                    K_str: int = 5,
                    gabor_orient: int = 8,
                    gabor_scales: int = 3,
                    near_iters_base: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive multi-cue GMM refinement for firm seeds.
    Appearance cue uses the provided 3 channel features.
    Structure cue uses a steerable-like Gabor bank on image intensity.
    Returns (seeds_fg_refined, seeds_bg_refined) boolean arrays.
    """
    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 features, got {img_feats_u8.shape}")
    H, W, _ = img_feats_u8.shape
    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError("Seed masks must match image size")

    if (not np.any(seeds_fg)) or (not np.any(seeds_bg)):
        return seeds_fg, seeds_bg

    # Appearance features in float
    X_app = img_feats_u8.reshape(-1, 3).astype(np.float32) / 255.0

    # Structure features from intensity and Gabor bank
    gray = img_feats_u8.mean(axis=2).astype(np.float32)
    bank = _gabor_bank(gray, num_orient=gabor_orient, num_scales=gabor_scales)  # HxWxD
    D = bank.shape[2]
    X_str = bank.reshape(-1, D).astype(np.float32)

    # Fit GMMs for FG and BG per cue on seed pixels (subsample if very large)
    fg_idx = np.flatnonzero(seeds_fg.ravel())
    bg_idx = np.flatnonzero(seeds_bg.ravel())

    def _sample_idx(idx: np.ndarray, cap: int = 50000) -> np.ndarray:
        if idx.size <= cap:
            return idx
        rs = np.random.RandomState(123)
        sel = rs.choice(idx, size=cap, replace=False)
        return np.sort(sel)

    fg_s = _sample_idx(fg_idx)
    bg_s = _sample_idx(bg_idx)

    # Appearance GMMs
    gmm_fg_app = _EMGMM(n_components=K_app, covariance_diag=True)
    gmm_bg_app = _EMGMM(n_components=K_app, covariance_diag=True)
    gmm_fg_app.fit(X_app[fg_s])
    gmm_bg_app.fit(X_app[bg_s])

    # Structure GMMs
    gmm_fg_str = _EMGMM(n_components=K_str, covariance_diag=True)
    gmm_bg_str = _EMGMM(n_components=K_str, covariance_diag=True)
    gmm_fg_str.fit(X_str[fg_s])
    gmm_bg_str.fit(X_str[bg_s])

    # Log likelihoods and cue-specific log odds
    ll_fg_app = gmm_fg_app.predict_loglik(X_app)
    ll_bg_app = gmm_bg_app.predict_loglik(X_app)
    s_app = (ll_fg_app - ll_bg_app).reshape(H, W).astype(np.float32)

    ll_fg_str = gmm_fg_str.predict_loglik(X_str)
    ll_bg_str = gmm_bg_str.predict_loglik(X_str)
    s_str = (ll_fg_str - ll_bg_str).reshape(H, W).astype(np.float32)

    # Cue confidences per pixel
    # Appearance confidence from posterior peakedness and |s_app|
    peak_fg_app = gmm_fg_app.predict_peakedness(X_app)
    peak_bg_app = gmm_bg_app.predict_peakedness(X_app)
    peak_app = np.maximum(peak_fg_app, peak_bg_app).reshape(H, W).astype(np.float32)
    c_app = 0.5 * _percentile_normalize(np.abs(s_app)) + 0.5 * peak_app
    c_app = np.clip(c_app, 0.0, 1.0)

    # Structure confidence from local energy and |s_str| and peakedness
    energy = bank.mean(axis=2)  # HxW in [0,1]
    energy = cv.GaussianBlur(energy, (0, 0), 1.0).astype(np.float32)
    peak_fg_str = gmm_fg_str.predict_peakedness(X_str).reshape(H, W).astype(np.float32)
    peak_bg_str = gmm_bg_str.predict_peakedness(X_str).reshape(H, W).astype(np.float32)
    peak_str = np.maximum(peak_fg_str, peak_bg_str).astype(np.float32)
    c_str = 0.4 * _percentile_normalize(np.abs(s_str)) + 0.4 * energy + 0.2 * peak_str
    c_str = np.clip(c_str, 0.0, 1.0)

    # Adaptive weights normalized to sum 1
    denom = c_app + c_str + 1e-6
    w_app = c_app / denom
    w_str = c_str / denom

    # Fused score
    s_fused = (w_app * s_app + w_str * s_str).astype(np.float32)

    # Robust normalization for thresholding
    s_norm = _percentile_normalize(s_fused, 10.0, 90.0)

    # Expand seeds near scribbles with adaptive thresholds
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    near_iter = int(max(1, near_iters_base))
    near_fg = cv.dilate(seeds_fg.astype(np.uint8), k, iterations=near_iter).astype(bool)
    near_bg = cv.dilate(seeds_bg.astype(np.uint8), k, iterations=near_iter).astype(bool)

    # thresholds derived from seed neighborhoods
    base_hi = float(np.percentile(s_norm[near_fg], 70)) if np.any(near_fg) else 0.7
    base_lo = float(np.percentile(s_norm[near_bg], 30)) if np.any(near_bg) else 0.3

    # Adjust thresholds using global dominance of the structure cue
    dom = float((c_str > c_app).mean()) if s_norm.size else 0.5
    hi = float(np.clip(base_hi - 0.10 * (1.0 - dom), 0.50, 0.95))
    lo = float(np.clip(base_lo + 0.10 * dom, 0.05, 0.50))

    cand_fg = (s_norm >= hi) & (~seeds_bg)
    cand_bg = (s_norm <= lo) & (~seeds_fg)

    add_fg = cand_fg & near_fg
    add_bg = cand_bg & near_bg

    # Resolve conflicts, keep original scribbles firm
    add_fg[seeds_bg] = False
    add_bg[seeds_fg] = False
    both = add_fg & add_bg
    add_fg[both] = False
    add_bg[both] = False

    seeds_fg_ref = seeds_fg | add_fg
    seeds_bg_ref = seeds_bg | add_bg

    return seeds_fg_ref, seeds_bg_ref
