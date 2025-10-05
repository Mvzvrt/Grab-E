# -*- coding: utf-8 -*-
"""
Adaptive, Multi-Cue GMMs for GrabCut seed refinement, optimized.

Key optimizations, behavior preserved:
1) Cache the Gabor structure bank per image in-process.
2) Evaluate GMM scores on pixels near the scribbles only, then scatter.
3) Use float32 throughout to reduce memory traffic.
4) Smaller filter bank by default (6 orientations, 2 scales).
5) Subsample for percentile estimates and EM training cap.
6) Fewer EM iterations.
7) Vectorized diagonal-covariance mixture evaluation, no EM.predict2 usage.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import cv2 as cv


# -----------------------------
# Helpers
# -----------------------------

def _percentile_normalize(x: np.ndarray,
                          p_lo: float = 5.0,
                          p_hi: float = 95.0,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize to [0,1] using percentiles, optionally over a mask only."""
    if mask is not None:
        vals = x[mask].astype(np.float32)
    else:
        vals = x.astype(np.float32).ravel()
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _gray_sig(gray: np.ndarray) -> Tuple[int, int, float, float]:
    """Cheap signature for caching, based on shape and coarse stats on a subsample."""
    g = gray.astype(np.float32, copy=False)
    h, w = g.shape
    total = h * w
    step = max(1, int(total // max(1, int(total * 0.01))))  # about 1 percent
    s = g.ravel()[::step]
    return (h, w, float(s.mean()), float(s.std()))


# -----------------------------
# Steerable-ish filter bank using paired Gabor kernels
# -----------------------------

def _gabor_bank(gray_f32: np.ndarray,
                num_orient: int = 6,
                num_scales: int = 2,
                ksize0: int = 9,
                sigma0: float = 2.0,
                lambda0: float = 5.0,
                gamma: float = 0.5) -> np.ndarray:
    """
    Build magnitude responses of a paired Gabor filter bank.
    Returns HxWx(num_orient*num_scales) float32 array in [0, 1] after per-channel normalization.
    Uses subsampled percentiles for normalization.
    """
    H, W = gray_f32.shape
    gray = gray_f32.astype(np.float32, copy=False)
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        gray = (gray - gmin) / (gmax - gmin)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    feats: List[np.ndarray] = []

    # Subsample indices for percentile estimation
    total = H * W
    take = max(2048, int(total * 0.02))
    rs = np.random.RandomState(123)
    idx = rs.choice(total, size=min(take, total), replace=False)

    for si in range(num_scales):
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
            mag = np.sqrt(r_even * r_even + r_odd * r_odd).astype(np.float32)

            # robust per-channel normalization to [0, 1] using subsampled percentiles
            svals = mag.ravel()[idx]
            p1, p99 = np.percentile(svals, [1, 99])
            if p99 > p1:
                mag = np.clip((mag - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)
            else:
                mag = np.zeros_like(mag, dtype=np.float32)
            feats.append(mag)

    bank = np.stack(feats, axis=2) if feats else np.zeros((H, W, 1), np.float32)
    return bank


# Very small LRU cache for structure banks
_GABOR_CACHE: dict = {}

def _get_structure_bank(gray: np.ndarray) -> np.ndarray:
    sig = _gray_sig(gray)
    hit = _GABOR_CACHE.get(sig)
    if hit is not None:
        return hit
    bank = _gabor_bank(gray.astype(np.float32))
    if len(_GABOR_CACHE) >= 4:
        # drop oldest
        _GABOR_CACHE.pop(next(iter(_GABOR_CACHE)))
    _GABOR_CACHE[sig] = bank
    return bank


# -----------------------------
# Lightweight GMM wrappers
# -----------------------------

def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class _EMGMM:
    """
    Wrapper around OpenCV EM that avoids predict2.
    After training, read weights, means, covariances, then evaluate mixture
    log-likelihoods and posteriors explicitly in NumPy.
    Falls back to a single Gaussian only if EM training fails entirely.
    """
    def __init__(self, n_components: int = 5, covariance_diag: bool = True):
        self.n_components_req = int(n_components)
        self.cov_diag = bool(covariance_diag)

        # learned params
        self._w: Optional[np.ndarray] = None      # [K]
        self._mu: Optional[np.ndarray] = None     # [K,D]
        self._cov: Optional[List[np.ndarray]] = None  # list of [D,D]
        self._var: Optional[np.ndarray] = None    # [K,D] diagonals for diag case
        self._dim: Optional[int] = None

        # single Gaussian fallback params
        self._sg_mu: Optional[np.ndarray] = None  # [1,D]
        self._sg_cov: Optional[np.ndarray] = None # [D,D]

        # EM handle
        self._em = None
        if hasattr(cv.ml, "EM_create"):
            em = cv.ml.EM_create()
            em.setCovarianceMatrixType(cv.ml.EM_COV_MAT_DIAGONAL if self.cov_diag else cv.ml.EM_COV_MAT_GENERIC)
            em.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            self._em = em

    def _single_gaussian_fit(self, X32: np.ndarray) -> None:
        mu = X32.mean(axis=0, keepdims=True)             # [1,D]
        cov = np.cov(X32, rowvar=False)
        cov = np.atleast_2d(cov).astype(np.float32)
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * 1e-9
        self._sg_mu, self._sg_cov = mu.astype(np.float32), cov
        self._w = None
        self._mu = None
        self._cov = None
        self._var = None
        self._dim = X32.shape[1]

    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < 3:
            self._single_gaussian_fit(X)
            return

        if self._em is None:
            self._single_gaussian_fit(X)
            return

        # Auto-reduce K to available unique samples, check at most 20k
        uniq = np.unique(X[: min(20000, X.shape[0])], axis=0)
        K = max(1, min(self.n_components_req, uniq.shape[0]))
        self._em.setClustersNumber(int(K))

        try:
            ok = self._em.trainEM(X)
        except cv.error:
            ok = False

        if not ok:
            self._single_gaussian_fit(X)
            return

        means = self._em.getMeans()
        covs = self._em.getCovs()
        w = self._em.getWeights()
        if means is None or covs is None or w is None:
            self._single_gaussian_fit(X)
            return
        if means.shape[1] != X.shape[1]:
            self._single_gaussian_fit(X)
            return

        # Store learned params as float32
        self._w = np.asarray(w, dtype=np.float32).reshape(-1)    # [K]
        self._mu = np.asarray(means, dtype=np.float32)           # [K,D]
        self._cov = [np.asarray(c, dtype=np.float32) for c in covs]  # K*[D,D]
        self._var = None
        if self.cov_diag:
            self._var = np.stack([np.diag(c).astype(np.float32) for c in self._cov], axis=0)  # [K,D]
        self._dim = int(self._mu.shape[1])

    def _mix_loglik_and_post(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per sample mixture log-likelihood and posteriors.
        Returns (ll[N], post[N,K]).
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        N, D = X.shape

        # Single Gaussian path
        if self._w is None or self._mu is None or self._cov is None:
            mu = self._sg_mu.astype(np.float32)  # [1,D]
            cov = self._sg_cov.astype(np.float32)
            try:
                inv = np.linalg.inv(cov)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
                eig = np.linalg.eigvalsh((cov + cov.T) * 0.5)
                eig = np.clip(eig, 1e-12, None).astype(np.float32)
                logdet = float(np.log(eig).sum())
            diff = X - mu
            maha = np.einsum("ni,ij,nj->n", diff, inv, diff, optimize=True)
            ll = -0.5 * (D * np.log(2.0 * np.pi) + logdet + maha).astype(np.float32)
            post = np.ones((N, 1), dtype=np.float32)
            return ll, post

        # Mixture case
        K = int(self._w.shape[0])
        log_prob = np.empty((N, K), dtype=np.float32)

        if self.cov_diag and self._var is not None:
            # Vectorized diagonal evaluation
            # X: [N,D], mu: [K,D], var: [K,D]
            diff = X[:, None, :] - self._mu[None, :, :]         # [N,K,D]
            var = np.clip(self._var, 1e-9, None)                # [K,D]
            maha = (diff * diff / var[None, :, :]).sum(axis=2)  # [N,K]
            logdet = np.log(var).sum(axis=1)                    # [K]
            log_prob = (np.log(self._w)[None, :]
                        - 0.5 * (D * np.log(2.0 * np.pi) + logdet[None, :] + maha)).astype(np.float32)
        else:
            # Full covariance, loop per component
            for k in range(K):
                mu = self._mu[k]
                cov = self._cov[k]
                cov = (cov + cov.T) * 0.5
                try:
                    inv = np.linalg.inv(cov)
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0:
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(cov)
                    eig = np.linalg.eigvalsh(cov)
                    eig = np.clip(eig, 1e-12, None).astype(np.float32)
                    logdet = float(np.log(eig).sum())
                diff = X - mu
                maha = np.einsum("ni,ij,nj->n", diff, inv, diff, optimize=True)
                log_prob[:, k] = (np.log(self._w[k])
                                  - 0.5 * (D * np.log(2.0 * np.pi) + logdet + maha)).astype(np.float32)

        ll = _logsumexp(log_prob, axis=1).astype(np.float32)           # [N]
        post = np.exp(log_prob - ll[:, None]).astype(np.float32)       # [N,K], sums to 1
        return ll, post

    def predict_loglik(self, X: np.ndarray) -> np.ndarray:
        ll, _ = self._mix_loglik_and_post(X)
        return ll.astype(np.float32)

    def predict_peakedness(self, X: np.ndarray) -> np.ndarray:
        _, post = self._mix_loglik_and_post(X)
        return post.max(axis=1).astype(np.float32)


# -----------------------------
# Adaptive, Multi-Cue refinement
# -----------------------------

def mc_refine_seeds(img_feats_u8: np.ndarray,
                    seeds_bg: np.ndarray,
                    seeds_fg: np.ndarray,
                    *,
                    K_app: int = 5,
                    K_str: int = 5,
                    gabor_orient: int = 6,
                    gabor_scales: int = 2,
                    near_iters_base: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive multi-cue GMM refinement for firm seeds.
    Appearance cue uses the provided 3 channel features.
    Structure cue uses a paired Gabor bank on image intensity.
    Returns (seeds_fg_refined, seeds_bg_refined) boolean arrays.
    """
    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 features, got {img_feats_u8.shape}")
    H, W, _ = img_feats_u8.shape
    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError("Seed masks must match image size")

    if (not np.any(seeds_fg)) or (not np.any(seeds_bg)):
        return seeds_fg, seeds_bg

    # Neighborhoods first, because we will score only near the scribbles
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    near_iter = int(max(1, near_iters_base))
    near_fg = cv.dilate(seeds_fg.astype(np.uint8), k, iterations=near_iter).astype(bool)
    near_bg = cv.dilate(seeds_bg.astype(np.uint8), k, iterations=near_iter).astype(bool)
    near_any_mask = (near_fg | near_bg)
    if not np.any(near_any_mask):
        return seeds_fg, seeds_bg

    near_any = near_any_mask.ravel()

    # Appearance features, only near pixels
    X_app_all = (img_feats_u8.reshape(-1, 3).astype(np.float32) / 255.0)
    X_app = X_app_all[near_any]

    # Structure features with cache, only near pixels
    gray = img_feats_u8.mean(axis=2).astype(np.float32)
    bank = _get_structure_bank(gray)  # H x W x D
    D = int(bank.shape[2])
    X_str_all = bank.reshape(-1, D).astype(np.float32)
    X_str = X_str_all[near_any]

    # Training indices within the near subset
    fg_near = seeds_fg.ravel()[near_any]
    bg_near = seeds_bg.ravel()[near_any]
    fg_idx = np.flatnonzero(fg_near)
    bg_idx = np.flatnonzero(bg_near)

    # Subsample for EM training
    def _sample_idx(idx: np.ndarray, cap: int = 20000) -> np.ndarray:
        if idx.size <= cap:
            return idx
        rs = np.random.RandomState(123)
        sel = rs.choice(idx, size=cap, replace=False)
        return np.sort(sel)

    fg_s = _sample_idx(fg_idx)
    bg_s = _sample_idx(bg_idx)

    # Train GMMs on near samples only
    gmm_fg_app = _EMGMM(n_components=K_app, covariance_diag=True)
    gmm_bg_app = _EMGMM(n_components=K_app, covariance_diag=True)
    gmm_fg_app.fit(X_app[fg_s])
    gmm_bg_app.fit(X_app[bg_s])

    gmm_fg_str = _EMGMM(n_components=K_str, covariance_diag=True)
    gmm_bg_str = _EMGMM(n_components=K_str, covariance_diag=True)
    gmm_fg_str.fit(X_str[fg_s])
    gmm_bg_str.fit(X_str[bg_s])

    # Evaluate log odds only on near
    ll_fg_app = gmm_fg_app.predict_loglik(X_app)
    ll_bg_app = gmm_bg_app.predict_loglik(X_app)
    s_app_near = (ll_fg_app - ll_bg_app)  # [N_near]

    ll_fg_str = gmm_fg_str.predict_loglik(X_str)
    ll_bg_str = gmm_bg_str.predict_loglik(X_str)
    s_str_near = (ll_fg_str - ll_bg_str)  # [N_near]

    # Scatter back to full image buffers
    s_app = np.zeros((H * W,), dtype=np.float32)
    s_str = np.zeros((H * W,), dtype=np.float32)
    s_app[near_any] = s_app_near
    s_str[near_any] = s_str_near
    s_app = s_app.reshape(H, W)
    s_str = s_str.reshape(H, W)

    # Cue confidences, compute only on near then scatter
    peak_app_near = np.maximum(
        gmm_fg_app.predict_peakedness(X_app),
        gmm_bg_app.predict_peakedness(X_app)
    ).astype(np.float32)

    peak_str_near = np.maximum(
        gmm_fg_str.predict_peakedness(X_str),
        gmm_bg_str.predict_peakedness(X_str)
    ).astype(np.float32)

    c_app = np.zeros((H * W,), dtype=np.float32)
    c_str = np.zeros((H * W,), dtype=np.float32)
    # Normalize absolute scores based on near pixels only
    abs_s_app = np.zeros((H * W,), dtype=np.float32); abs_s_app[near_any] = np.abs(s_app_near)
    abs_s_str = np.zeros((H * W,), dtype=np.float32); abs_s_str[near_any] = np.abs(s_str_near)
    norm_abs_app = _percentile_normalize(abs_s_app.reshape(H, W), mask=near_any_mask)
    norm_abs_str = _percentile_normalize(abs_s_str.reshape(H, W), mask=near_any_mask)

    c_app[near_any] = 0.5 * norm_abs_app.ravel()[near_any] + 0.5 * peak_app_near
    # Structure energy from bank, full image then blur, but normalize by near mask
    energy = bank.mean(axis=2).astype(np.float32)  # HxW in [0,1]
    energy = cv.GaussianBlur(energy, (0, 0), 1.0).astype(np.float32)
    energy_n = _percentile_normalize(energy, mask=near_any_mask)

    c_str[near_any] = 0.4 * norm_abs_str.ravel()[near_any] + 0.4 * energy_n.ravel()[near_any] + 0.2 * peak_str_near

    c_app = c_app.reshape(H, W)
    c_str = c_str.reshape(H, W)

    # Adaptive weights normalized to sum 1
    denom = c_app + c_str + 1e-6
    w_app = c_app / denom
    w_str = c_str / denom

    # Fuse, compute normalized scores using near pixels only
    s_fused = (w_app * s_app + w_str * s_str).astype(np.float32)
    s_norm = _percentile_normalize(s_fused, 10.0, 90.0, mask=near_any_mask)

    # thresholds derived from seed neighborhoods
    base_hi = float(np.percentile(s_norm[near_fg], 70)) if np.any(near_fg) else 0.7
    base_lo = float(np.percentile(s_norm[near_bg], 30)) if np.any(near_bg) else 0.3

    # Adjust thresholds using global dominance of the structure cue over near region
    dom = float((c_str[near_any_mask] > c_app[near_any_mask]).mean()) if np.any(near_any_mask) else 0.5
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
