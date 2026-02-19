import numpy as np
import functools
from typing import Tuple

# ---------- CAM16 Vectorized Constants ----------

# Matrix M16 (Equation 18): Derived to solve computational failures of CIECAM02.
# This matrix transforms XYZ tristimulus values into the CAM16 cone response space.
_CAT16 = np.array([
    [ 0.401288,  0.650173, -0.051461],
    [-0.250268,  1.204414,  0.045854],
    [-0.002079,  0.048952,  0.953127],
], dtype=np.float32)

# Standard conversion matrix from sRGB to CIE XYZ (assuming D65 white point).
_SRGB_TO_XYZ = np.array([
    [0.412456, 0.357576, 0.180438],
    [0.212673, 0.715152, 0.072175],
    [0.019334, 0.119192, 0.950304],
], dtype=np.float32)

def _whitepoint_D65_XYZ() -> np.ndarray:
    """
    Step 0: Input Specification
    Establishes the Reference White (Xw, Yw, Zw). These coordinates for D65
    are required for the chromatic adaptation and lightness induction steps.
    """
    x, y = 0.31270, 0.32900
    Y = 100.0
    X, Z = Y * x / y, Y * (1.0 - x - y) / y
    return np.array([X, Y, Z], dtype=np.float32)

def _cam16_nonlinear_response(t: np.ndarray) -> np.ndarray:
    """
    Forward Model Step 4: Post-adaptation Non-linear Compression
    Applies the dynamic range compression (Equation 8).
    This simulates the hyperbolic response of human cone cells, using a 
    power law of 0.42 to model brightness perception.
    """
    t = t.astype(np.float32, copy=False)
    out = np.empty_like(t, dtype=np.float32)
    pos = t >= 0
    # Dynamic range compression for positive and negative signals
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
    Forward Model Step 1: Preliminary Setup & Viewing Conditions
    Calculates environmental induction factors based on Table 1 (Dim surround).
    Determines Degree of Adaptation (D) and Luminance Level Adaptation (FL).
    """
    XYZ_w = _whitepoint_D65_XYZ()
    # F, c, Nc constants for "Dim" surround (Appendix A, Section 2)
    F, c, Nc = 0.9, 0.59, 0.9
    E_w = 64.0  # Illuminance of reference white in lux
    L_w = E_w / np.pi
    Y_b = 20.0  # Background luminance factor
    L_A = (L_w * Y_b) / XYZ_w[1]  # Adapting field luminance

    # Step 1.1: Calculate Degree of Adaptation (D) (Equation 4)
    RGB_w = _CAT16 @ XYZ_w
    D = np.clip(F * (1.0 - (1.0 / 3.6) * np.exp(-(L_A + 42.0) / 92.0)), 0.0, 1.0)
    D_RGB = D * XYZ_w[1] / RGB_w + 1.0 - D

    # Step 1.2: Calculate Luminance Adaptation Factor (FL) (Equation 5)
    k = 1.0 / (5.0 * L_A + 1.0)
    F_L = 0.2 * k**4 * 5.0 * L_A + 0.1 * (1.0 - k**4)**2 * (5.0 * L_A)**(1.0 / 3.0)

    # Step 1.3: Calculate induction factors (Equations 6 & 7)
    n = Y_b / XYZ_w[1]
    z = 1.48 + n**0.5
    N_bb = N_cb = 0.725 * (1.0 / n)**0.2

    # Step 1.4: Calculate adapted achromatic white (Aw)
    RGB_wc = D_RGB * RGB_w
    t = (F_L * RGB_wc / 100.0)
    RGB_aw = _cam16_nonlinear_response(t)
    A_w = (np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) @ RGB_aw - 0.305) * N_bb

    return {
        "F_L": float(F_L), "c": float(c), "Nc": float(Nc), "n": float(n), "z": float(z),
        "N_bb": float(N_bb), "N_cb": float(N_cb), "A_w": float(A_w),
        "D_RGB": D_RGB.astype(np.float32),
    }

def _cam16_forward_JMh_from_rgb(img_rgb_u8: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Forward Model Implementation
    Translates RGB image data into perceptual correlates J (Lightness), 
    M (Colorfulness), and h (Hue angle).
    """
    ctx = _camSetup = _cam16_setup()
    
    # Pre-step: Linearize sRGB and convert to XYZ Tristimulus Values
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    H, W, _ = rgb_lin.shape
    XYZ = (rgb_lin.reshape(-1, 3) @ _SRGB_TO_XYZ.T) * 100.0

        # Step 2: Chromatic Adaptation (CAT16) (Equation 3)
        # Map XYZ to cone space and apply the adaptation factor D.
    RGB = (XYZ @ _CAT16.T) * ctx["D_RGB"]

    # Step 3 & 4: Nonlinear Response Compression (Equation 8)
    t = (ctx["F_L"] * RGB / 100.0)
    RGB_a = _cam16_nonlinear_response(t)

    # Step 5: Compute Hue Angle (h) (Equations 9 & 10)
    # Calculate preliminary Cartesian components (a, b)
    a_comp = RGB_a @ np.array([1.0, -12.0 / 11.0, 1.0 / 11.0], dtype=np.float32)
    b_comp = RGB_a @ np.array([1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0], dtype=np.float32)
    h = np.degrees(np.arctan2(b_comp, a_comp)).astype(np.float32)
    h[h < 0.0] += 360.0
    h_rad = np.radians(h)

    # Step 6: Compute Lightness (J) (Equations 11-13)
    # Calculate Achromatic signal (A) and normalize against White Point (Aw).
    e = 0.25 * (np.cos(h_rad + 2.0) + 3.8)
    A = (RGB_a @ np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) - 0.305) * ctx["N_bb"]
    J = 100.0 * np.power(np.maximum(A, 0.0) / np.maximum(ctx["A_w"], 1e-6), ctx["c"] * ctx["z"])

    # Step 7: Compute Colorfulness (M) (Equations 14-16)
    p1 = (50000.0 / 13.0) * ctx["Nc"] * ctx["N_cb"] * e * np.sqrt(a_comp**2 + b_comp**2)
    p2 = RGB_a @ np.array([1.0, 1.0, 21.0 / 20.0], dtype=np.float32)
    p2 = np.where(np.abs(p2) < 1e-6, 1e-6, p2)
    C = np.power(p1 / p2, 0.9) * np.sqrt(J / 100.0) * np.power(1.64 - 0.29 ** ctx["n"], 0.73)
    M = C * (ctx["F_L"] ** 0.25)

    return H, W, J.astype(np.float32), M.astype(np.float32), h.astype(np.float32)

def _cam16_scd_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    CAM16-SCD (Small Color Difference) Transformation (Appendix B)
    
    Deviates from the Forward Model to create a Uniform Color Space (UCS)
    specifically tuned for small perceptual differences.
    
    Step 8: Apply SCD-Specific Constants (c1=0.007, c2=0.0363)
    Step 9: Cartesian Projection (J', a', b') for Euclidean distance parity.
    """
    H, W, J, M, h = _cam16_forward_JMh_from_rgb(img_rgb_u8)
    c1, c2 = 0.007, 0.0363 # SCD specific chroma compression
    
    # Equation A3: Non-linear compression for Uniformity
    Jp = ((1.0 + 100.0 * c1) * J) / (1.0 + c1 * J)
    Mp = np.log1p(c2 * M) / c2
    
    h_rad = np.radians(h)
    ap, bp = Mp * np.cos(h_rad), Mp * np.sin(h_rad)
    
    jab = np.stack([Jp, ap, bp], axis=1).reshape(H, W, 3).astype(np.float32)
    return _scale_to_uint8_per_channel(jab)