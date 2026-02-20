# Filename: color_space.py
# -*- coding: utf-8 -*-
"""
Color space converters and helpers for GrabCut features.

This module centralizes all color-space conversions and exposes:
- get_color_converter(mode): returns a callable converter
- convert_color_space(img_rgb_u8, mode): returns HxWx3 uint8 features

Supported keys match those used previously in grabcut.py:
  rgb, hsv_conic, cielab, c02_scd, c16_scd,
  oklab, oklch, jzazbz, jzczhz, ictcp_pq, xyz, ycbcr_bt709, srgb_linear,
  ruderman_lab, lalphabeta
"""
from __future__ import annotations
from typing import Optional, Callable, Tuple
import functools
import numpy as np
import cv2 as cv


def _ensure_hwc3(arr: np.ndarray, H: int, W: int, where: str = "converter") -> np.ndarray:
    """
    Ensures the array conforms to the standard (H, W, 3) image topology.
    
    This function enforces a 'Structural Contract' by detecting and correcting 
    common memory layout variations produced by different scientific libraries:
    
    1. HWC Pass-through: If already (H, W, 3), returns the array immediately.
    2. Flattened Reconstruction: Detects 'Bag of Pixels' format (H*W, 3) common in 
       color science engines (e.g., colorspacious) and reshapes to a spatial grid.
    3. Planar Transposition: Detects 'CHW' format (3, H, W) standard in deep 
       learning frameworks (e.g., PyTorch) and transposes it back to interleaved HWC.
    4. C-contiguous Buffer: Recovers data from raw 1D streams (H*W*3).
    
    Reference:
    - Documentation for OpenCV (cv2) and NumPy-based vision pipelines requires 
      interleaved (HWC) memory layout for efficient spatial neighborhood operations 
      and pixel-wise access.
    """
    a = np.asarray(arr)
    # Case 1: Standard HWC
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    # Case 2: Flattened List (Spatial recovery)
    if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == H * W:
        return a.reshape(H, W, 3)
    # Case 3: Planar/CHW (Standard in Deep Learning)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[1] == H and a.shape[2] == W:
        return a.transpose(1, 2, 0)
    # Case 4: Raw 1D Buffer
    if a.ndim == 1 and a.size == H * W * 3:
        return a.reshape(H, W, 3)
    
    raise ValueError(f"{where} produced array with shape {a.shape}, expected {(H, W, 3)}")


def _scale_to_uint8_per_channel(x: np.ndarray) -> np.ndarray:
    """
    Normalizes floating-point color correlates into an 8-bit Euclidean feature space.
    
    This function performs a dynamic Min-Max scaling per channel to map perceptual 
    dimensions (e.g., J', a', b') into the [0, 255] integer range. 
    
    Theoretical Basis:
    1. Feature Parity: Following the 'Whitening' principle (Shapiro & Stockman, 2001), 
       independent scaling ensures that channels with larger raw ranges (like Lightness) 
       do not disproportionately bias the Euclidean distance calculations used in 
       segmentation algorithms (e.g., GrabCut, K-Means).
    2. Quantization Strategy: Mirrors the integer-mapping logic used in OpenCV's 
       CIELAB implementation (cv.cvtColor), maximizing the 8-bit resolution to 
       preserve 'Small Color Differences' (SCD) as defined by Luo et al. (2006).
    
    Programmatic Implementation:
    - axis=(0, 1): Computes bounds across the spatial grid while preserving channels.
    - np.putmask: Ensures numerical stability by preventing division-by-zero on 
      achromatic or uniform-color planes.
    - np.clip/astype: Guarantees strict adherence to the uint8 memory contract.
    """
    x = x.astype(np.float32, copy=False)
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    x_range = x_max - x_min
    
    # Avoid division by zero for uniform channels
    np.putmask(x_range, x_range == 0, 1.0)
    
    normalized = (x - x_min) / x_range * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _hsv_conic_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    HSV conic form, H prime equals V, S prime equals V times S times sin H, V prime equals V times S times cos H.
    OpenCV HSV has H in [0,180] representing [0,360) degrees.

    By using the conic transformation, you avoid the computational errors of "undefined" variables because the multiplication by V=0 nullifies whatever noisy or arbitrary values H and S might hold for a black pixel.
    """
    img_rgb = img_rgb_u8.astype(np.uint8, copy=False)
    
    """
    cv.cvtColor on output returns hsv:
    
    0 <= V, S <= 1, and 0 <= 360 in degrees
    
    However, since we have a target destination type of int8, the following conversion are performed:

    V = 255 * V
    H = 255 * H
    H = H / 2 (fits 0 to 255 in degrees)

    Source: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
    """
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    
    """
    Offsets the int8 conversion from cv.cvtColor
    For H:
    H_deg = H * 2 (since OpenCV scales H to fit in 0..255 for 0..360 degrees)
    H_rad = H_deg * (pi / 180) = H * 2 * (pi / 180) = H * (pi / 90)
    """
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    H = hsv[:, :, 0].astype(np.float32)
    H_rad = (H * np.pi) / 90.0

    """
    Follows the formulation found in Shapiro's Computer Vision book explicitly mentioned as:
    \item $F(i) = [v, v \cdot s \cdot \sin(h),\ v \cdot s \cdot \cos(h)](i)$, where $h$, $s$, and $v$ are the HSV values, for color segmentation.
    """
    c0 = V
    c1 = V * S * np.sin(H_rad)
    c2 = V * S * np.cos(H_rad)

    """
    Since $c_0$ is in the range $[0.0, 1.0]$, it is linearly scaled to $[0, 255]$
    """
    C0 = np.clip(c0 * 255.0, 0, 255).astype(np.uint8)

    """
    c1 and c2 are in the range of [-1.0, 1.0] from the sin and cos components. 
    Adding 1.0 moves the range from [-1.0, 1.0] to [0.0, 2.0].
    Multiplying by 127.5 maps that [0.0, 2.0] range perfectly into the [0, 255] unsigned integer space.
    """
    C1 = np.clip((c1 + 1.0) * 127.5, 0, 255).astype(np.uint8)
    C2 = np.clip((c2 + 1.0) * 127.5, 0, 255).astype(np.uint8)


    return np.stack([C0, C1, C2], axis=2)


def _lab_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """CIELAB, D65, via OpenCV."""
    return cv.cvtColor(img_rgb_u8, cv.COLOR_RGB2LAB)


# ---------- CAM02 SCD using colourscience toolkits if available ----------

def _rgb_u8_to_float01(img_rgb_u8: np.ndarray) -> np.ndarray:
    return img_rgb_u8.astype(np.float32, copy=False) / 255.0


def _cam02_scd_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    H, W = img_rgb_u8.shape[:2]
    try:
        from colorspacious import cspace_convert  # type: ignore
        """
        Documentation states that they define the models from the paper:
        Uniform colour spaces based on CIECAMO2 colour appearance model
        Source: https://colour.readthedocs.io/en/develop/_modules/colour/models/cam02_ucs.html
        """
        jab = cspace_convert(
            img_rgb_u8.astype(np.float32) / 255.0, # colorspacious expects float in 0..1 for sRGB input
            "sRGB1",
            "CAM02-SCD"
        )
        jab = np.asarray(jab, dtype=np.float32)

        """
        jab.ndim = 2 checks if output has been flattened to a 2D matrix due to colorspacious optimizations
        jab.shape == (H * W, 3) checks if the flattened shape matches the expected number of pixels and 3 channels
        """
        if jab.ndim == 2 and jab.shape == (H * W, 3):
            jab = jab.reshape(H, W, 3)

        jab = _ensure_hwc3(jab, H, W, where="cam02_scd(colorspacious)")
        return _scale_to_uint8_per_channel(jab)
    except Exception:
        raise RuntimeError("Use colorspacious for CAM02 SCD")


# ---------- CAM16 vectorized implementation ----------
"""
Matrix M16 (Equation 18): Derived to solve computational failures of CIECAM02.
This matrix transforms XYZ tristimulus values into the CAM16 cone response space.

Source: Li C, Li Z, Wang Z, et al. Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS. Color Res Appl. 2017;00:1-12. https://doi.org/10.1002/col.22131
"""
_CAT16 = np.array([
    [ 0.401288,  0.650173, -0.051461],
    [-0.250268,  1.204414,  0.045854],
    [-0.002079,  0.048952,  0.953127],
], dtype=np.float32)

"""
Standard conversion matrix from sRGB to CIE XYZ (assuming D65 white point)
"""
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
    """
    White point chromaticity: x = 0.3127, y = 0.3290 (D65)
    
    D65 mentioned Li et. al. (2017) as a viable reference illuminant for CAM16
    Source: https://registry.color.org/rgb-registry/srgb
    """
    x = 0.31270
    y = 0.32900

    Y = 100.0
    X = Y * x / y
    Z = Y * (1.0 - x - y) / y
    return np.array([X, Y, Z], dtype=np.float32)

def _cam16_nonlinear_response(t: np.ndarray) -> np.ndarray:
    """
    Implements the post-adaptation non-linear compression (Step 4, Eq. 8).
    
    This function transforms adapted cone responses (R'c, G'c, B'c) into 
    compressed achromatic responses (R'a, G'a, B'a). It uses a Naka-Rushton 
    equation to model the 'S-curve' saturation of human vision.
    
    Args:
        t: The adapted cone signal scaled by F_L (Luminance level adaptation):
           t = (F_L * RGB_c / 100)
    """
    # Ensure 32-bit precision for fractional power calculation (0.42)
    t = t.astype(np.float32, copy=False)
    
    # Pre-allocate output array to match input dimensions (e.g., Image H x W x 3)
    out = np.empty_like(t, dtype=np.float32)
    
    # Identify positive signals (Standard physical light behavior)
    pos = t >= 0
    
    if np.any(pos):
        # Extract only positive values to avoid complex numbers in np.power
        tp = t[pos]
        
        # Apply the 0.42 power law (Eq. 8: 't' raised to 0.42)
        tp42 = np.power(tp, 0.42)
        
        # Calculate compressed response: 400 * (t^0.42 / (t^0.42 + 27.13)) + 0.1
        # 400.0: Max neural response
        # 27.13: Semi-saturation constant (where response is half-max)
        # 0.1:   The 'pedestal' or noise floor offset
        out[pos] = 400.0 * tp42 / (tp42 + 27.13) + 0.1

    # Handle negative signals (Numerical noise or out-of-gamut artifacts)
    neg = ~pos
    if np.any(neg):
        # Treat negative values symmetrically: -f(|t|) + 0.1
        # This prevents NaN results while maintaining mathematical continuity
        tn = -t[neg]
        tn42 = np.power(tn, 0.42)
        
        # Flipped curve for negative inputs
        out[neg] = -400.0 * tn42 / (tn42 + 27.13) + 0.1
        
    return out


@functools.lru_cache(maxsize=None)
def _cam16_setup():
    """
    Precompute context for CAM16 under dim surround, sRGB like viewing.
    """
    XYZ_w = _whitepoint_D65_XYZ()
    """
    Taken from Table A1 surround parameters
    """
    F, c, Nc = 0.9, 0.59, 0.9

    """
    Ambient illuminance: 64 = E_w
    Background illuminance factor: y_b = 20 taken from
        White point luminance: 80 cd/m2 / 
        Image background (proximal field): 16 cd/m
    Source: https://registry.color.org/rgb-registry/srgb
    """
    E_w = 64.0
    L_w = E_w / np.pi
    Y_b = 20.0
    L_A = (L_w * Y_b) / XYZ_w[1]

    # Step 1.1: Calculate Degree of Adaptation (D) (Equation 4)
    RGB_w = _CAT16 @ XYZ_w
    D = F * (1.0 - (1.0 / 3.6) * np.exp(-(L_A + 42.0) / 92.0))
    D = np.clip(D, 0.0, 1.0)
    D_RGB = D * XYZ_w[1] / RGB_w + 1.0 - D

    # Step 1.2: Calculate Luminance Adaptation Factor (FL) (Equation 5)
    k = 1.0 / (5.0 * L_A + 1.0)
    F_L = 0.2 * k**4 * 5.0 * L_A + 0.1 * (1.0 - k**4)**2 * (5.0 * L_A)**(1.0 / 3.0)

    # Step 1.3: Calculate induction factors (Equations 6 & 7)
    n = Y_b / XYZ_w[1]
    z = 1.48 + n**0.5
    N_bb = 0.725 * (1.0 / n)**0.2
    N_cb = N_bb

    # Step 1.4: Calculate adapted achromatic white (Aw)
    RGB_wc = D_RGB * RGB_w
    # t here is a modularization of the expression the computation of RGB_aw
    t = (F_L * RGB_wc / 100.0) 
    RGB_aw = _cam16_nonlinear_response(t)

    # Note that @ here leads to a scalar value since first argument is 1x3 @ 3x1 = 1x1 or a scalar value
    A_w = (np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) @ RGB_aw - 0.305) * N_bb

    return {
        "F_L": float(F_L), "c": float(c), "Nc": float(Nc), "n": float(n), "z": float(z),
        "N_bb": float(N_bb), "N_cb": float(N_cb), "A_w": float(A_w),
        "D_RGB": D_RGB.astype(np.float32),
    }

def _srgb_u8_to_linear01(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Converts 8-bit sRGB values to linear-light [0, 1] floats.
    
    This function removes the sRGB transfer function (gamma encoding) to 
    recover the linear Tristimulus values required for colorimetry.
    
    Source: https://bottosson.github.io/posts/colorwrong/#what-can-we-do%3F
    """
    # Normalize 0-255 integers to 0.0-1.0 floats
    x = img_rgb_u8.astype(np.float32, copy=False) / 255.0
    
    # Identify very dark pixels that fall within the linear slope (Toe)
    mask = x <= 0.04045
    
    # Pre-allocate output array
    y = np.empty_like(x, dtype=np.float32)
    
    # Linear segment: Applied to dark values to prevent infinite slope at zero
    y[mask]  = x[mask] / 12.92
    
    # Power-law segment: The main 'gamma' curve for midtones and highlights
    y[~mask] = ((x[~mask] + 0.055) / 1.055) ** 2.4
    
    return y


def _cam16_forward_JMh_from_rgb(img_rgb_u8: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CAM16 J, M, h for an HxWx3 RGB uint8 image.
    Returns H, W, and flattened J, M, h vectors.
    """
    ### Precompute components independent of the input sample
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

    """
    Pre-step: Linearize sRGB and convert to XYZ Tristimulus Values
    """
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    H, W, _ = rgb_lin.shape
    rgb_flat = rgb_lin.reshape(-1, 3) # Flattens to (H*W, 3) for matrix multiplication
    XYZ = (rgb_flat @ _SRGB_TO_XYZ.T) * 100.0

    """
    Step 1: Calculate cone responses
    Step 2: Complete the color adaptation of the illuminant in the corresponding cone response space
    """
    RGB = (XYZ @ _CAT16.T) * D_RGB

    """
    Step 3: Calculate the postadaptation cone response
    """
    t = (F_L * RGB / 100.0)
    RGB_a = _cam16_nonlinear_response(t)

    """
    Step 4: Calculate Redness - Greenness (a),Yellowness Blueness (b) components,
        and hue angle (h)
    """
    a = RGB_a @ np.array([1.0, -12.0 / 11.0, 1.0 / 11.0], dtype=np.float32)
    b = RGB_a @ np.array([1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0], dtype=np.float32)
    h = np.degrees(np.arctan2(b, a)).astype(np.float32)
    h[h < 0.0] += 360.0
    h_rad = np.radians(h)

    """
    Step 5: Calculate eccentricity factor (e)

    Note: H and H_c are not computed since they are 
        only necessary if you need to describe a color in words 
        (e.g., "50% yellow, 50% red")
    """
    e = 0.25 * (np.cos(h_rad + 2.0) + 3.8)

    """
    Step 6: Calculate achromatic response A
    """
    A = (RGB_a @ np.array([2.0, 1.0, 1.0 / 20.0], dtype=np.float32) - 0.305) * N_bb

    """
    Step 7: Calculate the correlate of lightness J
    """
    A_w_safe = float(np.maximum(A_w, 1e-6))
    J = 100.0 * np.power(np.maximum(A, 0.0) / A_w_safe, c * z)

    # Step 8 is skipped as correlate of brightness Q is not needed for SCD and JMh features.

    """
    Step 9
    """
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
    """
    Transforms sRGB image to the CAM16-SCD (Small Colour Difference) Uniform Colour Space.
    
    This space is optimized for predicting visual differences in the range of 
    0-5 Delta E units. It applies non-linear compression to CAM16 Lightness (J) 
    and Colourfulness (M) to achieve perceptual uniformity.

    Source: 
        Luo, M. R., Cui, G., & Li, C. (2006). "Uniform Colour Spaces Based on 
        CIECAM02 Colour Appearance Model". Color Research & Application.
        Equation (10) and Table II.

    Returns:
        An array of (J', a', b') coordinates scaled to uint8 for visualization 
        or storage.
    """
    # 1. Obtain the standard CAM16 correlates (Step 1-10 of the CAM16 Forward Model)
    H, W, J, M, h = _cam16_forward_JMh_from_rgb(img_rgb_u8)

    # 2. Define UCS constants for the SCD (Small Colour Difference) version
    # c1: Lightness scaling constant (Standardized across UCS/SCD/LCD)
    # c2: Colourfulness compression constant (0.0363 is specific to SCD)
    c1, c2 = 0.007, 0.0363

    # 3. Calculate Modified Lightness (J')
    # This hyperbolic function maps J [0, 100] to J' [0, 100]
    # Reference Paper Eq: J' = ((1 + 100 * c1) * J) / (1 + c1 * J)
    Jp = ((1.0 + 100.0 * c1) * J) / (1.0 + c1 * J)

    # 4. Calculate Modified Colourfulness (M')
    # Uses a logarithmic compression to match the human eye's saturation response.
    # np.log1p(x) is used for numerical precision of ln(1 + x)
    # Reference Paper Eq: M' = (1 / c2) * ln(1 + c2 * M)
    Mp = np.log1p(c2 * M) / c2

    # 5. Convert Polar coordinates (M', h) to Cartesian coordinates (a', b')
    # This allows for Euclidean distance calculation: Delta_E = sqrt(dJ'^2 + da'^2 + db'^2)
    h_rad = np.radians(h)
    ap = Mp * np.cos(h_rad)  # Red-Green dimension
    bp = Mp * np.sin(h_rad)  # Yellow-Blue dimension

    # 6. Finalize output structure
    # Stack into a single HxWx3 array and cast to float32 for downstream processing
    jab = np.stack([Jp, ap, bp], axis=1).reshape(H, W, 3).astype(np.float32)
    
    # Scale back to 0-255 range for image representation
    return _scale_to_uint8_per_channel(jab)

# ---------- Added modern color spaces ----------

"""
Directly taken from Ottoson blog

Source: https://bottosson.github.io/posts/oklab/
"""
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
    """
    Follows the exact step by step in:
    Lab linear_srgb_to_oklab(RGB c) pseudocode
    Source: https://bottosson.github.io/posts/oklab/
    """
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    lms = rgb_lin @ _OKLAB_M1.T

    # Guard rail for extreme cases
    lms = np.clip(lms, 0.0, None) 

    # Stands for cube root
    lms_cbrt = np.cbrt(lms)

    lab = lms_cbrt @ _OKLAB_M2.T
    return _scale_to_uint8_per_channel(lab)

def _oklch_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Follows the exact ste by step to produce oklab in _oklab_from_rgb function
    """
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    lms = rgb_lin @ _OKLAB_M1.T
    lms = np.clip(lms, 0.0, None)
    lms_cbrt = np.cbrt(lms)
    lab = lms_cbrt @ _OKLAB_M2.T

    """
    Clips lightness to [0, 1] to prevent out-of-gamut issues in the final LCh conversion.
    Extracts the L, a, b channels from the Oklab space for the subsequent conversion to LCh.
    """
    L = np.clip(lab[:, :, 0], 0.0, 1.0)
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    """
    Follows the exact conic representation formulas in Ottosom
    https://bottosson.github.io/posts/oklab/
    """
    C = np.sqrt(a * a + b * b)
    h = np.degrees(np.arctan2(b, a))
    h[h < 0.0] += 360.0
    
    """
    Transforming to target int8 representation
    """
    # L contains [0, 1], so we can directly scale to [0, 255]
    L8 = np.clip(L * 255.0, 0, 255).astype(np.uint8)

    # _scale_to_uint8_per_channel is a general purpose function that expects a 3D array with shape (H, W, 1) for the channel to be scaled. By adding a new axis with None, we create a temporary shape of (H, W, 1) that allows the function to compute the min and max across the spatial dimensions while treating the single channel as a separate entity. After scaling, we take the first (and only) channel back out with [:, :, 0] to return to the original 2D shape for that channel.
    C8 = _scale_to_uint8_per_channel(C[:, :, None])[:, :, 0] 

    # h is in degrees [0, 360), so we can directly scale to [0, 255] by multiplying by (255/360)
    h8 = np.clip((h / 360.0) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([L8, C8, h8], axis=2)

"""
Taken from the paper:
Perceptually uniform color space for image signals including high dynamic range and wide gamut

Constants defined in Section 5.2: Full Model of Jzazbz
"""
_JZ_b  = 1.15
_JZ_g  = 0.66
_JZ_c1 = 0.8359375
_JZ_c2 = 18.8515625
_JZ_c3 = 18.6875
_JZ_n  = 0.1593017578125
_JZ_p  = 134.034375
_JZ_d  = -0.56
_JZ_d0 = 1.6295499532821566e-11 # e-11 = 10 ** (-11)

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
    """
    Follows formulation of Equation 10 but does not include
    1/10000 since input is linearized and scaled to [0, 1] 
    where in the formulations scales to [0, 10000] where 10000 is peak luminance
    """
    x = np.clip(x.astype(np.float32), 0.0, None)
    x_m = np.power(x, _JZ_n)
    num = _JZ_c1 + _JZ_c2 * x_m
    den = 1.0 + _JZ_c3 * x_m
    y = np.power(num / np.maximum(den, 1e-12), _JZ_p)
    return y.astype(np.float32)

def _jzazbz_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Follows the process defined in Section 5.2: Full Model of Jzazbz
    """

    """
    Converts RGB to XYZ
    """
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]

    """
    Corresponds to Eq. 8
    """
    Xp = _JZ_b * X - (_JZ_b - 1.0) * Z
    Yp = _JZ_g * Y - (_JZ_g - 1.0) * X
    Zp = Z

    """
    Corresponds to Eq. 9
    """
    XYZp = np.stack([Xp, Yp, Zp], axis=2)
    LMS = XYZp @ _JZ_M1.T

    """
    Corresponds to Eq. 10
    """
    LMS_p = _pq_oetf_inverse(LMS)
    
    """
    Corresponds to Eq. 11
    """
    IzAzBz = LMS_p @ _JZ_M2.T
    Iz = IzAzBz[:, :, 0]
    az = IzAzBz[:, :, 1]
    bz = IzAzBz[:, :, 2]

    """
    Corresponds to Eq. 12
    """
    Jz = ((1.0 + _JZ_d) * Iz) / (1.0 + _JZ_d * Iz) - _JZ_d0
    
    jzazbz = np.stack([Jz, az, bz], axis=2).astype(np.float32)
    return _scale_to_uint8_per_channel(jzazbz)

def _jzczhz_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Follows the same process as _jzazbz_from_rgb but adds the final step of converting from Cartesian (a, b) to polar (C, h) coordinates for the chroma components.
    """
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

    """
    Corresponds to Eq. 13 and 14
    """
    Cz = np.sqrt(az * az + bz * bz)
    hz = np.degrees(np.arctan2(bz, az))
    hz[hz < 0.0] += 360.0

    
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
    rp = img_rgb_u8[:, :, 0].astype(np.float32) / 255.0
    gp = img_rgb_u8[:, :, 1].astype(np.float32) / 255.0
    bp = img_rgb_u8[:, :, 2].astype(np.float32) / 255.0
    Yp = 0.2126 * rp + 0.7152 * gp + 0.0722 * bp
    Cb = -0.1146 * rp - 0.3854 * gp + 0.5 * bp
    Cr =  0.5 * rp - 0.4542 * gp - 0.0458 * bp
    Y8  = np.clip(Yp * 255.0, 0, 255).astype(np.uint8)
    Cb8 = np.clip((Cb + 0.5) * 255.0, 0, 255).astype(np.uint8)
    Cr8 = np.clip((Cr + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([Y8, Cb8, Cr8], axis=2)

def _ictcp_pq_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_u8_to_linear01(img_rgb_u8)
    XYZ = rgb_lin @ _SRGB_TO_XYZ.T
    rgb2020_lin = XYZ @ _XYZ_TO_BT2020.T
    LMS = rgb2020_lin @ _ICTCP_RGB2020_TO_LMS.T
    LMS = np.clip(LMS, 0.0, None)
    LMS_p = _pq_oetf_inverse(LMS)
    ICTCP = LMS_p @ _ICTCP_LMS_TO_ICTCP_PQ.T
    return _scale_to_uint8_per_channel(ICTCP)

# ---------- Ruderman l alpha beta opponent space ----------

def _ruderman_lab_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Ruderman l alpha beta opponent space.
    Steps:
      sRGB to linear RGB
      linear RGB to LMS using Smith and Pokorny fundamentals
      log10 on LMS
      orthonormal transform to l, alpha, beta
    Returns uint8 features scaled per channel.
    """
    # sRGB to linear
    lin = _srgb_u8_to_linear01(img_rgb_u8)

    # linear RGB to LMS
    M_rgb2lms = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ], dtype=np.float32)
    lms = lin @ M_rgb2lms.T

    # avoid log of zero
    lms = np.maximum(lms, 1e-6)
    lms_log = np.log10(lms)

    # log LMS to l alpha beta via orthonormal matrix
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    inv_sqrt6 = 1.0 / np.sqrt(6.0)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    M_lms2lab = np.array([
        [inv_sqrt3,  inv_sqrt3,  inv_sqrt3],
        [inv_sqrt6,  inv_sqrt6, -2.0 * inv_sqrt6],
        [inv_sqrt2, -inv_sqrt2, 0.0]
    ], dtype=np.float32)

    lab = lms_log @ M_lms2lab.T
    return _scale_to_uint8_per_channel(lab)

# --- Opponent O1,O2,O3 ---
def _opponent_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    x = img_rgb_u8.astype(np.float32) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    O1 = (R + G + B) / 3.0
    O2 = G - R
    O3 = B - (R + G) / 2.0
    opp = np.stack([O1, O2, O3], axis=2).astype(np.float32)
    return _scale_to_uint8_per_channel(opp)

# --- Log-chromaticity, I, log(R/G), log(B/G) ---
def _log_chroma_from_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    eps = 1e-6
    x = img_rgb_u8.astype(np.float32) / 255.0
    R, G, B = x[..., 0] + eps, x[..., 1] + eps, x[..., 2] + eps
    I = 0.299 * R + 0.587 * G + 0.114 * B
    Lrg = np.log(R / G)
    Lbg = np.log(B / G)
    out = np.stack([I, Lrg, Lbg], axis=2).astype(np.float32)
    return _scale_to_uint8_per_channel(out)

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
        'oklab': _oklab_from_rgb,
        'oklch': _oklch_from_rgb,
        'jzazbz': _jzazbz_from_rgb,
        'jzczhz': _jzczhz_from_rgb,
        'ictcp_pq': _ictcp_pq_from_rgb,
        'xyz': _xyz_from_rgb,
        'ycbcr_bt709': _ycbcr_bt709_from_rgb,
        'srgb_linear': _srgb_linear_from_rgb,
        'ruderman_lab': _ruderman_lab_from_rgb,
        'opponent': _opponent_from_rgb,
        'log_chroma': _log_chroma_from_rgb,
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
