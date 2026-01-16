# Color Space Implementation Methods

This document groups the supported color spaces in GrabCut by their implementation approach: using libraries, applying coefficients/formulas, or following published papers.

---

## 1. Library-Based Implementations

These use external libraries to handle the conversion:

### **OpenCV**

- **`cielab`** (CIELAB, D65)
  - Uses `cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)`
  - Standard CIE perceptual color space with D65 illuminant
  - Most widely adopted perceptual color space

### **Colorspacious Library**

- **`c02_scd`** (CAM02-SCD)
  - Uses `colorspacious.cspace_convert(..., "sRGB1", "CAM02-SCD")`
  - Requires external dependency: `colorspacious`
  - Color Appearance Model from Ebner & Fairchild (1998)

---

## 2. Coefficient-Based Implementations

These apply fixed coefficients or simple mathematical transformations without referencing a specific paper:

### **Direct Coefficients (No Paper Reference)**

- **`rgb`** - Raw RGB pass-through
- **`ycbcr_bt709`** (YCbCr, BT.709)

  - Uses ITU-R BT.709 coefficients:
    - Y = 0.2126*R + 0.7152*G + 0.0722\*B
    - Cb = -0.1146*R - 0.3854*G + 0.5\*B
    - Cr = 0.5*R - 0.4542*G - 0.0458\*B
  - Standard for HDTV/streaming video

- **`opponent`** (O1, O2, O3 opponent space)

  - O1 = (R + G + B) / 3
  - O2 = G - R
  - O3 = B - (R + G) / 2
  - Simple opponent-process type color representation

- **`log_chroma`** (Log-Chromaticity)
  - I = 0.299*R + 0.587*G + 0.114\*B
  - Log(R/G), Log(B/G) for chromatic components
  - Illumination-robust representation

### **Transform-Based**

- **`srgb_linear`**

  - Applies sRGB gamma decompression (sRGB EOTF):
    - For x ≤ 0.04045: linear = x / 12.92
    - For x > 0.04045: linear = ((x + 0.055) / 1.055)^2.4
  - Converts from gamma-encoded sRGB to linear RGB

- **`xyz`** (CIE XYZ)
  - Uses fixed sRGB-to-XYZ transformation matrix:
    - 3×3 matrix multiplication with sRGB linear values
  - CIE standard intermediate color space

---

## 3. Paper-Based Implementations

These implement color spaces defined in published research papers with precise mathematical formulas and constants:

### **HSV Conic (Cylindrical Transformation)**

- **`hsv_conic`**
  - Paper reference: HSV conic formulation
  - Transforms HSV to conic coordinates:
    - C0 = V
    - C1 = V × S × sin(H)
    - C2 = V × S × cos(H)
  - H converted from OpenCV range [0, 180] to radians

### **CAM16 (Color Appearance Model)**

- **`c16_scd`** (CAM16-SCD)
- **`c16_ucs`** (CAM16-UCS) [used internally]
  - Paper: "Comprehensive color solutions: CAM16, CAM16-UCS, and CAM02-SCD" (Li et al., 2017)
  - Complex appearance model with:
    - Chromatic adaptation transform (CAT16)
    - Nonlinear response function (power law)
    - Context-dependent surround parameters
    - Dimension scaling and correction factors
  - Precomputed context: F_L, c, Nc, n, z, N_bb, N_cb, A_w, D_RGB
  - Computes J (lightness), M (colorfulness), h (hue)
  - SCD variant: Applies specific scaling: Jp = ((1 + 100c₁)J) / (1 + c₁J), Mp = log₁p(c₂M) / c₂

### **OKLab & OKLch**

- **`oklab`**
- **`oklch`**
  - Paper: "OK Color spaces" by Ottosson (2020)
  - https://bottosson.github.io/posts/oklab/
  - Two-stage transformation:
    1. sRGB linear → LMS (cone response) via M1 matrix
    2. Cube root of LMS → Lab via M2 matrix
  - LCh variant computes chroma and hue from Lab

### **JzAzBz & JzCzHz**

- **`jzazbz`**
- **`jzczhz`**
  - Paper: "Safdar et al. - Perceptually Uniform Color Space with Improved Hue Uniformity" (2017)
  - Constants per Safdar specification
  - Pipeline:
    1. sRGB linear → XYZ
    2. Adapt XYZ with b=1.15, g=0.66 parameters
    3. XYZ → LMS via M1 matrix
    4. Apply PQ (Perceptual Quantizer) EOTF inverse
    5. LMS → JzAzBz via M2 matrix
    6. Final adjustment: Jz = ((1 + d)Iz) / (1 + d×Iz) - d₀
  - CzHz variant: Convert Az, Bz to polar coordinates (chroma, hue)

### **ICtCp-PQ**

- **`ictcp_pq`** (Image Color Transform, based on BT.2100 PQ)
  - Paper: "Towards a Common Image Quality Metric for High Dynamic Range Video" (Dolby)
  - BT.2100-2 specification for HDR
  - Pipeline:
    1. sRGB linear → XYZ
    2. XYZ → BT.2020 RGB via \_XYZ_TO_BT2020 matrix
    3. BT.2020 RGB → LMS via scaled matrix (1/4096)
    4. Apply PQ inverse EOTF
    5. LMS → ICtCp via scaled transformation matrix
  - Designed for high dynamic range content

### **Ruderman Lab (Opponent-Process Based)**

- **`ruderman_lab`**
  - Paper: "Statistics of natural images: scaling in the woods" by Ruderman (1994)
  - Biological vision-inspired transformation:
    1. sRGB → linear RGB
    2. Linear RGB → LMS using Smith & Pokorny cone fundamentals:
       - M_rgb2lms with constants [0.3811, 0.5783, 0.0402], etc.
    3. log₁₀ applied to LMS (avoid log(0))
    4. Orthonormal transformation to l, α, β channels:
       - l = (1/√3)(log L + log M + log S)
       - α = (1/√6)(log L + log M - 2 log S)
       - β = (1/√2)(log L - log M)

---

## Implementation Summary Table

| Color Space    | Method Type                        | Source                   |
| -------------- | ---------------------------------- | ------------------------ |
| `rgb`          | Pass-through                       | N/A                      |
| `srgb_linear`  | Coefficients (Gamma decompression) | sRGB standard            |
| `xyz`          | Coefficients (Matrix)              | CIE standard matrix      |
| `cielab`       | Library (OpenCV)                   | CIE standard (D65)       |
| `hsv_conic`    | Coefficients (Paper formula)       | HSV conic formulation    |
| `ycbcr_bt709`  | Coefficients                       | ITU-R BT.709             |
| `opponent`     | Coefficients                       | Classic opponent-process |
| `log_chroma`   | Coefficients                       | Log-chromaticity         |
| `c02_scd`      | Library (Colorspacious)            | Ebner & Fairchild (1998) |
| `c16_scd`      | Paper (Full implementation)        | Li et al., CAM16 (2017)  |
| `oklab`        | Paper (Ottosson)                   | Ottosson (2020)          |
| `oklch`        | Paper (Ottosson)                   | Ottosson (2020)          |
| `jzazbz`       | Paper (Safdar et al.)              | Safdar et al. (2017)     |
| `jzczhz`       | Paper (Safdar et al.)              | Safdar et al. (2017)     |
| `ictcp_pq`     | Paper (BT.2100)                    | Dolby/BT.2100-2 HDR      |
| `ruderman_lab` | Paper (Ruderman)                   | Ruderman (1994)          |

---

## Notes

1. **Library implementations** offer simplicity and maintenance advantages but require external dependencies.
2. **Coefficient-based implementations** are lightweight and fast but less sophisticated.
3. **Paper-based implementations** provide state-of-the-art perceptual uniformity and appearance modeling, with comprehensive mathematical foundations.
4. **CAM16 and JzAzBz** are the most advanced, designed for modern HDR and color appearance modeling.
5. **OKLab/OKLch** offer a modern, simpler alternative to traditional Lab with improved hue uniformity.
6. **Ruderman Lab** connects to biological vision research and is particularly useful for natural image statistics.
