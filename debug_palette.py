"""Debug VOC palette calculation."""

import numpy as np

def voc_palette_grabcut() -> np.ndarray:
    """VOC palette from grabcut.py"""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lab = i
        for j in range(8):
            pal[i, 0] |= (((lab >> 0) & 1) << (7 - j))
            pal[i, 1] |= (((lab >> 1) & 1) << (7 - j))
            pal[i, 2] |= (((lab >> 2) & 1) << (7 - j))
            lab >>= 3
    return pal

def voc_palette_correct() -> np.ndarray:
    """Corrected VOC palette."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(8):
            pal[i, 0] |= (((i >> 0) & 1) << (7 - j))
            pal[i, 1] |= (((i >> 1) & 1) << (7 - j))
            pal[i, 2] |= (((i >> 2) & 1) << (7 - j))
    return pal

pal_grabcut = voc_palette_grabcut()
pal_correct = voc_palette_correct()

print("Comparing palettes for classes 0-10:")
print("Class | GrabCut RGB      | Correct RGB      | Match")
print("------|------------------|------------------|------")
for i in range(11):
    gc = pal_grabcut[i]
    co = pal_correct[i]
    match = "✓" if np.array_equal(gc, co) else "✗"
    print(f"  {i:2d}  | {gc[0]:3d},{gc[1]:3d},{gc[2]:3d}     | {co[0]:3d},{co[1]:3d},{co[2]:3d}     | {match}")

print("\n\nDetailed calculation for class 9:")
print("Binary: 1001 (9 in decimal)")
print("Expected RGB: R=255 (bit 0 set), G=0 (bit 1 clear), B=0 (bit 2 clear)")
print("               Before bit 3, which resets everything after 8 iterations")

# Manually trace through class 9 in grabcut version
i = 9
lab = i
pal_manual = np.zeros(3, dtype=np.uint8)
print("\nGrabCut version trace for i=9:")
for j in range(8):
    r_bit = ((lab >> 0) & 1)
    g_bit = ((lab >> 1) & 1)
    b_bit = ((lab >> 2) & 1)
    r_shift = 7 - j
    print(f"  j={j}: lab={lab:04b}, bits=[{r_bit},{g_bit},{b_bit}], shift={r_shift}")
    pal_manual[0] |= (r_bit << r_shift)
    pal_manual[1] |= (g_bit << r_shift)
    pal_manual[2] |= (b_bit << r_shift)
    print(f"       RGB so far: [{pal_manual[0]:3d}, {pal_manual[1]:3d}, {pal_manual[2]:3d}]")
    lab >>= 3

print(f"\nFinal GrabCut RGB for class 9: {pal_grabcut[9]}")
print(f"Expected RGB for class 9: [255, 0, 0] (bright red)")
