"""Compare palette application between view_npy.py and grabcut.py methods."""

import numpy as np
from PIL import Image
from pathlib import Path

def voc_palette() -> np.ndarray:
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

def save_with_grabcut_method(mask_2d: np.ndarray, path: str) -> None:
    """Save using grabcut.py's method (PIL palette mode)"""
    img = Image.fromarray(mask_2d.astype(np.uint8))
    img = img.convert("P")
    img.putpalette(voc_palette().ravel())
    img.save(path)

def save_with_viewnpy_method(anns: np.ndarray, path: str) -> None:
    """Save using view_npy.py's method (manual RGB coloring)"""
    anns = np.asarray(anns, dtype=np.int32)
    viz = np.zeros((*anns.shape, 3), dtype=np.uint8)
    
    # 0 -> white (unlabeled)
    viz[anns == 0] = [255, 255, 255]
    
    # 1 -> black (background)
    viz[anns == 1] = [0, 0, 0]
    
    # >1 -> VOC palette (foreground classes)
    pal = voc_palette()
    max_label = int(anns.max())
    if max_label > 1:
        for class_id in range(2, min(max_label + 1, 256)):
            mask = (anns == class_id)
            if mask.any():
                viz[mask] = pal[class_id]
    
    Image.fromarray(viz).save(path)

# Load both files
original = np.load("diagram/2011_000758.npy")
refined = np.load("diagram/refined_indexed.npy")

print("Original shape:", original.shape, "dtype:", original.dtype)
print("Refined shape:", refined.shape, "dtype:", refined.dtype)
print("\nOriginal unique values:", np.unique(original))
print("Refined unique values:", np.unique(refined))

# Save with both methods
print("\nSaving original with grabcut method...")
save_with_grabcut_method(original, "diagram/test_original_grabcut.png")

print("Saving original with view_npy method...")
save_with_viewnpy_method(original, "diagram/test_original_viewnpy.png")

print("Saving refined with grabcut method...")
save_with_grabcut_method(refined, "diagram/test_refined_grabcut.png")

print("Saving refined with view_npy method...")
save_with_viewnpy_method(refined, "diagram/test_refined_viewnpy.png")

print("\nNow checking the actual pixel colors...")
# Load back and check colors
grabcut_img = np.array(Image.open("diagram/test_original_grabcut.png").convert("RGB"))
viewnpy_img = np.array(Image.open("diagram/test_original_viewnpy.png"))

# Find a pixel that should be class 9 (red)
class_9_mask = original == 9
if class_9_mask.any():
    y, x = np.where(class_9_mask)
    print(f"\nPixel at ({y[0]}, {x[0]}) with class 9:")
    print(f"  Grabcut method RGB: {grabcut_img[y[0], x[0]]}")
    print(f"  View_npy method RGB: {viewnpy_img[y[0], x[0]]}")
    print(f"  Expected (VOC palette[9]): {voc_palette()[9]}")

print("\nFiles saved successfully!")
