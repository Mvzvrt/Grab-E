"""Test the corrected palette offset."""
import numpy as np
from PIL import Image
from diagrams import voc_palette

# Load annotation and visualization
lbl = np.load('diagram/refined_indexed.npy')
viz = np.array(Image.open('diagram/test_corrected.png'))

print("Testing corrected palette offset:")
print("="*50)

# Get VOC palette
pal = voc_palette()

# Check class 9 (should map to VOC index 8)
mask_9 = (lbl == 9)
if mask_9.any():
    y, x = np.where(mask_9)
    viz_color = viz[y[0], x[0]]
    expected_color = pal[8]  # 9 - 1 = 8
    print(f"\nAnnotation class 9:")
    print(f"  Visualization RGB: {viz_color}")
    print(f"  Expected VOC[8]:   {expected_color}")
    print(f"  Match: {'✓' if np.array_equal(viz_color, expected_color) else '✗'}")

# Check background (1 -> should map to VOC index 0)
mask_1 = (lbl == 1)
if mask_1.any():
    y, x = np.where(mask_1)
    viz_color = viz[y[0], x[0]]
    expected_color = pal[0]  # 1 - 1 = 0
    print(f"\nAnnotation class 1 (background):")
    print(f"  Visualization RGB: {viz_color}")
    print(f"  Expected VOC[0]:   {expected_color}")
    print(f"  Match: {'✓' if np.array_equal(viz_color, expected_color) else '✗'}")

# Check unlabeled (0 -> should map to VOC index 255)
mask_0 = (lbl == 0)
if mask_0.any():
    y, x = np.where(mask_0)
    viz_color = viz[y[0], x[0]]
    expected_color = pal[255]  # 0 -> 255
    print(f"\nAnnotation class 0 (unlabeled):")
    print(f"  Visualization RGB: {viz_color}")
    print(f"  Expected VOC[255]: {expected_color}")
    print(f"  Match: {'✓' if np.array_equal(viz_color, expected_color) else '✗'}")

print("\n" + "="*50)
print("VOC palette key colors:")
print(f"  VOC[0] (background):   {pal[0]}")
print(f"  VOC[8] (class 8):      {pal[8]}")
print(f"  VOC[255] (unlabeled):  {pal[255]}")
