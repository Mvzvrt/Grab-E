"""Verify the fixed palettes match."""

import numpy as np
import sys
import pathlib

# Import from grabcut
sys.path.append(str(pathlib.Path(__file__).parent))
from grabcut import voc_palette as grabcut_palette
from view_npy import voc_palette as viewnpy_palette

gc_pal = grabcut_palette()
vn_pal = viewnpy_palette()

print("Comparing fixed palettes:")
print("Class | GrabCut RGB      | ViewNpy RGB      | Match")
print("------|------------------|------------------|------")
all_match = True
for i in range(20):
    gc = gc_pal[i]
    vn = vn_pal[i]
    match = "✓" if np.array_equal(gc, vn) else "✗"
    if not np.array_equal(gc, vn):
        all_match = False
    print(f"  {i:2d}  | {gc[0]:3d},{gc[1]:3d},{gc[2]:3d}     | {vn[0]:3d},{vn[1]:3d},{vn[2]:3d}     | {match}")

print("\n" + "="*50)
if all_match:
    print("✓ SUCCESS: All palettes match!")
else:
    print("✗ ERROR: Palettes don't match!")

print("\nKey colors:")
print(f"  Class 0 (unlabeled):  {gc_pal[0]}  (should be black [0,0,0])")
print(f"  Class 1 (background): {gc_pal[1]}  (should be bright red [255,0,0])")
print(f"  Class 9 (class 9):    {gc_pal[9]}  (should be bright red [255,0,0])")
