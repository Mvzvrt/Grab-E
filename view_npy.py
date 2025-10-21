# -*- coding: utf-8 -*-
"""
Simple viewer for .npy and .mat annotation files.

Displays the shape, unique labels, and creates a visualization PNG
where 0=white (unlabeled), 1=black (background), >1=VOC palette colors (foreground classes).

Usage:
  python view_npy.py path/to/annotations.npy
  python view_npy.py path/to/annotations.mat
  python view_npy.py path/to/annotations.npy --out custom_output.png
  python view_npy.py path/to/annotations.mat --key GTcls  # Specify .mat key
  python view_npy.py path/to/annotations.npy --stats-only  # Just print info, no image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def voc_palette() -> np.ndarray:
    """Generate VOC-style palette (256x3 uint8) - matches grabcut.py exactly."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lab = i
        for j in range(8):
            pal[i, 0] |= (((lab >> 0) & 1) << (7 - j))
            pal[i, 1] |= (((lab >> 1) & 1) << (7 - j))
            pal[i, 2] |= (((lab >> 2) & 1) << (7 - j))
            lab >>= 3
    return pal


def visualize_annotations(anns: np.ndarray) -> np.ndarray:
    """
    Create RGB visualization of annotation map using PIL palette mode.
    Mirrors the save_indexed_png() method from grabcut.py exactly.
    
    Args:
        anns: HxW int array with labels (0=unlabeled, 1=bg, >1=fg classes)
    
    Returns:
        HxWx3 uint8 RGB image with VOC palette applied
    """
    # Convert to uint8 and create PIL palette image (same as grabcut.py save_indexed_png)
    img = Image.fromarray(anns.astype(np.uint8))
    img = img.convert("P")
    img.putpalette(voc_palette().ravel())
    
    # Convert back to RGB for visualization
    img_rgb = img.convert("RGB")
    return np.asarray(img_rgb, dtype=np.uint8)


def load_annotation_file(filepath: Path, mat_key: str = None) -> np.ndarray:
    """
    Load annotation array from .npy or .mat file.
    
    Args:
        filepath: Path to .npy or .mat file
        mat_key: Key to use for .mat files (if None, auto-detect or list keys)
    
    Returns:
        2D numpy array with annotations
    """
    ext = filepath.suffix.lower()
    
    if ext == '.npy':
        return np.load(filepath)
    
    elif ext == '.mat':
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        
        mat_data = sio.loadmat(str(filepath))
        
        # Filter out metadata keys
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if mat_key is not None:
            # User specified a key
            if mat_key not in mat_data:
                raise KeyError(f"Key '{mat_key}' not found in .mat file. Available keys: {data_keys}")
            return np.asarray(mat_data[mat_key])
        
        # Auto-detect: look for common annotation keys
        common_keys = ['GTcls', 'gt', 'annotation', 'labels', 'mask', 'seg']
        for key in common_keys:
            if key in data_keys:
                print(f"Auto-detected key: '{key}'")
                return np.asarray(mat_data[key])
        
        # If only one data key, use it
        if len(data_keys) == 1:
            key = data_keys[0]
            print(f"Using single available key: '{key}'")
            return np.asarray(mat_data[key])
        
        # Multiple keys and no match
        raise ValueError(
            f"Could not auto-detect annotation key in .mat file.\n"
            f"Available keys: {data_keys}\n"
            f"Please specify with --key option."
        )
    
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: .npy, .mat")


def print_stats(anns: np.ndarray, filepath: Path, mat_key: str = None) -> None:
    """Print statistics about the annotation array."""
    print(f"\nFile: {filepath}")
    if mat_key:
        print(f"Key: {mat_key}")
    print(f"Shape: {anns.shape}")
    print(f"Dtype: {anns.dtype}")
    print(f"Min value: {anns.min()}")
    print(f"Max value: {anns.max()}")
    
    unique_labels = np.unique(anns)
    print(f"\nUnique labels ({len(unique_labels)} total):")
    
    for label in sorted(unique_labels):
        count = np.sum(anns == label)
        percentage = 100.0 * count / anns.size
        
        if label == 0:
            label_type = "unlabeled"
        elif label == 1:
            label_type = "background"
        else:
            label_type = f"class {label}"
        
        print(f"  {label:3d} ({label_type:12s}): {count:8d} pixels ({percentage:5.2f}%)")


def main():
    ap = argparse.ArgumentParser(description="View and visualize .npy or .mat annotation files")
    ap.add_argument("file", type=Path, help="Path to .npy or .mat annotation file")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path for visualization PNG (default: same name with .png extension)")
    ap.add_argument("--key", type=str, default=None,
                    help="Key to use for .mat files (e.g., 'GTcls'). If not specified, will auto-detect.")
    ap.add_argument("--stats-only", action="store_true",
                    help="Only print statistics, don't create visualization")
    args = ap.parse_args()
    
    # Check file exists
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    # Load the file
    try:
        anns = load_annotation_file(args.file, mat_key=args.key)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure it's 2D (squeeze if needed)
    if anns.ndim > 2:
        # Try to squeeze out singleton dimensions
        anns = np.squeeze(anns)
        if anns.ndim != 2:
            print(f"Error: Expected 2D array, got shape {anns.shape} (after squeeze)", file=sys.stderr)
            sys.exit(1)
    elif anns.ndim < 2:
        print(f"Error: Expected 2D array, got shape {anns.shape}", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    print_stats(anns, args.file, mat_key=args.key)
    
    # Create visualization unless stats-only mode
    if not args.stats_only:
        viz = visualize_annotations(anns)
        
        # Determine output path
        if args.out is None:
            out_path = args.file.with_suffix('.png')
        else:
            out_path = args.out
        
        # Save
        Image.fromarray(viz).save(out_path)
        print(f"\nVisualization saved to: {out_path}")


if __name__ == "__main__":
    main()
