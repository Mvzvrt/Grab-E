#!/usr/bin/env python3
# Filename: verify_installation.py
# -*- coding: utf-8 -*-
"""
Verify that all dependencies are correctly installed for the Interactive GrabCut application.
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_opencv_grabcut():
    """Check if OpenCV GrabCut is available."""
    try:
        import cv2 as cv
        # Check if GrabCut constants exist
        assert hasattr(cv, 'GC_INIT_WITH_MASK')
        assert hasattr(cv, 'GC_EVAL')
        assert hasattr(cv, 'GC_BGD')
        assert hasattr(cv, 'GC_FGD')
        print(f"✓ OpenCV GrabCut support")
        return True
    except Exception as e:
        print(f"✗ OpenCV GrabCut support: {e}")
        return False

def check_parent_modules():
    """Check if parent directory modules are accessible."""
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))
    
    modules_ok = True
    
    # Check color_space.py
    try:
        from color_space import convert_color_space
        print(f"✓ color_space.py (parent directory)")
    except ImportError as e:
        print(f"✗ color_space.py: {e}")
        modules_ok = False
    
    # Check mgc_api.py
    try:
        from mgc_api import _expand_seeds, _apply_guided_filter
        print(f"✓ mgc_api.py (parent directory)")
    except ImportError as e:
        print(f"✗ mgc_api.py: {e}")
        modules_ok = False
    
    return modules_ok

def main():
    """Main verification routine."""
    print("=" * 60)
    print("Interactive GrabCut - Installation Verification")
    print("=" * 60)
    print()
    
    print("Checking Python version...")
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 8:
        print(f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    else:
        print(f"✗ Python version too old: {sys.version}")
        print("  Requires Python 3.8 or higher")
        return False
    
    print()
    print("Checking core dependencies...")
    
    all_ok = True
    
    # Core dependencies
    all_ok &= check_import("numpy")
    all_ok &= check_import("cv2", "opencv-python")
    all_ok &= check_import("PIL", "Pillow")
    all_ok &= check_import("skimage", "scikit-image")
    all_ok &= check_import("PySide6")
    
    print()
    print("Checking PySide6 components...")
    all_ok &= check_import("PySide6.QtWidgets")
    all_ok &= check_import("PySide6.QtCore")
    all_ok &= check_import("PySide6.QtGui")
    
    print()
    print("Checking OpenCV capabilities...")
    all_ok &= check_opencv_grabcut()
    
    print()
    print("Checking parent directory modules...")
    all_ok &= check_parent_modules()
    
    print()
    print("Checking application modules...")
    
    try:
        from grabcut_refine_api import (
            opencv_grabcut_refine,
            SegmentationState,
            MultiClassSegmentationSession
        )
        print(f"✓ grabcut_refine_api.py")
    except ImportError as e:
        print(f"✗ grabcut_refine_api.py: {e}")
        all_ok = False
    
    try:
        from canvas_widget import CanvasWidget
        print(f"✓ canvas_widget.py")
    except ImportError as e:
        print(f"✗ canvas_widget.py: {e}")
        all_ok = False
    
    try:
        from main_window import MainWindow
        print(f"✓ main_window.py")
    except ImportError as e:
        print(f"✗ main_window.py: {e}")
        all_ok = False
    
    try:
        from utils import voc_palette, save_indexed_png
        print(f"✓ utils.py")
    except ImportError as e:
        print(f"✗ utils.py: {e}")
        all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All checks passed! You're ready to run the application.")
        print()
        print("Start the application with:")
        print("  python main.py")
        print()
        print("Or use the quick-start scripts:")
        print("  Windows: run.bat")
        print("  Linux/Mac: ./run.sh")
        return True
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("If parent directory modules are missing, ensure you're running")
        print("from the src/ directory and that color_space.py, mgc_api.py")
        print("exist in the parent directory.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
