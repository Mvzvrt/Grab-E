# Interactive GrabCut GUI Application - Quick Start

This directory contains a **complete PySide6-based GUI application** for interactive multi-class image segmentation with **iterative refinement** support.

## What's New: Iterative Refinement Support

The key innovation is the ability to **refine segmentation results with new scribbles WITHOUT reinitializing the GMM models**. This addresses the limitation identified in the base `grabcut.py` implementation.

### How It Works

1. **Initial Segmentation**: Draw scribbles → Run segmentation → GrabCut learns color models
2. **Iterative Refinement**: Add more scribbles → Click "Refine" → Models are preserved and updated
3. **Result**: Faster convergence and better results through model persistence

## Quick Installation

```bash
# Navigate to the src directory
cd src

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

Or use the quick-start scripts:

- **Windows**: Double-click `run.bat`
- **Linux/Mac**: `chmod +x run.sh && ./run.sh`

## Application Structure

```
src/
├── main.py                    # Application entry point
├── main_window.py             # Main GUI window
├── canvas_widget.py           # Interactive drawing canvas
├── grabcut_refine_api.py      # Refinement API (KEY FILE!)
├── utils.py                   # Helper functions
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── run.bat                    # Windows quick-start
└── run.sh                     # Linux/Mac quick-start
```

## Key Components

### `grabcut_refine_api.py`

The **core innovation** - extends base GrabCut with:

- **`opencv_grabcut_refine()`**: Accepts pre-existing models, uses `cv.GC_EVAL` for refinement
- **`SegmentationState`**: Per-class model persistence
- **`MultiClassSegmentationSession`**: Complete session management

### `canvas_widget.py`

Interactive canvas with:

- Multi-class scribble drawing
- Pan/zoom navigation
- Undo/redo support
- Real-time segmentation overlay

### `main_window.py`

Complete GUI with:

- Menu bar and toolbar
- Dockable control panels
- Settings for color space, iterations, etc.
- Session save/load functionality

## Basic Usage

1. **Launch**: Run `python main.py` from the `src/` directory
2. **Load Image**: File → Open Image (or click "Open Image" button)
3. **Draw Scribbles**:
   - Select class (Background or Foreground 1-20)
   - Left-click and drag to draw
4. **Initial Segmentation**: Click "Run Segmentation"
5. **Refine**:
   - Add more scribbles to correct errors
   - Click "Refine (Keep Models)" ← **This preserves learned models!**
6. **Export**: File → Save Segmentation Mask

## Controls

- **Left Mouse**: Draw scribbles
- **Middle Mouse**: Pan image
- **Mouse Wheel**: Zoom
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **R**: Reset view
- **T**: Toggle overlay

## Requirements

- Python 3.8+
- PySide6 (Qt6 GUI framework)
- NumPy
- OpenCV
- Pillow
- scikit-image

All dependencies are in `requirements.txt`.

## Advanced Features

- **10 Color Spaces**: RGB, JzAzBz, CIELAB, OKLAB, and more
- **Seed Refinement**: Geodesic expansion for better initialization
- **Post-Smoothing**: Edge-aware boundary refinement
- **Session Persistence**: Save/load complete sessions with models
- **Multi-threading**: Non-blocking segmentation

## Technical Comparison

| Feature          | Base grabcut.py         | Interactive App      |
| ---------------- | ----------------------- | -------------------- |
| GUI              | ❌ CLI only             | ✅ Full GUI          |
| Model Reuse      | ❌ Always reinitialize  | ✅ Preserve & refine |
| Interactive      | ❌ Batch only           | ✅ Real-time         |
| Scribble Drawing | ❌ Pre-made annotations | ✅ Draw in app       |
| Session Save     | ❌ No                   | ✅ Yes               |

## Documentation

For complete documentation, see:

- **`src/README.md`**: Full application documentation
- **`grabcut_refine_api.py`**: Detailed API documentation in docstrings

## Troubleshooting

### "Module not found" errors

```bash
cd src
pip install -r requirements.txt
```

### Application won't start

Check Python version:

```bash
python --version  # Should be 3.8+
```

### Slow performance

- Use smaller images (< 2000×2000)
- Reduce GrabCut iterations (try 2-3)
- Disable advanced options

## For Your Thesis

This implementation demonstrates:

1. ✅ **Model Persistence**: GMM models are saved and reused
2. ✅ **Iterative Refinement**: New scribbles refine existing models via `cv.GC_EVAL`
3. ✅ **User-Driven Workflow**: Interactive GUI for practical usage
4. ✅ **Multi-Class Support**: Handle complex scenes with multiple objects
5. ✅ **Session Management**: Complete reproducibility with save/load

### Key Code to Reference

**Model Reuse (grabcut_refine_api.py, line ~85)**:

```python
if bgdModel is not None and fgdModel is not None:
    bgdModel = bgdModel.copy()
    fgdModel = fgdModel.copy()
    mode = cv.GC_EVAL if use_eval_mode else cv.GC_INIT_WITH_MASK
else:
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mode = cv.GC_INIT_WITH_MASK

cv.grabCut(img_feats_u8, mask, None, bgdModel, fgdModel, int(iters), mode)
```

This is the **critical difference** from the base implementation!

## Questions?

Open an issue on GitHub or refer to the comprehensive `src/README.md` for more details.

---

**Happy Segmenting! 🎨✂️**
