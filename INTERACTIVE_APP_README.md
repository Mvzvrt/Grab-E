# Interactive GrabCut Application - Repository Update

## What's New

A complete **PySide6 GUI application** for interactive multi-class image segmentation has been added to the `src/` directory. This application **addresses the key limitation** identified in the analysis: **the inability to perform iterative refinement with new scribbles without reinitializing GMM models**.

## Quick Start

```bash
cd src
pip install -r requirements.txt
python main.py
```

Or use the quick-start scripts:

- Windows: `run.bat`
- Linux/Mac: `run.sh`

## Key Innovation: Iterative Refinement

The application enables **true iterative refinement** by:

1. **Preserving learned GMM models** between refinement iterations
2. **Using OpenCV's `cv.GC_EVAL` mode** instead of always reinitializing
3. **Providing a clean API** for model persistence and reuse

### Technical Implementation

**New Core Function**: `opencv_grabcut_refine()` in `src/grabcut_refine_api.py`

```python
def opencv_grabcut_refine(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    bgdModel: Optional[np.ndarray] = None,  # ← NEW: Accept existing models
    fgdModel: Optional[np.ndarray] = None,  # ← NEW: Accept existing models
    iters: int = 2,
    use_eval_mode: bool = False             # ← NEW: Enable GC_EVAL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

This differs from the original `opencv_grabcut_once()` which always creates fresh models.

## Application Features

### Core Capabilities

- ✅ Load images and draw multi-class scribbles
- ✅ Run initial GrabCut segmentation
- ✅ **Refine results with new scribbles (preserves models!)**
- ✅ Save/load complete sessions with model persistence
- ✅ Export segmentation masks

### Advanced Features

- 🌈 10+ color spaces (RGB, JzAzBz, CIELAB, OKLAB, etc.)
- 🔧 Geodesic seed refinement
- 🎯 Edge-aware post-smoothing
- 📊 Real-time segmentation overlay
- ⚡ Multi-threaded processing

### User Interface

- 🖱️ Interactive canvas with pan/zoom
- ↩️ Full undo/redo support
- 🧹 Eraser tool
- ⌨️ Keyboard shortcuts
- 🎛️ Dockable control panels

## Repository Structure

```
grab-cut/
├── grabcut.py                      # Original batch CLI (no refinement)
├── mgc_core/                       # Modern GrabCut enhancements
│   ├── modern_grabcut.py
│   ├── setup.py
│   └── ...
├── src/                            # ← NEW: Interactive GUI Application
│   ├── main.py                     # Application entry point
│   ├── main_window.py              # Main GUI window
│   ├── canvas_widget.py            # Interactive drawing canvas
│   ├── grabcut_refine_api.py       # ⭐ Core refinement API
│   ├── utils.py                    # Helper functions
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Full documentation
│   ├── QUICKSTART.md               # Quick start guide
│   ├── WORKFLOW.md                 # Visual workflow diagrams
│   ├── TECHNICAL_SUMMARY.md        # Technical implementation details
│   ├── run.bat                     # Windows launcher
│   └── run.sh                      # Linux/Mac launcher
└── ...
```

## Comparison: Original vs New

| Feature                   | Original `grabcut.py`  | New `src/` App              |
| ------------------------- | ---------------------- | --------------------------- |
| **GUI**                   | ❌ CLI only            | ✅ Full PySide6 GUI         |
| **Model Reuse**           | ❌ Always reinitialize | ✅ Preserve & refine        |
| **GC_EVAL Mode**          | ❌ Not used            | ✅ Used for refinement      |
| **Interactive Scribbles** | ❌ Pre-made only       | ✅ Draw in app              |
| **Iterative Workflow**    | ❌ Batch processing    | ✅ Incremental refinement   |
| **Session Persistence**   | ❌ No                  | ✅ Save/load complete state |
| **Real-time Overlay**     | ❌ No                  | ✅ Adjustable opacity       |
| **Multi-class**           | ✅ Yes (1 vs rest)     | ✅ Yes (1 vs rest)          |
| **Batch Processing**      | ✅ Yes                 | ❌ Single image             |

## Answer to Your Thesis Adviser's Question

**Question**: "Can our implementation of `grabcut.py` handle adding new scribbles for refinement?"

**Answer**:

- **Original `grabcut.py`**: ❌ **No** - it always reinitializes GMM models
- **New `src/` application**: ✅ **Yes** - it preserves models and uses `cv.GC_EVAL`

**Key Evidence**:

1. `src/grabcut_refine_api.py:115-125` - Conditional model reuse logic
2. `src/grabcut_refine_api.py:299-320` - Refinement in `segment_class()`
3. `src/main_window.py:451-453` - GUI "Refine (Keep Models)" button

**Demo**: Run `python src/main.py` and use the "Refine (Keep Models)" button!

## Documentation

Comprehensive documentation is available in the `src/` directory:

1. **`README.md`**: Complete user guide and API documentation
2. **`QUICKSTART.md`**: Installation and basic usage
3. **`WORKFLOW.md`**: Visual workflow diagrams and examples
4. **`TECHNICAL_SUMMARY.md`**: Implementation details and comparison

## Example Workflow

```
1. Load image → Draw scribbles → "Run Segmentation"
   ↓
   [Models learned: bgdModel, fgdModel for each class]

2. View result → Identify errors → Draw corrective scribbles
   ↓

3. "Refine (Keep Models)" ← Uses cv.GC_EVAL with existing models!
   ↓
   [Models updated, NOT reinitialized]

4. View improved result → Repeat steps 2-3 as needed
   ↓

5. Save segmentation mask + Save session (optional)
```

## Installation Requirements

- Python 3.8+
- PySide6 (Qt6 GUI framework)
- NumPy, OpenCV, Pillow, scikit-image

All dependencies listed in `src/requirements.txt`.

## For Research/Thesis Use

This implementation demonstrates:

1. **Model persistence** across refinement iterations
2. **Use of `cv.GC_EVAL`** for efficient updates
3. **User-driven interactive segmentation** workflow
4. **Multi-class support** with overlap resolution
5. **Complete reproducibility** via session save/load

The key innovation is the `opencv_grabcut_refine()` function that conditionally reuses models when available, addressing the limitation identified in the base implementation.

## Testing

Basic test to verify refinement works:

```python
from src.grabcut_refine_api import MultiClassSegmentationSession
import numpy as np

# Load image
img = ...  # Your image

# Create session
session = MultiClassSegmentationSession(img)

# Initial segmentation
session.update_annotations(initial_scribbles)
mask1 = session.segment_class(2)  # force_reinit=False by default

# Check models exist
assert session.states[2].has_models()
bgd_before = session.states[2].bgdModel.copy()

# Refinement
session.update_annotations(more_scribbles)
mask2 = session.segment_class(2)  # Uses existing models!

# Verify models updated, not reinitialized
bgd_after = session.states[2].bgdModel
assert not np.array_equal(bgd_before, bgd_after)  # Different
assert session.states[2].iteration_count == 2  # Incremented
```

## Support

For questions or issues:

1. Check `src/README.md` for detailed documentation
2. Review `src/TECHNICAL_SUMMARY.md` for implementation details
3. Open an issue on GitHub

## Credits

This interactive application extends the base GrabCut implementation with:

- Modern GrabCut (MGC) enhancements for seed refinement
- PySide6 GUI framework
- Iterative refinement with model persistence

---

**The `src/` directory contains a complete, production-ready interactive segmentation application that fully supports iterative refinement with model persistence.**
