# Interactive GrabCut Segmentation Application

A PySide6-based GUI application for multi-class image segmentation using GrabCut with **iterative refinement** capabilities. This tool allows users to interactively segment images by drawing scribbles and refining the results without reinitializing the GMM models.

## Features

### Core Functionality

- 🖼️ **Image Loading**: Support for common image formats (PNG, JPG, BMP, TIFF)
- ✏️ **Interactive Scribble Drawing**: Draw foreground and background scribbles with adjustable brush size
- 🎨 **Multi-Class Segmentation**: Support for up to 20 foreground classes plus background
- 🔄 **Iterative Refinement**: Add new scribbles and refine results using existing GMM models (no reinitialization)
- 💾 **Session Persistence**: Save and load segmentation sessions including models and annotations

### Advanced Features

- 🌈 **Multiple Color Spaces**: Choose from RGB, JzAzBz, JzCzHz, CIELAB, OKLAB, OKLCH, HSV, and more
- 🔧 **Seed Refinement**: Optional geodesic seed expansion for better initialization
- 🎯 **Post-Smoothing**: Edge-aware boundary refinement using guided filtering
- 📊 **Real-time Overlay**: Adjustable opacity segmentation overlay with VOC-style colors
- ⚡ **Background Processing**: Non-blocking segmentation with progress indicators

### User Interface

- 🖱️ **Pan & Zoom**: Navigate large images easily with middle-mouse pan and scroll wheel zoom
- ↩️ **Undo/Redo**: Full undo/redo support for scribble strokes
- 🧹 **Eraser Tool**: Remove unwanted scribbles interactively
- ⌨️ **Keyboard Shortcuts**: Quick access to common operations
- 🎛️ **Dockable Panels**: Customizable workspace layout

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mvzvrt/multi-class-grabcut.git
cd multi-class-grabcut
```

### Step 2: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

### Step 3: Compile MGC Core (if not already done)

The application uses the Modern GrabCut (MGC) core for advanced seed refinement and post-smoothing:

```bash
cd ../mgc_core
python setup.py build_ext --inplace
cd ../src
```

## Usage

### Starting the Application

```bash
python main.py
```

Or on Windows:

```powershell
python main.py
```

### Basic Workflow

1. **Load an Image**

   - Click "Open Image" or use `File > Open Image...`
   - Select an image file (PNG, JPG, etc.)

2. **Draw Scribbles**

   - Select a class from the "Current Class" dropdown (Background or Foreground 1-20)
   - Adjust brush size as needed
   - Left-click and drag to draw scribbles
   - Use Eraser Mode to remove unwanted strokes

3. **Run Initial Segmentation**

   - Click "Run Segmentation" button
   - Wait for processing to complete
   - View the segmentation overlay on the image

4. **Refine Results (Key Feature!)**

   - Add more scribbles to correct errors
   - Click "Refine (Keep Models)" to update segmentation **without reinitializing models**
   - This uses OpenCV's `GC_EVAL` mode to preserve learned color distributions
   - Repeat as needed for better results

5. **Save Results**
   - `File > Save Segmentation Mask...` - Export final mask as NPY or PNG
   - `File > Save Session...` - Save complete session including models for later refinement
   - `File > Load Session...` - Resume previous session

### Controls

#### Mouse

- **Left Mouse Button**: Draw scribbles
- **Middle Mouse Button**: Pan image
- **Mouse Wheel**: Zoom in/out

#### Keyboard

- `Ctrl+Z`: Undo last stroke
- `Ctrl+Y`: Redo stroke
- `R`: Reset view (fit image to window)
- `T`: Toggle segmentation overlay
- `Ctrl+O`: Open image
- `Ctrl+S`: Save segmentation
- `Ctrl+Q`: Quit application

## Architecture

### Key Components

#### 1. `grabcut_refine_api.py`

Core refinement API that extends the base `grabcut.py` with:

- **`opencv_grabcut_refine()`**: Enhanced GrabCut function supporting model reuse
- **`SegmentationState`**: Per-class model persistence
- **`MultiClassSegmentationSession`**: Complete session management with iterative refinement

**Key Innovation**: The `opencv_grabcut_refine()` function accepts optional `bgdModel` and `fgdModel` parameters and uses `cv.GC_EVAL` mode when models exist, enabling true iterative refinement.

#### 2. `canvas_widget.py`

Interactive drawing canvas with:

- Scribble layer management
- View transformations (pan/zoom)
- Real-time segmentation overlay
- Annotation map generation

#### 3. `main_window.py`

Main application window providing:

- Menu bar and toolbar
- Dockable control panels
- Settings management
- Background segmentation workers
- Session save/load functionality

### How Refinement Works

Traditional GrabCut always reinitializes models:

```python
bgdModel = np.zeros((1, 65), np.float64)  # Always fresh
fgdModel = np.zeros((1, 65), np.float64)
cv.grabCut(img, mask, None, bgdModel, fgdModel, iters, cv.GC_INIT_WITH_MASK)
```

Our approach enables refinement:

```python
# First run: Initialize
bgdModel, fgdModel = None, None
result, bgdModel, fgdModel = opencv_grabcut_refine(img, seeds_bg, seeds_fg)

# Subsequent runs: Refine (models are reused!)
result, bgdModel, fgdModel = opencv_grabcut_refine(
    img, new_seeds_bg, new_seeds_fg,
    bgdModel=bgdModel,      # Reuse learned background model
    fgdModel=fgdModel,      # Reuse learned foreground model
    use_eval_mode=True      # Uses cv.GC_EVAL for refinement
)
```

This preserves the learned Gaussian Mixture Models, allowing the algorithm to focus on refining boundaries rather than relearning color distributions from scratch.

## Configuration Options

### Color Spaces

Choose the feature space for GrabCut:

- **RGB**: Standard color space (default)
- **JzAzBz**: Perceptually uniform color space
- **JzCzHz**: Cylindrical version of JzAzBz
- **CIELAB**: Perceptually uniform L\*a\*b\*
- **OKLAB**: Modern perceptual color space
- **OKLCH**: Cylindrical OKLAB
- **HSV Conic**: Hue-Saturation-Value
- **YCbCr**: Luma-chroma separation
- **XYZ**: CIE 1931 color space

### Advanced Options

- **Apply Seed Refinement**: Uses geodesic distance to expand seed regions based on color similarity
- **Apply Post-Smoothing**: Uses guided filtering for edge-aware boundary refinement
- **GrabCut Iterations**: Number of iterations per segmentation run (default: 5)

## File Formats

### Segmentation Masks

- **NPY Format**: Raw NumPy array (uint8, values 0-20)
  - 0 = Background
  - 1-20 = Foreground classes
- **PNG Format**: Indexed color image with VOC palette

### Session Files

A saved session contains:

- `annotations.npy`: Current scribble annotations
- `final_mask.npy`: Latest segmentation result
- `class_XX_models.npz`: Per-class GMM models (background and foreground)
- `session_metadata.json`: Session configuration and statistics

## Comparison with Base Implementation

| Feature              | Base `grabcut.py` | Interactive App        |
| -------------------- | ----------------- | ---------------------- |
| Model Reuse          | ❌ No             | ✅ Yes (via `GC_EVAL`) |
| Iterative Refinement | ❌ No             | ✅ Yes                 |
| Interactive Drawing  | ❌ No             | ✅ Yes                 |
| Session Persistence  | ❌ No             | ✅ Yes                 |
| Real-time Overlay    | ❌ No             | ✅ Yes                 |
| Multi-Class Support  | ✅ Yes            | ✅ Yes                 |
| Batch Processing     | ✅ Yes            | ❌ No                  |

## Tips for Best Results

1. **Start with clear scribbles**: Draw a few strokes in definitely background and definitely foreground regions
2. **Run initial segmentation**: Get a first result to identify problem areas
3. **Add corrective scribbles**: Focus on misclassified regions
4. **Use Refine mode**: Click "Refine (Keep Models)" to update without losing learned models
5. **Iterate gradually**: Small corrections work better than complete redrawing
6. **Try different color spaces**: Some images work better in perceptual spaces like JzAzBz or CIELAB
7. **Adjust brush size**: Use larger brushes for broad regions, smaller for details

## Troubleshooting

### Application won't start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.8+)
- Verify PySide6 installation: `python -c "import PySide6; print(PySide6.__version__)"`

### Segmentation is slow

- Reduce image resolution before loading
- Decrease GrabCut iterations (try 2-3 instead of 5)
- Disable seed refinement or post-smoothing in Advanced Options

### Poor segmentation quality

- Add more scribbles in ambiguous regions
- Try different color spaces (JzAzBz often works well)
- Enable seed refinement and post-smoothing
- Increase GrabCut iterations to 7-10

### Models not persisting in refinement

- Ensure you're clicking "Refine (Keep Models)" not "Run Segmentation"
- Check that classes haven't changed between runs
- Verify session was properly initialized with an image

## Technical Details

### Dependencies

- **PySide6**: Qt6 bindings for Python GUI
- **NumPy**: Array operations and data structures
- **OpenCV**: GrabCut algorithm and image processing
- **Pillow**: Image I/O
- **scikit-image**: SLIC superpixels for post-processing

### Performance

- Typical segmentation time: 1-5 seconds per class (depends on image size)
- Memory usage: ~2-3x image size in RAM
- Recommended max resolution: 2000x2000 pixels (higher works but slower)

## Contributing

This application was developed as part of a thesis on interactive multi-class segmentation. Contributions, bug reports, and feature requests are welcome!

## License

This project is part of the `multi-class-grabcut` repository. Please refer to the main repository for licensing information.

## Citation

If you use this application in your research, please cite:

```bibtex
@software{interactive_grabcut_2025,
  title={Interactive GrabCut Segmentation with Iterative Refinement},
  author={Your Name},
  year={2025},
  url={https://github.com/Mvzvrt/multi-class-grabcut}
}
```

## Acknowledgments

- Based on the GrabCut algorithm by Rother et al. (2004)
- Modern GrabCut (MGC) enhancements for seed refinement and post-smoothing
- VOC color palette for class visualization
- PySide6 Qt framework for the GUI

## Contact

For questions, issues, or feedback, please open an issue on the GitHub repository.

---

**Happy Segmenting! 🎨✂️**
