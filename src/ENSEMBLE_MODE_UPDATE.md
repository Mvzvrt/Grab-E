# Ensemble Mode Update

## New Features Added

The Interactive GrabCut application now supports **two segmentation modes**:

### 1. Single Color Space Mode (with Iterative Refinement)

- Uses one color space for feature extraction
- **Supports iterative refinement** with model persistence
- **Default**: Ruderman LAB (top-performing color space)
- Best for: Interactive workflows requiring refinement

### 2. Ensemble Mode (Majority Voting)

- Uses three color spaces simultaneously
- Performs majority voting on results from each color space
- **Defaults**: Ruderman LAB + OKLAB + JzCzHz (top-performing combination)
- **Note**: Refinement not supported in ensemble mode (each run reinitializes)
- Best for: Maximum accuracy on first pass

## Available Color Spaces

All 16 color spaces from the research are now available:

1. **ruderman_lab** ⭐ (Default for single mode, top performer)
2. **oklab** ⭐ (Ensemble default #2)
3. **jzczhz** ⭐ (Ensemble default #3)
4. jzazbz
5. cielab
6. oklch
7. c16_scd
8. c02_scd
9. rgb
10. hsv_conic
11. ycbcr_bt709
12. xyz
13. srgb_linear
14. opponent
15. log_chroma
16. ictcp_pq

## UI Changes

### Segmentation Settings Panel

#### Mode Selection

- **Dropdown**: "Single Color Space" or "Ensemble (Majority Voting)"
- Switches between the two configuration panels

#### Single Color Space Panel (visible when Single mode selected)

- **Color Space**: Dropdown with all 16 options
- Default: ruderman_lab

#### Ensemble Panel (visible when Ensemble mode selected)

- **Color Space 1**: Dropdown (default: ruderman_lab)
- **Color Space 2**: Dropdown (default: oklab)
- **Color Space 3**: Dropdown (default: jzczhz)
- **Tie Strategy**: How to break 3-way ties (First/Second/Third)

### Button Behavior

#### Run Segmentation Button

- Single mode: Initializes models and runs segmentation
- Ensemble mode: Runs all three color spaces and votes

#### Refine Button

- **Only enabled in Single Color Space mode**
- Uses existing models with new scribbles
- Disabled in Ensemble mode

## Usage Examples

### Single Color Space Workflow (with Refinement)

```
1. Select "Single Color Space" mode
2. Choose color space (e.g., ruderman_lab)
3. Draw initial scribbles
4. Click "Run Segmentation"
   → Models learned and saved
5. Add more scribbles to correct errors
6. Click "Refine (Keep Models)"
   → Models preserved and updated
7. Repeat steps 5-6 as needed
8. Save result
```

### Ensemble Workflow (Maximum Accuracy)

```
1. Select "Ensemble (Majority Voting)" mode
2. Configure three color spaces:
   - Color Space 1: ruderman_lab
   - Color Space 2: oklab
   - Color Space 3: jzczhz
3. Draw scribbles
4. Click "Run Segmentation"
   → Runs 3 segmentations and votes
5. View result
6. If unsatisfied, add more scribbles and run again
   (Note: Each run reinitializes, no model preservation)
7. Save result
```

## Technical Details

### Implementation

**New Function**: `segment_all_classes_ensemble()` in `grabcut_refine_api.py`

- Takes 3 color spaces as input
- Creates temporary session for each color space
- Runs full segmentation on each
- Applies `majority_vote_indexed()` to combine results

**New Worker**: `EnsembleSegmentationWorker` in `main_window.py`

- Runs ensemble segmentation in background thread
- Reports progress for each color space
- Returns final voted mask

### Majority Voting Logic

For each pixel:

1. If 2+ color spaces agree → Use agreed label
2. If all 3 disagree (3-way tie) → Use tie-breaking strategy:
   - "First": Use Color Space 1's label
   - "Second": Use Color Space 2's label
   - "Third": Use Color Space 3's label

### Performance Comparison

| Feature    | Single Mode                   | Ensemble Mode              |
| ---------- | ----------------------------- | -------------------------- |
| Speed      | Fast (~2-5s)                  | Slower (~6-15s, 3x single) |
| Accuracy   | Good (depends on color space) | Excellent (voted result)   |
| Refinement | ✅ Yes (with model reuse)     | ❌ No                      |
| Best For   | Interactive workflows         | One-shot accuracy          |

## Why Ruderman LAB is Default

Based on research results:

- Consistently top performer across test images
- Good balance of color and lightness separation
- Robust to various lighting conditions
- Used in successful ensemble combinations

## Why OKLAB and JzCzHz for Ensemble

Ensemble defaults chosen based on:

- **Diversity**: Different color space designs
  - Ruderman LAB: Decorrelation transform
  - OKLAB: Perceptual uniformity
  - JzCzHz: Cylindrical perceptual
- **Performance**: All top performers individually
- **Complementarity**: Different strengths on different image types

## Session Persistence

**Single Mode**: Full session save/load including models

- `class_XX_models.npz`: Per-class GMM models
- `annotations.npy`: Current scribbles
- `final_mask.npy`: Current segmentation
- `session_metadata.json`: Configuration

**Ensemble Mode**: Cannot save models (no persistence)

- Only final mask can be saved
- Each run is independent

## UI State Management

- Refine button automatically disabled in Ensemble mode
- Mode switch shows/hides appropriate configuration panels
- Status bar indicates current mode
- Progress dialog shows which color space is processing

## Backwards Compatibility

- Existing single-mode sessions can be loaded normally
- Default mode is "Single Color Space" for familiar behavior
- All existing keyboard shortcuts and controls remain the same

---

**This update brings the power of ensemble voting to the interactive application while maintaining the iterative refinement workflow for single color space mode.**
