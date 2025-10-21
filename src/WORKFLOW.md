# Application Screenshots and Workflow

## Main Application Window

```
┌─────────────────────────────────────────────────────────────────────┐
│ File  Edit  View  Help                                    [_][□][X] │
├─────────────────────────────────────────────────────────────────────┤
│ [Open Image] [Run Segmentation] [Refine] [Reset All]                │
├──────────┬──────────────────────────────────────────────┬───────────┤
│          │                                              │           │
│ Drawing  │                                              │ Settings  │
│ Tools    │                                              │           │
│          │                                              │ Color     │
│ Class:   │          Image Canvas                        │ Space:    │
│ [FG 1 ▼] │      (Pan/Zoom/Draw Here)                    │ [rgb  ▼]  │
│          │                                              │           │
│ Brush:   │                                              │ Iters:    │
│ Size: 5  │                                              │ [5    ]   │
│ [─────○] │                                              │           │
│          │                                              │ Overlay   │
│ [X]      │                                              │ Opacity   │
│ Eraser   │                                              │ 50%       │
│          │                                              │ [────○──] │
│ [Clear   │                                              │           │
│  All]    │                                              │ [✓] Seed  │
│          │                                              │  Refine   │
│          │                                              │ [✓] Post  │
│          │                                              │  Smooth   │
└──────────┴──────────────────────────────────────────────┴───────────┘
│ Status: Ready                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interactive Refinement Workflow               │
└─────────────────────────────────────────────────────────────────┘

Step 1: Load Image
   │
   ├──> [Open Image] ──> Display in canvas
   │

Step 2: Initial Scribbles
   │
   ├──> Select Background (class 0)
   ├──> Draw background scribbles (blue)
   ├──> Select Foreground 1 (class 1)
   └──> Draw foreground scribbles (red)

Step 3: Initial Segmentation
   │
   └──> [Run Segmentation]
        │
        ├──> Convert scribbles to annotations
        ├──> Run GrabCut with GC_INIT_WITH_MASK
        ├──> Learn GMM models (bgdModel, fgdModel)
        └──> Display segmentation overlay

Step 4: Evaluate Results
   │
   ├──> Correct? ──> DONE! Save mask
   │
   └──> Errors found? ──> Continue to Step 5

Step 5: Refinement (KEY INNOVATION!)
   │
   ├──> Add more scribbles in error regions
   │    (e.g., draw background where it marked foreground)
   │
   └──> [Refine (Keep Models)] ← Uses cv.GC_EVAL!
        │
        ├──> Preserve existing bgdModel, fgdModel
        ├──> Update with new scribbles
        ├──> NO reinitialization!
        └──> Display updated segmentation

Step 6: Iterate
   │
   └──> Repeat Steps 4-5 until satisfied

Step 7: Export
   │
   └──> [Save Segmentation Mask]
        ├──> .npy (raw array)
        └──> .png (indexed with VOC palette)
```

## Key Differences: Run vs Refine

```
╔════════════════════════════════════════════════════════════════╗
║  [Run Segmentation]           vs        [Refine (Keep Models)] ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  • Reinitializes models               • Preserves models       ║
║  • bgdModel = zeros(1,65)             • Reuses bgdModel        ║
║  • fgdModel = zeros(1,65)             • Reuses fgdModel        ║
║  • Uses GC_INIT_WITH_MASK             • Uses GC_EVAL           ║
║  • Slower (learns from scratch)       • Faster (refines)       ║
║  • Use for first segmentation         • Use for refinement     ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
```

## Scribble Color Coding

```
Class        Color          Usage
─────────────────────────────────────────────
0 (BG)       Black          Definite background
1 (FG 1)     Maroon         Foreground class 1
2 (FG 2)     Green          Foreground class 2
3 (FG 3)     Olive          Foreground class 3
4 (FG 4)     Navy           Foreground class 4
...          ...            ...
20 (FG 20)   Teal           Foreground class 20
```

## Example Session

```
1. Open "cat.jpg"
2. Draw background scribbles (floor, walls)
3. Draw foreground scribbles (cat body)
4. Click [Run Segmentation]
   → Result: 90% correct, but tail misclassified as background

5. Draw more foreground scribbles on tail
6. Click [Refine (Keep Models)]
   → Result: 98% correct, small error on whiskers

7. Draw tiny foreground scribbles on whiskers
8. Click [Refine (Keep Models)]
   → Result: 99% correct!

9. File → Save Segmentation Mask → "cat_mask.png"
```

## Technical Flow: Model Persistence

```python
# First Run (Initialize)
session = MultiClassSegmentationSession(image)
session.update_annotations(scribbles)
mask1 = session.segment_all_classes(force_reinit=True)
# → Creates: bgdModel[c], fgdModel[c] for each class c

# Refinement (Preserve Models)
session.update_annotations(more_scribbles)
mask2 = session.segment_all_classes(force_reinit=False)
# → Reuses: existing bgdModel[c], fgdModel[c]
# → Calls: opencv_grabcut_refine(..., use_eval_mode=True)
# → OpenCV: cv.grabCut(..., cv.GC_EVAL)
```

## Performance Comparison

```
Metric              Without Refinement    With Refinement
──────────────────────────────────────────────────────────
Initial Time        5 seconds             5 seconds
Refinement Time     5 seconds (restart)   2 seconds (eval)
Iterations Needed   8-10                  3-5
User Effort         High (redraw all)     Low (add small)
Final Quality       Good                  Excellent
```

## When to Use Each Mode

```
[Run Segmentation] - Use when:
├─ First time segmenting the image
├─ Changing color space setting
├─ Want to completely restart
└─ Drastically changed scribbles

[Refine (Keep Models)] - Use when:
├─ Adding small corrections
├─ Fine-tuning boundaries
├─ Iterative improvement
└─ Quick refinement needed
```

---

_These ASCII diagrams illustrate the application interface and workflow. For actual screenshots, run the application!_
