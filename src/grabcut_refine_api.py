# Filename: grabcut_refine_api.py
# -*- coding: utf-8 -*-
"""
GrabCut Refinement API for Interactive Segmentation

This module extends the base grabcut.py functionality to support:
- Loading pre-existing GMM models (bgdModel, fgdModel)
- Iterative refinement with new scribbles using cv.GC_EVAL mode
- Model persistence for interactive workflows
- State management for multi-class segmentation sessions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2 as cv

# Import from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from color_space import convert_color_space
from mgc_api import mgc_refine_seeds, mgc_post_smooth_mask

# Fix the model path to be absolute relative to repository root
_REPO_ROOT = Path(__file__).parent.parent
_STRUCTURED_MODEL_PATH = str(_REPO_ROOT / "mgc_core" / "third_party" / "sed" / "model.yml.gz")


class SegmentationState:
    """Manages state for a single class segmentation with model persistence."""
    
    def __init__(self, class_id: int):
        self.class_id = class_id
        self.bgdModel: Optional[np.ndarray] = None
        self.fgdModel: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.iteration_count: int = 0
        
    def has_models(self) -> bool:
        """Check if GMM models exist."""
        return self.bgdModel is not None and self.fgdModel is not None
    
    def save_models(self, output_path: Path) -> None:
        """Save models to NPZ file."""
        if not self.has_models():
            raise ValueError(f"No models to save for class {self.class_id}")
        
        np.savez(
            output_path,
            bgdModel=self.bgdModel,
            fgdModel=self.fgdModel,
            current_mask=self.current_mask if self.current_mask is not None else np.array([]),
            class_id=np.array([self.class_id]),
            iteration_count=np.array([self.iteration_count])
        )
    
    def load_models(self, input_path: Path) -> None:
        """Load models from NPZ file."""
        data = np.load(input_path)
        self.bgdModel = data['bgdModel']
        self.fgdModel = data['fgdModel']
        
        if 'current_mask' in data and data['current_mask'].size > 0:
            self.current_mask = data['current_mask']
        
        if 'iteration_count' in data:
            self.iteration_count = int(data['iteration_count'])
        
        if 'class_id' in data:
            loaded_class_id = int(data['class_id'])
            if loaded_class_id != self.class_id:
                raise ValueError(f"Class ID mismatch: expected {self.class_id}, got {loaded_class_id}")


def opencv_grabcut_refine(
    img_feats_u8: np.ndarray,
    seeds_bg: np.ndarray,
    seeds_fg: np.ndarray,
    bgdModel: Optional[np.ndarray] = None,
    fgdModel: Optional[np.ndarray] = None,
    iters: int = 2,
    use_eval_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run GrabCut with support for model refinement.
    
    Args:
        img_feats_u8: HxWx3 uint8 image in any color space
        seeds_bg: HxW boolean mask for definite background
        seeds_fg: HxW boolean mask for definite foreground
        bgdModel: Optional 1x65 float64 array, pre-existing background GMM
        fgdModel: Optional 1x65 float64 array, pre-existing foreground GMM
        iters: Number of GrabCut iterations
        use_eval_mode: If True and models provided, use GC_EVAL mode for refinement
    
    Returns:
        Tuple of (binary_mask, bgdModel, fgdModel)
        - binary_mask: HxW uint8 array with 0=BG, 1=FG
        - bgdModel: Updated 1x65 float64 background GMM
        - fgdModel: Updated 1x65 float64 foreground GMM
    """
    if img_feats_u8.dtype != np.uint8:
        img_feats_u8 = np.clip(img_feats_u8, 0, 255).astype(np.uint8)
    if img_feats_u8.ndim != 3 or img_feats_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got shape {img_feats_u8.shape}")
    
    H, W, _ = img_feats_u8.shape
    
    if seeds_bg.shape != (H, W) or seeds_fg.shape != (H, W):
        raise ValueError(
            f"Seed masks must match image size, got {seeds_bg.shape} and {seeds_fg.shape}, expected {(H, W)}"
        )
    
    # Handle empty foreground seeds
    if not np.any(seeds_fg):
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    
    # Initialize mask with seeds
    mask = np.full((H, W), cv.GC_PR_BGD, dtype=np.uint8)
    mask[seeds_bg] = cv.GC_BGD
    mask[seeds_fg] = cv.GC_FGD
    
    # Initialize or use existing models
    if bgdModel is not None and fgdModel is not None:
        # Validate model shapes
        if bgdModel.shape != (1, 65) or fgdModel.shape != (1, 65):
            raise ValueError(f"Models must be shape (1, 65), got bgd={bgdModel.shape}, fgd={fgdModel.shape}")
        
        # Copy models to avoid modifying originals
        bgdModel = bgdModel.copy()
        fgdModel = fgdModel.copy()
        
        # Use eval mode for refinement if requested
        mode = cv.GC_EVAL if use_eval_mode else cv.GC_INIT_WITH_MASK
    else:
        # Initialize new models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mode = cv.GC_INIT_WITH_MASK
    
    try:
        cv.grabCut(img_feats_u8, mask, None, bgdModel, fgdModel, int(iters), mode)
    except cv.error as e:
        raise RuntimeError(f"OpenCV GrabCut failed: {e}") from e
    
    # Extract binary mask
    bin_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)
    
    return bin_mask, bgdModel, fgdModel


class MultiClassSegmentationSession:
    """
    Manages interactive multi-class segmentation with iterative refinement.
    
    Supports:
    - Multiple foreground classes (1..20)
    - Background class (0)
    - Per-class model persistence
    - Iterative refinement with new scribbles
    - Combined final segmentation map
    """
    
    def __init__(
        self,
        img_rgb: np.ndarray,
        color_space: str = "rgb",
        gc_iters: int = 5,
        tie_mode: str = "nearest-scribble",
        apply_seed_refinement: bool = True,
        apply_post_smoothing: bool = True
    ):
        """
        Initialize a segmentation session.
        
        Args:
            img_rgb: HxW RGB image as uint8
            color_space: Color space for GrabCut features (rgb, jzazbz, cielab, etc.)
            gc_iters: Iterations per GrabCut call
            tie_mode: How to resolve class overlaps ("nearest-scribble" or "first-wins")
            apply_seed_refinement: Apply MGC seed refinement before GrabCut
            apply_post_smoothing: Apply MGC post-smoothing after GrabCut
        """
        self.img_rgb = img_rgb.copy()
        self.H, self.W = img_rgb.shape[:2]
        self.color_space = color_space
        self.gc_iters = gc_iters
        self.tie_mode = tie_mode
        self.apply_seed_refinement = apply_seed_refinement
        self.apply_post_smoothing = apply_post_smoothing
        
        # Convert to feature space
        self.img_feats = convert_color_space(img_rgb, color_space)
        
        # State storage: class_id -> SegmentationState
        self.states: Dict[int, SegmentationState] = {}
        
        # Current annotation map: HxW int32, 0=unknown, 1=bg, 2..21=fg classes
        self.annotations = np.zeros((self.H, self.W), dtype=np.int32)
        
        # Current combined segmentation: HxW uint8, 0=bg, 1..20=fg classes
        self.final_mask = np.zeros((self.H, self.W), dtype=np.uint8)
    
    def update_annotations(self, new_annotations: np.ndarray) -> None:
        """
        Update annotation map with new scribbles.
        
        Args:
            new_annotations: HxW int32 array, 0=no change, 1=bg, 2..21=fg classes
                            Only non-zero values update the annotation map
        """
        if new_annotations.shape != (self.H, self.W):
            raise ValueError(f"Annotations must be {(self.H, self.W)}, got {new_annotations.shape}")
        
        # Update annotations where new_annotations is non-zero
        mask = new_annotations > 0
        self.annotations[mask] = new_annotations[mask]
    
    def get_classes(self) -> List[int]:
        """Get list of foreground classes that have scribbles."""
        unique = np.unique(self.annotations)
        return sorted([int(x) for x in unique if x > 1])
    
    def segment_class(
        self,
        class_id: int,
        force_reinit: bool = False
    ) -> np.ndarray:
        """
        Segment a single class using one-vs-rest approach.
        
        Args:
            class_id: The class to segment (2..21)
            force_reinit: If True, discard existing models and start fresh
        
        Returns:
            Binary mask for this class (0=BG, 1=FG)
        """
        if class_id < 2:
            raise ValueError(f"Class ID must be >= 2 for foreground, got {class_id}")
        
        # Get or create state
        if class_id not in self.states:
            self.states[class_id] = SegmentationState(class_id)
        
        state = self.states[class_id]
        
        if force_reinit:
            state.bgdModel = None
            state.fgdModel = None
            state.iteration_count = 0
        
        # Build seeds for this class
        seeds_fg = (self.annotations == class_id)
        seeds_bg = (self.annotations == 1) | ((self.annotations > 1) & (self.annotations != class_id))
        
        if not np.any(seeds_fg):
            # No foreground seeds for this class
            return np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Apply seed refinement if enabled
        if self.apply_seed_refinement:
            try:
                seeds_fg, seeds_bg = mgc_refine_seeds(
                    self.img_rgb,
                    seeds_bg=seeds_bg,
                    seeds_fg=seeds_fg,
                    conf_img=self.img_feats,
                    structured_model=_STRUCTURED_MODEL_PATH
                )
            except Exception as e:
                print(f"Warning: Seed refinement failed for class {class_id}: {e}")
                print(f"Model path: {_STRUCTURED_MODEL_PATH}")
                print("Continuing without seed refinement...")
        
        # Run GrabCut with or without existing models
        use_eval = state.has_models() and state.iteration_count > 0
        
        bin_mask, bgdModel, fgdModel = opencv_grabcut_refine(
            self.img_feats,
            seeds_bg=seeds_bg,
            seeds_fg=seeds_fg,
            bgdModel=state.bgdModel,
            fgdModel=state.fgdModel,
            iters=self.gc_iters,
            use_eval_mode=use_eval
        )
        
        # Update state
        state.bgdModel = bgdModel
        state.fgdModel = fgdModel
        state.iteration_count += 1
        
        # Apply post-smoothing if enabled
        if self.apply_post_smoothing:
            bin_mask = mgc_post_smooth_mask(
                self.img_rgb,
                bin_mask,
                guide_img=self.img_rgb,
                structured_model=_STRUCTURED_MODEL_PATH
            )
        
        state.current_mask = bin_mask
        
        return bin_mask
    
    def segment_all_classes(self, force_reinit: bool = False) -> np.ndarray:
        """
        Segment all classes and combine into final mask.
        
        Args:
            force_reinit: If True, reinitialize all models
        
        Returns:
            Final segmentation mask HxW uint8, 0=bg, 1..20=fg classes
        """
        classes = self.get_classes()
        
        if not classes:
            self.final_mask = np.zeros((self.H, self.W), dtype=np.uint8)
            return self.final_mask
        
        # Segment each class
        fg_masks: Dict[int, np.ndarray] = {}
        for c in classes:
            fg_masks[c] = self.segment_class(c, force_reinit=force_reinit)
        
        # Combine masks
        self.final_mask = self._combine_masks(fg_masks)
        
        return self.final_mask
    
    def _combine_masks(self, fg_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Combine per-class binary masks into final indexed mask.
        
        Uses nearest-scribble distance for overlap resolution.
        """
        classes = sorted(fg_masks.keys())
        if not classes:
            return np.zeros((self.H, self.W), dtype=np.uint8)
        
        final = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Stack all masks to detect overlaps
        stack = np.stack([fg_masks[c] for c in classes], axis=2)
        overlap_count = stack.sum(axis=2)
        
        # No overlaps or not using nearest-scribble
        if not (overlap_count > 1).any() or self.tie_mode != "nearest-scribble":
            for c in classes:
                m = fg_masks[c] > 0
                final[m] = c  # Preserve class ID as-is in output
            return final
        
        # Resolve overlaps using nearest-scribble
        overlap_mask = (overlap_count > 1)
        
        dist_to_scrib: Dict[int, np.ndarray] = {}
        classes_for_dt: List[int] = []
        
        for c in classes:
            if np.any(fg_masks[c] & overlap_mask):
                s = (self.annotations == c).astype(np.uint8)
                if np.any(s):
                    ones = np.ones_like(s, dtype=np.uint8)
                    ones[s > 0] = 0
                    d = cv.distanceTransform(ones, cv.DIST_L2, 3).astype(np.float32)
                else:
                    d = np.full(s.shape, 1e6, dtype=np.float32)
                dist_to_scrib[c] = d
                classes_for_dt.append(c)
        
        if classes_for_dt:
            INF = 1e9
            dstack = np.stack(
                [np.where(fg_masks[c] > 0, dist_to_scrib[c], INF) for c in classes_for_dt],
                axis=2
            )
            arg = np.argmin(dstack, axis=2)
            
            # Assign non-overlapping pixels
            for c in classes:
                m = (fg_masks[c] > 0) & (~overlap_mask)
                final[m] = c
            
            # Assign overlapping pixels to nearest scribble
            for idx, c in enumerate(classes_for_dt):
                m = overlap_mask & (arg == idx)
                final[m] = c
        else:
            for c in classes:
                m = fg_masks[c] > 0
                final[m] = c
        
        return final
    
    def save_session(self, output_dir: Path) -> None:
        """Save session state including all models and annotations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotations
        np.save(output_dir / "annotations.npy", self.annotations)
        
        # Save final mask
        if self.final_mask is not None:
            np.save(output_dir / "final_mask.npy", self.final_mask)
        
        # Save per-class models
        for class_id, state in self.states.items():
            if state.has_models():
                state.save_models(output_dir / f"class_{class_id:02d}_models.npz")
        
        # Save metadata
        metadata = {
            "image_shape": [self.H, self.W],
            "color_space": self.color_space,
            "gc_iters": self.gc_iters,
            "tie_mode": self.tie_mode,
            "classes": list(self.states.keys()),
            "iteration_counts": {c: self.states[c].iteration_count for c in self.states}
        }
        
        with open(output_dir / "session_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_session(self, input_dir: Path) -> None:
        """Load session state from saved files."""
        input_dir = Path(input_dir)
        
        # Load annotations
        self.annotations = np.load(input_dir / "annotations.npy")
        
        # Load final mask if exists
        final_mask_path = input_dir / "final_mask.npy"
        if final_mask_path.exists():
            self.final_mask = np.load(final_mask_path)
        
        # Load metadata
        with open(input_dir / "session_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load per-class models
        for class_id in metadata["classes"]:
            model_path = input_dir / f"class_{class_id:02d}_models.npz"
            if model_path.exists():
                state = SegmentationState(class_id)
                state.load_models(model_path)
                self.states[class_id] = state


def majority_vote_indexed(a: np.ndarray, b: np.ndarray, c: np.ndarray, tie_pref: int = 0) -> np.ndarray:
    """
    Majority vote over three 2D uint8 indexed masks.
    
    Args:
        a, b, c: Three HxW uint8 masks with class labels
        tie_pref: For three-way ties, choose 0=first, 1=second, 2=third
    
    Returns:
        HxW uint8 mask with majority-voted labels
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(f"Shape mismatch: {a.shape}, {b.shape}, {c.shape}")
    
    a = a.astype("uint8", copy=False)
    b = b.astype("uint8", copy=False)
    c = c.astype("uint8", copy=False)
    
    # Check pairwise equality
    eq_ab = (a == b)
    eq_ac = (a == c)
    eq_bc = (b == c)
    
    out = a.copy()
    
    # If a==b, use a
    mask_ab = eq_ab
    out[mask_ab] = a[mask_ab]
    
    # If a==c (but not a==b), use a
    mask_ac = eq_ac & (~mask_ab)
    out[mask_ac] = a[mask_ac]
    
    # If b==c (but not a==b and not a==c), use b
    mask_bc = eq_bc & (~(mask_ab | mask_ac))
    out[mask_bc] = b[mask_bc]
    
    # Three-way tie (all different)
    mask_tie = ~(mask_ab | mask_ac | mask_bc)
    if mask_tie.any():
        if tie_pref == 0:
            out[mask_tie] = a[mask_tie]
        elif tie_pref == 1:
            out[mask_tie] = b[mask_tie]
        else:
            out[mask_tie] = c[mask_tie]
    
    return out


def segment_all_classes_ensemble(
    img_rgb: np.ndarray,
    annotations: np.ndarray,
    color_spaces: List[str],
    gc_iters: int = 5,
    tie_mode: str = "nearest-scribble",
    apply_seed_refinement: bool = True,
    apply_post_smoothing: bool = True,
    label_tie_pref: int = 0
) -> np.ndarray:
    """
    Run ensemble segmentation with majority voting over multiple color spaces.
    
    This is for initial segmentation only (no model reuse across color spaces).
    DEPRECATED: Use EnsembleSegmentationSession for refinement support.
    
    Args:
        img_rgb: HxW RGB image as uint8
        annotations: HxW int32 annotation map
        color_spaces: List of 3 color space names for ensemble
        gc_iters: GrabCut iterations per color space
        tie_mode: Overlap resolution strategy
        apply_seed_refinement: Apply MGC seed refinement
        apply_post_smoothing: Apply MGC post-smoothing
        label_tie_pref: Tie-breaking preference (0=first, 1=second, 2=third)
    
    Returns:
        HxW uint8 final segmentation mask
    """
    if len(color_spaces) != 3:
        raise ValueError(f"Ensemble requires exactly 3 color spaces, got {len(color_spaces)}")
    
    masks = []
    
    for cs in color_spaces:
        # Create temporary session for this color space
        session = MultiClassSegmentationSession(
            img_rgb,
            color_space=cs,
            gc_iters=gc_iters,
            tie_mode=tie_mode,
            apply_seed_refinement=apply_seed_refinement,
            apply_post_smoothing=apply_post_smoothing
        )
        session.update_annotations(annotations)
        mask = session.segment_all_classes(force_reinit=True)
        masks.append(mask)
    
    # Majority vote
    final_mask = majority_vote_indexed(masks[0], masks[1], masks[2], tie_pref=label_tie_pref)
    
    return final_mask


class EnsembleSegmentationSession:
    """
    Manages ensemble segmentation with iterative refinement support.
    
    Maintains three separate MultiClassSegmentationSession instances,
    one for each color space, enabling model persistence and refinement
    across all three ensemble members.
    """
    
    def __init__(
        self,
        img_rgb: np.ndarray,
        color_spaces: List[str],
        gc_iters: int = 5,
        tie_mode: str = "nearest-scribble",
        apply_seed_refinement: bool = True,
        apply_post_smoothing: bool = True,
        label_tie_pref: int = 0
    ):
        """
        Initialize ensemble session.
        
        Args:
            img_rgb: HxW RGB image as uint8
            color_spaces: List of 3 color space names
            gc_iters: GrabCut iterations per color space
            tie_mode: Overlap resolution strategy
            apply_seed_refinement: Apply MGC seed refinement
            apply_post_smoothing: Apply MGC post-smoothing
            label_tie_pref: Tie-breaking preference (0=first, 1=second, 2=third)
        """
        if len(color_spaces) != 3:
            raise ValueError(f"Ensemble requires exactly 3 color spaces, got {len(color_spaces)}")
        
        self.img_rgb = img_rgb.copy()
        self.H, self.W = img_rgb.shape[:2]
        self.color_spaces = color_spaces
        self.gc_iters = gc_iters
        self.tie_mode = tie_mode
        self.apply_seed_refinement = apply_seed_refinement
        self.apply_post_smoothing = apply_post_smoothing
        self.label_tie_pref = label_tie_pref
        
        # Create three separate sessions, one per color space
        self.sessions: List[MultiClassSegmentationSession] = []
        for cs in color_spaces:
            session = MultiClassSegmentationSession(
                img_rgb,
                color_space=cs,
                gc_iters=gc_iters,
                tie_mode=tie_mode,
                apply_seed_refinement=apply_seed_refinement,
                apply_post_smoothing=apply_post_smoothing
            )
            self.sessions.append(session)
        
        # Current annotation map (shared across all sessions)
        self.annotations = np.zeros((self.H, self.W), dtype=np.int32)
        
        # Final ensemble mask
        self.final_mask = np.zeros((self.H, self.W), dtype=np.uint8)
    
    def update_annotations(self, new_annotations: np.ndarray) -> None:
        """
        Update annotations for all sessions.
        
        Args:
            new_annotations: HxW int32 array with scribble annotations
        """
        if new_annotations.shape != (self.H, self.W):
            raise ValueError(f"Annotations must be {(self.H, self.W)}, got {new_annotations.shape}")
        
        # Update shared annotations
        mask = new_annotations > 0
        self.annotations[mask] = new_annotations[mask]
        
        # Update all sessions
        for session in self.sessions:
            session.update_annotations(new_annotations)
    
    def segment_all_classes(self, force_reinit: bool = False) -> np.ndarray:
        """
        Run ensemble segmentation across all three color spaces.
        
        Args:
            force_reinit: If True, reinitialize all models. If False, refine existing models.
        
        Returns:
            HxW uint8 final ensemble mask
        """
        masks = []
        
        # Segment with each color space session
        for session in self.sessions:
            mask = session.segment_all_classes(force_reinit=force_reinit)
            masks.append(mask)
        
        # Majority vote
        self.final_mask = majority_vote_indexed(
            masks[0], masks[1], masks[2], 
            tie_pref=self.label_tie_pref
        )
        
        return self.final_mask
    
    def get_classes(self) -> List[int]:
        """Get list of foreground classes with scribbles."""
        unique = np.unique(self.annotations)
        return sorted([int(x) for x in unique if x > 1])
    
    def update_settings(
        self,
        gc_iters: Optional[int] = None,
        apply_seed_refinement: Optional[bool] = None,
        apply_post_smoothing: Optional[bool] = None,
        label_tie_pref: Optional[int] = None
    ) -> None:
        """
        Update segmentation settings.
        
        Args:
            gc_iters: New iteration count
            apply_seed_refinement: Enable/disable seed refinement
            apply_post_smoothing: Enable/disable post-smoothing
            label_tie_pref: New tie-breaking preference
        """
        if gc_iters is not None:
            self.gc_iters = gc_iters
            for session in self.sessions:
                session.gc_iters = gc_iters
        
        if apply_seed_refinement is not None:
            self.apply_seed_refinement = apply_seed_refinement
            for session in self.sessions:
                session.apply_seed_refinement = apply_seed_refinement
        
        if apply_post_smoothing is not None:
            self.apply_post_smoothing = apply_post_smoothing
            for session in self.sessions:
                session.apply_post_smoothing = apply_post_smoothing
        
        if label_tie_pref is not None:
            self.label_tie_pref = label_tie_pref
    
    def save_session(self, output_dir: Path) -> None:
        """Save ensemble session including all three color space sessions."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save shared annotations
        np.save(output_dir / "annotations.npy", self.annotations)
        
        # Save final mask
        if self.final_mask is not None:
            np.save(output_dir / "final_mask.npy", self.final_mask)
        
        # Save each color space session
        for idx, (session, cs) in enumerate(zip(self.sessions, self.color_spaces)):
            cs_dir = output_dir / f"ensemble_{idx}_{cs}"
            session.save_session(cs_dir)
        
        # Save ensemble metadata
        metadata = {
            "image_shape": [self.H, self.W],
            "color_spaces": self.color_spaces,
            "gc_iters": self.gc_iters,
            "tie_mode": self.tie_mode,
            "label_tie_pref": self.label_tie_pref,
            "apply_seed_refinement": self.apply_seed_refinement,
            "apply_post_smoothing": self.apply_post_smoothing
        }
        
        with open(output_dir / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_session(self, input_dir: Path) -> None:
        """Load ensemble session from saved files."""
        input_dir = Path(input_dir)
        
        # Load shared annotations
        self.annotations = np.load(input_dir / "annotations.npy")
        
        # Load final mask if exists
        final_mask_path = input_dir / "final_mask.npy"
        if final_mask_path.exists():
            self.final_mask = np.load(final_mask_path)
        
        # Load metadata
        with open(input_dir / "ensemble_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load each color space session
        for idx, (session, cs) in enumerate(zip(self.sessions, self.color_spaces)):
            cs_dir = input_dir / f"ensemble_{idx}_{cs}"
            if cs_dir.exists():
                session.load_session(cs_dir)
