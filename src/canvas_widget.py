# Filename: canvas_widget.py
# -*- coding: utf-8 -*-
"""
Interactive Canvas Widget for Scribble Drawing

Supports:
- Drawing scribbles with different class labels
- Pan and zoom
- Undo/redo
- Eraser mode
- Adjustable brush size
- Overlay of segmentation results
"""

from typing import Optional, List, Tuple
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint, QRect, Signal, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QImage, QPixmap, 
    QPaintEvent, QMouseEvent, QWheelEvent, QPainterPath
)


class ScribbleLayer:
    """Represents a single scribble stroke."""
    
    def __init__(self, class_id: int, color: QColor, points: List[QPoint], brush_size: int):
        self.class_id = class_id
        self.color = color
        # Store points in IMAGE coordinates (not view coordinates)
        self.points = points.copy()
        self.brush_size = brush_size


class CanvasWidget(QWidget):
    """Interactive canvas for drawing scribbles and viewing segmentation."""
    
    # Signals
    scribbles_changed = Signal()  # Emitted when scribbles are modified
    
    # Default class color palette (VOC-style colors)
    _DEFAULT_COLORS = [
        QColor(0, 0, 0),         # 0: Background (black)
        QColor(128, 0, 0),       # 1: Background marker (maroon)
        QColor(0, 128, 0),       # 2: Class 1 (green)
        QColor(128, 128, 0),     # 3: Class 2 (olive)
        QColor(0, 0, 128),       # 4: Class 3 (navy)
        QColor(128, 0, 128),     # 5: Class 4 (purple)
        QColor(0, 128, 128),     # 6: Class 5 (teal)
        QColor(128, 128, 128),   # 7: Class 6 (gray)
        QColor(64, 0, 0),        # 8: Class 7
        QColor(192, 0, 0),       # 9: Class 8
        QColor(64, 128, 0),      # 10: Class 9
        QColor(192, 128, 0),     # 11: Class 10
        QColor(64, 0, 128),      # 12: Class 11
        QColor(192, 0, 128),     # 13: Class 12
        QColor(64, 128, 128),    # 14: Class 13
        QColor(192, 128, 128),   # 15: Class 14
        QColor(0, 64, 0),        # 16: Class 15
        QColor(128, 64, 0),      # 17: Class 16
        QColor(0, 192, 0),       # 18: Class 17
        QColor(128, 192, 0),     # 19: Class 18
        QColor(0, 64, 128),      # 20: Class 19
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Dynamic class colors dictionary: annotation_class_id -> QColor
        self.class_colors = {0: self._DEFAULT_COLORS[0], 1: self._DEFAULT_COLORS[1]}  # BG and BG marker
        
        # Image data
        self.original_image: Optional[QImage] = None
        self.image_array: Optional[np.ndarray] = None
        
        # Segmentation overlay
        self.segmentation_mask: Optional[np.ndarray] = None
        self.show_segmentation = True
        self.segmentation_opacity = 0.5
        
        # Scribble layers
        self.scribbles: List[ScribbleLayer] = []
        self.undo_stack: List[List[ScribbleLayer]] = []
        self.redo_stack: List[List[ScribbleLayer]] = []
        
        # Current drawing session scribbles (visible in UI)
        self.current_session_scribbles: List[ScribbleLayer] = []
        
        # Drawing state
        self.is_drawing = False
        self.current_stroke_points: List[QPoint] = []
        self.current_class = 1  # Start with background by default
        self.brush_size = 3 # Taken from the average thickness of scribbles in the s4Pascal dataset
        self.eraser_mode = False
        
        # View transformation
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_point = QPoint()
        
        # Widget settings
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
    
    def set_image(self, image_array: np.ndarray) -> None:
        """
        Set the image to display.
        
        Args:
            image_array: HxWx3 RGB numpy array (uint8)
        """
        self.image_array = image_array.copy()
        h, w = image_array.shape[:2]
        
        # Convert numpy array to QImage
        bytes_per_line = 3 * w
        self.original_image = QImage(
            image_array.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        
        # Reset view
        self.reset_view()
        
        # Clear all scribbles (both accumulated and current session)
        self.clear_scribbles(emit_signal=False)
        self.current_session_scribbles.clear()
        
        self.update()
    
    def set_segmentation_mask(self, mask: Optional[np.ndarray]) -> None:
        """
        Set segmentation mask overlay.
        
        Args:
            mask: HxW uint8 array with class labels 0..20, or None to clear
        """
        self.segmentation_mask = mask
        self.update()
    
    def clear_segmentation(self) -> None:
        """Clear the segmentation overlay."""
        self.segmentation_mask = None
        self.update()
    
    def set_show_segmentation(self, show: bool) -> None:
        """Toggle segmentation overlay visibility."""
        self.show_segmentation = show
        self.update()
    
    def set_segmentation_opacity(self, opacity: float) -> None:
        """Set segmentation overlay opacity (0.0 to 1.0)."""
        self.segmentation_opacity = max(0.0, min(1.0, opacity))
        self.update()
    
    def commit_scribbles_after_segmentation(self) -> None:
        """
        Commit current session scribbles to permanent storage and clear the visible session.
        Called after segmentation/refinement to hide old scribbles and make room for new ones.
        The scribbles remain in self.scribbles for GrabCut but disappear from UI.
        """
        # Current session scribbles are already in self.scribbles (for GrabCut)
        # Just clear the visible session to make room for new scribbles
        self.current_session_scribbles.clear()
        self.update()
    
    def set_current_class(self, class_id: int) -> None:
        """Set the current class for drawing (0=bg, 1=bg_marker, 2+=fg_classes)."""
        self.current_class = max(0, class_id)
    
    def add_class_color(self, class_id: int, color: QColor) -> None:
        """Add or update a class color."""
        self.class_colors[class_id] = color
        self.update()
    
    def get_class_color(self, class_id: int) -> QColor:
        """Get color for a class, or default if not defined."""
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        # Generate a default color if not defined
        if class_id < len(self._DEFAULT_COLORS):
            return self._DEFAULT_COLORS[class_id]
        # Generate a random-ish color for high class IDs
        import random
        random.seed(class_id)
        return QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def set_brush_size(self, size: int) -> None:
        """Set brush size in pixels."""
        self.brush_size = max(1, min(50, size))
    
    def set_eraser_mode(self, enabled: bool) -> None:
        """Toggle eraser mode."""
        self.eraser_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
    
    def clear_scribbles(self, emit_signal: bool = True) -> None:
        """Clear all scribbles."""
        if self.scribbles or self.current_session_scribbles:
            # Save current accumulated scribbles to undo stack
            self.undo_stack.append([s for s in self.scribbles])

            # Clear both accumulated and current-session (visible) scribbles
            self.scribbles.clear()
            self.current_session_scribbles.clear()

            # Clear redo stack and update UI
            self.redo_stack.clear()
            if emit_signal:
                self.scribbles_changed.emit()
            self.update()
    
    def undo(self) -> None:
        """Undo last scribble action."""
        if self.undo_stack:
            self.redo_stack.append([s for s in self.scribbles])
            self.scribbles = self.undo_stack.pop()
            self.scribbles_changed.emit()
            self.update()
    
    def redo(self) -> None:
        """Redo last undone action."""
        if self.redo_stack:
            self.undo_stack.append([s for s in self.scribbles])
            self.scribbles = self.redo_stack.pop()
            self.scribbles_changed.emit()
            self.update()
    
    def reset_view(self) -> None:
        """Reset zoom and pan to fit image."""
        if self.original_image is None:
            return
        
        # Fit image to widget
        img_w = self.original_image.width()
        img_h = self.original_image.height()
        widget_w = self.width()
        widget_h = self.height()
        
        zoom_w = widget_w / img_w if img_w > 0 else 1.0
        zoom_h = widget_h / img_h if img_h > 0 else 1.0
        
        self.zoom_factor = min(zoom_w, zoom_h, 1.0) * 0.9  # 90% to add padding
        self.pan_offset = QPoint(0, 0)
        self.update()
    
    def get_annotation_map(self) -> np.ndarray:
        """
        Convert scribbles to annotation map.
        
        Returns:
            HxW int32 array with class labels where values match the class_id directly
            (1=background, 2=first_fg, 3=second_fg, etc.)
        """
        if self.image_array is None:
            return np.array([[]], dtype=np.int32)
        
        h, w = self.image_array.shape[:2]
        annotations = np.zeros((h, w), dtype=np.int32)
        
        for scribble in self.scribbles:
            class_id = scribble.class_id  # Use class_id directly (1, 2, 3...)
            
            # Rasterize the stroke (points are already in image coordinates)
            for i in range(len(scribble.points) - 1):
                p1 = scribble.points[i]
                p2 = scribble.points[i + 1]
                
                # Draw line on annotations
                self._draw_line_on_array(
                    annotations, p1, p2, class_id, scribble.brush_size
                )
        
        return annotations
    
    def _draw_line_on_array(
        self, 
        arr: np.ndarray, 
        p1: QPoint, 
        p2: QPoint, 
        value: int, 
        thickness: int
    ) -> None:
        """Draw a line on numpy array."""
        h, w = arr.shape[:2]
        
        # Convert to numpy coordinates
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        
        # Clip to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        
        # Bresenham's line algorithm with thickness
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        radius = thickness // 2
        
        while True:
            # Draw circle at current point
            for dy_offset in range(-radius, radius + 1):
                for dx_offset in range(-radius, radius + 1):
                    if dx_offset * dx_offset + dy_offset * dy_offset <= radius * radius:
                        nx = x1 + dx_offset
                        ny = y1 + dy_offset
                        if 0 <= nx < w and 0 <= ny < h:
                            arr[ny, nx] = value
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def view_to_image_coords(self, view_point: QPoint) -> QPoint:
        """Convert view coordinates to image coordinates."""
        if self.original_image is None:
            return QPoint(0, 0)
        
        # Account for centering
        img_w = self.original_image.width() * self.zoom_factor
        img_h = self.original_image.height() * self.zoom_factor
        
        offset_x = (self.width() - img_w) / 2 + self.pan_offset.x()
        offset_y = (self.height() - img_h) / 2 + self.pan_offset.y()
        
        # Convert to image space
        img_x = int((view_point.x() - offset_x) / self.zoom_factor)
        img_y = int((view_point.y() - offset_y) / self.zoom_factor)
        
        return QPoint(img_x, img_y)
    
    def image_to_view_coords(self, image_point: QPoint) -> QPoint:
        """Convert image coordinates to view coordinates."""
        if self.original_image is None:
            return QPoint(0, 0)
        
        img_w = self.original_image.width() * self.zoom_factor
        img_h = self.original_image.height() * self.zoom_factor
        
        offset_x = (self.width() - img_w) / 2 + self.pan_offset.x()
        offset_y = (self.height() - img_h) / 2 + self.pan_offset.y()
        
        view_x = int(image_point.x() * self.zoom_factor + offset_x)
        view_y = int(image_point.y() * self.zoom_factor + offset_y)
        
        return QPoint(view_x, view_y)
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        if self.original_image is None:
            return
        
        # Calculate image position (centered)
        img_w = int(self.original_image.width() * self.zoom_factor)
        img_h = int(self.original_image.height() * self.zoom_factor)
        
        x = int((self.width() - img_w) / 2 + self.pan_offset.x())
        y = int((self.height() - img_h) / 2 + self.pan_offset.y())
        
        # Draw image
        target_rect = QRect(x, y, img_w, img_h)
        painter.drawImage(target_rect, self.original_image)
        
        # Draw segmentation overlay
        if self.show_segmentation and self.segmentation_mask is not None:
            self._draw_segmentation_overlay(painter, target_rect)
        
        # Draw scribbles (only current session - visible scribbles)
        for scribble in self.current_session_scribbles:
            self._draw_scribble(painter, scribble)
        
        # Draw current stroke (always show while drawing)
        if self.is_drawing and len(self.current_stroke_points) > 1:
            color = self.get_class_color(self.current_class) if not self.eraser_mode else QColor(255, 255, 255)
            pen = QPen(color, self.brush_size * self.zoom_factor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            
            # Convert image coordinates to view coordinates for drawing
            path = QPainterPath()
            view_pt = self.image_to_view_coords(self.current_stroke_points[0])
            path.moveTo(view_pt)
            for img_pt in self.current_stroke_points[1:]:
                view_pt = self.image_to_view_coords(img_pt)
                path.lineTo(view_pt)
            painter.drawPath(path)
    
    def _draw_segmentation_overlay(self, painter: QPainter, target_rect: QRect) -> None:
        """Draw segmentation mask as colored overlay."""
        if self.segmentation_mask is None or self.image_array is None:
            return
        
        h, w = self.segmentation_mask.shape
        
        # Create colored overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Segmentation mask values are labels = class_id - 1
        # mask_value 0 -> background (skipped)
        # mask_value 1 -> class_id 2 (first foreground)
        # mask_value 2 -> class_id 3 (second foreground)
        # etc.
        unique_values = np.unique(self.segmentation_mask)
        for mask_value in unique_values:
            if mask_value == 0:  # Skip zero (background / no segmentation)
                continue
            mask = (self.segmentation_mask == mask_value)
            if np.any(mask):
                # Map label back to class_id: label + 1 = class_id
                class_id = int(mask_value) + 1
                color = self.get_class_color(class_id)
                overlay[mask] = [color.red(), color.green(), color.blue(), 
                                int(255 * self.segmentation_opacity)]
        
        # Convert to QImage
        bytes_per_line = 4 * w
        overlay_img = QImage(overlay.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        
        # Draw overlay
        painter.drawImage(target_rect, overlay_img)
    
    def _draw_scribble(self, painter: QPainter, scribble: ScribbleLayer) -> None:
        """Draw a scribble stroke (converting from image to view coordinates)."""
        if len(scribble.points) < 2:
            return
        
        pen = QPen(scribble.color, scribble.brush_size * self.zoom_factor, 
                  Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        
        # Convert image coordinates to view coordinates for drawing
        path = QPainterPath()
        view_pt = self.image_to_view_coords(scribble.points[0])
        path.moveTo(view_pt)
        for img_pt in scribble.points[1:]:
            view_pt = self.image_to_view_coords(img_pt)
            path.lineTo(view_pt)
        painter.drawPath(path)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # Start drawing - store in IMAGE coordinates
            self.is_drawing = True
            img_pos = self.view_to_image_coords(event.pos())
            self.current_stroke_points = [img_pos]
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        if self.is_panning:
            # Update pan
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update()
        elif self.is_drawing:
            # Continue stroke - store in IMAGE coordinates
            img_pos = self.view_to_image_coords(event.pos())
            self.current_stroke_points.append(img_pos)
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            # Stop panning
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self.is_drawing:
            # Finish stroke
            self.is_drawing = False
            
            if len(self.current_stroke_points) > 1:
                # Save to undo stack
                self.undo_stack.append([s for s in self.scribbles])
                self.redo_stack.clear()
                
                # Create scribble layer (points already in IMAGE coordinates)
                if self.eraser_mode:
                    # Remove scribbles near eraser path
                    self._erase_scribbles()
                else:
                    # Add new scribble to both permanent storage and current session
                    color = self.get_class_color(self.current_class)
                    scribble = ScribbleLayer(
                        self.current_class, color, 
                        self.current_stroke_points, self.brush_size
                    )
                    self.scribbles.append(scribble)  # For GrabCut refinement
                    self.current_session_scribbles.append(scribble)  # For UI visibility
                
                self.current_stroke_points.clear()
                self.scribbles_changed.emit()
                self.update()
    
    def _erase_scribbles(self) -> None:
        """Remove scribbles near the eraser path (in image coordinates)."""
        # Simple implementation: remove scribbles that overlap with eraser path
        # More sophisticated version could partially erase strokes
        
        eraser_radius = self.brush_size  # Work in image space
        
        to_remove = []
        removed_objects = set()
        for idx, scribble in enumerate(self.scribbles):
            for scribble_pt in scribble.points:
                for eraser_pt in self.current_stroke_points:
                    # Both are now in image coordinates
                    dx = scribble_pt.x() - eraser_pt.x()
                    dy = scribble_pt.y() - eraser_pt.y()
                    dist = (dx * dx + dy * dy) ** 0.5
                    
                    if dist < eraser_radius:
                        to_remove.append(idx)
                        removed_objects.add(scribble)
                        break
                if idx in to_remove:
                    break
        
        # Remove in reverse order to maintain indices
        for idx in sorted(set(to_remove), reverse=True):
            del self.scribbles[idx]

        # Also remove the same scribble objects from current session visibility
        if removed_objects:
            self.current_session_scribbles = [s for s in self.current_session_scribbles if s not in removed_objects]
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if self.original_image is None:
            return
        
        # Get zoom delta
        delta = event.angleDelta().y()
        zoom_change = 1.1 if delta > 0 else 0.9
        
        # Get mouse position before zoom
        mouse_pos = event.position()
        old_image_pos = self.view_to_image_coords(mouse_pos.toPoint())
        
        # Apply zoom
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_change
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        # Adjust pan to keep mouse position stable
        new_image_pos_view = self.image_to_view_coords(old_image_pos)
        delta_x = mouse_pos.x() - new_image_pos_view.x()
        delta_y = mouse_pos.y() - new_image_pos_view.y()
        self.pan_offset += QPoint(int(delta_x), int(delta_y))
        
        self.update()
    
    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
            self.redo()
        elif event.key() == Qt.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)
