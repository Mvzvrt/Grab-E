# Filename: main_window.py
# -*- coding: utf-8 -*-
"""
Main Window for Interactive GrabCut Segmentation Application

Provides:
- Image loading
- Scribble drawing with multiple classes
- Interactive segmentation with GrabCut
- Iterative refinement
- Export results
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QSlider, QComboBox, QFileDialog, QMessageBox,
    QToolBar, QStatusBar, QDockWidget, QGroupBox, QCheckBox,
    QProgressDialog, QSplitter, QColorDialog, QInputDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QAction, QKeySequence, QIcon, QColor
from PIL import Image

from canvas_widget import CanvasWidget
from grabcut_refine_api import (
    MultiClassSegmentationSession, 
    EnsembleSegmentationSession,
    segment_all_classes_ensemble
)
from utils import voc_palette

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from color_space import convert_color_space


class SegmentationWorker(QThread):
    """Worker thread for running segmentation."""
    
    finished = Signal(np.ndarray)  # Emits final mask
    error = Signal(str)  # Emits error message
    progress = Signal(int, str)  # Emits (percentage, status message)
    
    def __init__(self, session: MultiClassSegmentationSession, force_reinit: bool = False, mode: str = "single"):
        super().__init__()
        self.session = session
        self.force_reinit = force_reinit
        self.mode = mode
    
    def run(self):
        """Run segmentation in background."""
        try:
            classes = self.session.get_classes()
            if not classes:
                self.error.emit("No foreground scribbles found. Please draw some scribbles first.")
                return
            
            total = len(classes)
            
            for idx, class_id in enumerate(classes):
                self.progress.emit(
                    int((idx / total) * 100),
                    f"Segmenting class {class_id - 1}..."
                )
                self.session.segment_class(class_id, force_reinit=self.force_reinit)
            
            self.progress.emit(95, "Combining results...")
            
            # Combine all masks
            final_mask = self.session.segment_all_classes(force_reinit=False)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(final_mask)
            
        except Exception as e:
            self.error.emit(f"Segmentation failed: {str(e)}")


class EnsembleSegmentationWorker(QThread):
    """Worker thread for running ensemble segmentation with refinement support."""
    
    finished = Signal(np.ndarray)  # Emits final mask
    error = Signal(str)  # Emits error message
    progress = Signal(int, str)  # Emits (percentage, status message)
    
    def __init__(
        self,
        ensemble_session: EnsembleSegmentationSession,
        force_reinit: bool = False
    ):
        super().__init__()
        self.ensemble_session = ensemble_session
        self.force_reinit = force_reinit
    
    def run(self):
        """Run ensemble segmentation in background."""
        try:
            classes = self.ensemble_session.get_classes()
            if not classes:
                self.error.emit("No foreground scribbles found. Please draw some scribbles first.")
                return
            
            color_spaces = self.ensemble_session.color_spaces
            
            # Progress for each color space
            self.progress.emit(10, f"Color space 1: {color_spaces[0]}...")
            self.progress.emit(40, f"Color space 2: {color_spaces[1]}...")
            self.progress.emit(70, f"Color space 3: {color_spaces[2]}...")
            self.progress.emit(90, "Majority voting...")
            
            # Run ensemble segmentation
            final_mask = self.ensemble_session.segment_all_classes(force_reinit=self.force_reinit)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(final_mask)
            
        except Exception as e:
            self.error.emit(f"Ensemble segmentation failed: {str(e)}")



class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.image_array: Optional[np.ndarray] = None
        self.session: Optional[MultiClassSegmentationSession] = None
        self.ensemble_session: Optional[EnsembleSegmentationSession] = None
        self.current_image_path: Optional[Path] = None
        self.segmentation_worker: Optional[SegmentationWorker] = None
        
        # Class management: maps class_id -> {"name": str, "color": QColor}
        self.classes = {}
        self.next_class_id = 1  # Start from 1 (background)
        
        # UI setup
        self.setWindowTitle("Interactive GrabCut Segmentation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply modern stylesheet
        self._apply_stylesheet()
        
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_widget()
        self._create_dock_widgets()
        self._create_status_bar()
        
        # Initialize with background class
        self._initialize_default_classes()
        
        self._update_ui_state()
    
    def _apply_stylesheet(self):
        """Apply modern stylesheet to the application."""
        stylesheet = """
        /* Main window and general widgets */
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9pt;
        }
        
        /* Dock widgets */
        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
            font-weight: bold;
        }
        
        QDockWidget::title {
            background-color: #2c3e50;
            color: white;
            padding: 8px;
            font-size: 10pt;
            font-weight: bold;
        }
        
        /* Group boxes */
        QGroupBox {
            font-weight: bold;
            border: 2px solid #d0d0d0;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 8px;
            background-color: white;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #2c3e50;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 28px;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #21618c;
        }
        
        QPushButton:disabled {
            background-color: #bdc3c7;
            color: #7f8c8d;
        }
        
        /* Primary action buttons */
        QPushButton#primaryButton {
            background-color: #27ae60;
            font-size: 10pt;
            padding: 10px 20px;
        }
        
        QPushButton#primaryButton:hover {
            background-color: #229954;
        }
        
        QPushButton#primaryButton:pressed {
            background-color: #1e8449;
        }
        
        /* Secondary action buttons */
        QPushButton#secondaryButton {
            background-color: #e67e22;
        }
        
        QPushButton#secondaryButton:hover {
            background-color: #d35400;
        }
        
        /* Danger buttons */
        QPushButton#dangerButton {
            background-color: #e74c3c;
        }
        
        QPushButton#dangerButton:hover {
            background-color: #c0392b;
        }
        
        /* Small buttons */
        QPushButton#smallButton {
            padding: 4px 8px;
            min-height: 20px;
            font-size: 8pt;
        }
        
        /* Combo boxes */
        QComboBox {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 5px 10px;
            background-color: white;
            min-height: 24px;
        }
        
        QComboBox:hover {
            border: 1px solid #3498db;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        
        QComboBox QAbstractItemView {
            border: 1px solid #bdc3c7;
            selection-background-color: #3498db;
            background-color: white;
        }
        
        /* Spin boxes */
        QSpinBox {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
            min-height: 24px;
        }
        
        QSpinBox:hover {
            border: 1px solid #3498db;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            border: 1px solid #bdc3c7;
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: #3498db;
            border: 1px solid #2980b9;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #2980b9;
        }
        
        /* Checkboxes */
        QCheckBox {
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #bdc3c7;
            border-radius: 3px;
            background-color: white;
        }
        
        QCheckBox::indicator:hover {
            border: 2px solid #3498db;
        }
        
        QCheckBox::indicator:checked {
            background-color: #3498db;
            border: 2px solid #2980b9;
        }
        
        /* Labels */
        QLabel {
            color: #2c3e50;
        }
        
        QLabel#headerLabel {
            font-size: 11pt;
            font-weight: bold;
            color: #2c3e50;
            padding: 5px 0px;
        }
        
        QLabel#subHeaderLabel {
            font-size: 9pt;
            font-weight: bold;
            color: #34495e;
        }
        
        QLabel#hintLabel {
            color: #7f8c8d;
            font-size: 8pt;
            font-style: italic;
        }
        
        /* Status bar */
        QStatusBar {
            background-color: #34495e;
            color: white;
            font-size: 9pt;
        }
        
        QStatusBar::item {
            border: none;
        }
        
        /* Toolbar */
        QToolBar {
            background-color: #ecf0f1;
            border: none;
            padding: 4px;
            spacing: 8px;
        }
        
        QToolBar::separator {
            background-color: #bdc3c7;
            width: 1px;
            margin: 4px 8px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical {
            border: none;
            background-color: #ecf0f1;
            width: 12px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #bdc3c7;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #95a5a6;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        /* Progress dialog */
        QProgressDialog {
            background-color: white;
        }
        
        QProgressBar {
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            text-align: center;
            background-color: #ecf0f1;
        }
        
        QProgressBar::chunk {
            background-color: #3498db;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(stylesheet)
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)
        
        save_mask_action = QAction("&Save Segmentation Mask...", self)
        save_mask_action.setShortcut(QKeySequence.Save)
        save_mask_action.triggered.connect(self._save_mask)
        file_menu.addAction(save_mask_action)
        
        save_session_action = QAction("Save &Session...", self)
        save_session_action.triggered.connect(self._save_session)
        file_menu.addAction(save_session_action)
        
        load_session_action = QAction("Load S&ession...", self)
        load_session_action.triggered.connect(self._load_session)
        file_menu.addAction(load_session_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(lambda: self.canvas.undo())
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(lambda: self.canvas.redo())
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        clear_action = QAction("&Clear Scribbles", self)
        clear_action.triggered.connect(self._clear_scribbles)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("R")
        reset_view_action.triggered.connect(lambda: self.canvas.reset_view())
        view_menu.addAction(reset_view_action)
        
        toggle_seg_action = QAction("&Toggle Segmentation Overlay", self)
        toggle_seg_action.setShortcut("T")
        toggle_seg_action.setCheckable(True)
        toggle_seg_action.setChecked(True)
        toggle_seg_action.triggered.connect(
            lambda checked: self.canvas.set_show_segmentation(checked)
        )
        view_menu.addAction(toggle_seg_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create toolbar with workflow-oriented actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Step 1: Open image
        workflow_label = QLabel("  Workflow:  ")
        workflow_label.setObjectName("headerLabel")
        toolbar.addWidget(workflow_label)
        
        open_btn = QPushButton("📁 Open Image")
        open_btn.setObjectName("primaryButton")
        open_btn.clicked.connect(self._open_image)
        open_btn.setToolTip("Step 1: Load an image to segment")
        toolbar.addWidget(open_btn)
        
        toolbar.addSeparator()
        
        # Step 2-7: Draw scribbles (handled in side panel)
        draw_label = QLabel("→ Draw Scribbles →")
        draw_label.setObjectName("hintLabel")
        toolbar.addWidget(draw_label)
        
        toolbar.addSeparator()
        
        # Step 3/6: Run segmentation
        self.segment_btn = QPushButton("▶ Segment")
        self.segment_btn.setObjectName("primaryButton")
        self.segment_btn.clicked.connect(self._run_segmentation)
        self.segment_btn.setEnabled(False)
        self.segment_btn.setToolTip("Step 3/6: Run segmentation with current scribbles")
        toolbar.addWidget(self.segment_btn)
        
        # Step 8: Refine segmentation
        self.refine_btn = QPushButton("🔧 Refine")
        self.refine_btn.setObjectName("secondaryButton")
        self.refine_btn.clicked.connect(lambda: self._run_segmentation(refine=True))
        self.refine_btn.setEnabled(False)
        self.refine_btn.setToolTip("Step 8: Refine segmentation (keeps existing models)")
        toolbar.addWidget(self.refine_btn)
        
        toolbar.addSeparator()
        
        # Step 9: Save results
        save_btn = QPushButton("💾 Save Mask")
        save_btn.clicked.connect(self._save_mask)
        save_btn.setToolTip("Step 9: Export segmentation mask")
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()
        
        # Reset
        reset_btn = QPushButton("🔄 Reset All")
        reset_btn.setObjectName("dangerButton")
        reset_btn.clicked.connect(self._reset_segmentation)
        reset_btn.setToolTip("Clear all scribbles and segmentation")
        toolbar.addWidget(reset_btn)
        
        # Add stretch to push everything to the left
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
    
    def _create_central_widget(self):
        """Create central widget with canvas."""
        self.canvas = CanvasWidget()
        self.canvas.scribbles_changed.connect(self._on_scribbles_changed)
        
        # Will be set to background (class 1) after initialization
        self.canvas.set_current_class(1)
        
        self.setCentralWidget(self.canvas)
    
    def _create_dock_widgets(self):
        """Create dockable control panels organized by workflow."""
        
        # LEFT PANEL: Drawing and Class Management (Steps 2-7)
        draw_dock = QDockWidget("📝 Draw & Classify", self)
        draw_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        draw_widget = QWidget()
        draw_layout = QVBoxLayout()
        draw_layout.setSpacing(12)
        
        # Workflow guide at top
        workflow_guide = QLabel(
            "<b>Workflow Guide:</b><br>"
            "1. Open Image<br>"
            "2. Add Classes (objects to segment)<br>"
            "3. Draw Scribbles per class<br>"
            "4. Click Segment<br>"
            "5. Add more classes if needed<br>"
            "6. Add Background scribbles<br>"
            "7. Click Segment again<br>"
            "8. Add refinement scribbles<br>"
            "9. Click Refine<br>"
            "10. Save Mask"
        )
        workflow_guide.setObjectName("hintLabel")
        workflow_guide.setWordWrap(True)
        workflow_guide.setFrameStyle(QLabel.StyledPanel)
        workflow_guide.setStyleSheet("padding: 8px; background-color: #fff3cd; border-radius: 4px;")
        draw_layout.addWidget(workflow_guide)
        
        # Class selection and management
        class_group = QGroupBox("Step 2-5: Class Management")
        class_layout = QVBoxLayout()
        class_layout.setSpacing(8)
        
        # Current class selector
        class_select_label = QLabel("Current Class:")
        class_select_label.setObjectName("subHeaderLabel")
        class_layout.addWidget(class_select_label)
        
        self.class_combo = QComboBox()
        self.class_combo.setToolTip("Select which class to draw scribbles for")
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        class_layout.addWidget(self.class_combo)
        
        # Add new class button (prominent)
        self.add_class_btn = QPushButton("➕ Add New Class")
        self.add_class_btn.setObjectName("primaryButton")
        self.add_class_btn.clicked.connect(self._add_new_class)
        self.add_class_btn.setToolTip("Add a new object class (e.g., dog, chair, person)")
        class_layout.addWidget(self.add_class_btn)
        
        # Edit/Remove buttons (smaller, side by side)
        class_btn_layout = QHBoxLayout()
        
        self.edit_class_btn = QPushButton("✏️ Edit")
        self.edit_class_btn.setObjectName("smallButton")
        self.edit_class_btn.clicked.connect(self._edit_current_class)
        self.edit_class_btn.setEnabled(False)
        self.edit_class_btn.setToolTip("Change class name or color")
        class_btn_layout.addWidget(self.edit_class_btn)
        
        self.remove_class_btn = QPushButton("🗑️ Remove")
        self.remove_class_btn.setObjectName("smallButton")
        self.remove_class_btn.clicked.connect(self._remove_current_class)
        self.remove_class_btn.setEnabled(False)
        self.remove_class_btn.setToolTip("Delete this class")
        class_btn_layout.addWidget(self.remove_class_btn)
        
        class_layout.addLayout(class_btn_layout)
        
        class_group.setLayout(class_layout)
        draw_layout.addWidget(class_group)
        
        # Drawing tools
        draw_tools_group = QGroupBox("Step 3-8: Drawing Tools")
        draw_tools_layout = QVBoxLayout()
        draw_tools_layout.setSpacing(10)
        
        # Brush size
        brush_size_label = QLabel("Brush Size:")
        brush_size_label.setObjectName("subHeaderLabel")
        draw_tools_layout.addWidget(brush_size_label)
        
        self.brush_label = QLabel("Size: 5 px")
        draw_tools_layout.addWidget(self.brush_label)
        
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(5)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        draw_tools_layout.addWidget(self.brush_slider)
        
        # Eraser mode
        self.eraser_checkbox = QCheckBox("🧹 Eraser Mode")
        self.eraser_checkbox.toggled.connect(self._on_eraser_toggled)
        self.eraser_checkbox.setToolTip("Enable to erase scribbles")
        draw_tools_layout.addWidget(self.eraser_checkbox)
        
        # Clear scribbles button
        clear_scribbles_btn = QPushButton("Clear All Scribbles")
        clear_scribbles_btn.setObjectName("dangerButton")
        clear_scribbles_btn.clicked.connect(self._clear_scribbles)
        clear_scribbles_btn.setToolTip("Remove all drawn scribbles")
        draw_tools_layout.addWidget(clear_scribbles_btn)
        
        draw_tools_group.setLayout(draw_tools_layout)
        draw_layout.addWidget(draw_tools_group)
        
        # Quick tips
        tips_label = QLabel(
            "<b>💡 Tips:</b><br>"
            "• Left Click: Draw<br>"
            "• Middle Click: Pan<br>"
            "• Scroll: Zoom<br>"
            "• Ctrl+Z: Undo<br>"
            "• Ctrl+Y: Redo"
        )
        tips_label.setObjectName("hintLabel")
        tips_label.setWordWrap(True)
        tips_label.setFrameStyle(QLabel.StyledPanel)
        tips_label.setStyleSheet("padding: 8px; background-color: #d1ecf1; border-radius: 4px;")
        draw_layout.addWidget(tips_label)
        
        draw_layout.addStretch()
        
        draw_widget.setLayout(draw_layout)
        draw_dock.setWidget(draw_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, draw_dock)
        
        # RIGHT PANEL: Segmentation Settings
        seg_dock = QDockWidget("⚙️ Segmentation Settings", self)
        seg_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        seg_widget = QWidget()
        seg_layout = QVBoxLayout()
        seg_layout.setSpacing(12)
        
        # Mode selection (Single vs Ensemble)
        mode_group = QGroupBox("Segmentation Mode")
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(8)
        
        mode_label = QLabel("Choose mode:")
        mode_label.setObjectName("subHeaderLabel")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("🎯 Single Color Space", "single")
        self.mode_combo.addItem("🎲 Ensemble (3 Color Spaces)", "ensemble")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        mode_hint = QLabel("Single: Faster, good for most images<br>Ensemble: More robust, uses voting")
        mode_hint.setObjectName("hintLabel")
        mode_hint.setWordWrap(True)
        mode_layout.addWidget(mode_hint)
        
        mode_group.setLayout(mode_layout)
        seg_layout.addWidget(mode_group)
        
        # Single color space selection
        self.single_color_space_group = QGroupBox("Color Space (Single Mode)")
        single_color_space_layout = QVBoxLayout()
        single_color_space_layout.setSpacing(8)
        
        cs_label = QLabel("Feature Space:")
        cs_label.setObjectName("subHeaderLabel")
        single_color_space_layout.addWidget(cs_label)
        
        self.color_space_combo = QComboBox()
        color_spaces = [
            "ruderman_lab", "oklab", "jzczhz", "jzazbz", "cielab", "oklch",
            "c16_scd", "c02_scd", "rgb", "hsv_conic", "ycbcr_bt709", 
            "xyz", "srgb_linear", "opponent", "log_chroma", "ictcp_pq"
        ]
        for cs in color_spaces:
            self.color_space_combo.addItem(cs, cs)
        self.color_space_combo.setCurrentText("ruderman_lab")  # Default: top performer
        single_color_space_layout.addWidget(self.color_space_combo)
        
        cs_hint = QLabel("Default (ruderman_lab) works best for most images")
        cs_hint.setObjectName("hintLabel")
        cs_hint.setWordWrap(True)
        single_color_space_layout.addWidget(cs_hint)
        
        self.single_color_space_group.setLayout(single_color_space_layout)
        seg_layout.addWidget(self.single_color_space_group)
        
        # Ensemble color space selection
        self.ensemble_group = QGroupBox("Color Spaces (Ensemble Mode)")
        ensemble_layout = QVBoxLayout()
        ensemble_layout.setSpacing(8)
        
        # Create three dropdowns for ensemble
        self.ensemble_combo1 = QComboBox()
        self.ensemble_combo2 = QComboBox()
        self.ensemble_combo3 = QComboBox()
        
        for combo in [self.ensemble_combo1, self.ensemble_combo2, self.ensemble_combo3]:
            for cs in color_spaces:
                combo.addItem(cs, cs)
        
        # Set defaults: ruderman_lab, oklab, jzczhz (top performing combination)
        self.ensemble_combo1.setCurrentText("ruderman_lab")
        self.ensemble_combo2.setCurrentText("oklab")
        self.ensemble_combo3.setCurrentText("jzczhz")
        
        ensemble_layout.addWidget(QLabel("Color Space 1:"))
        ensemble_layout.addWidget(self.ensemble_combo1)
        ensemble_layout.addWidget(QLabel("Color Space 2:"))
        ensemble_layout.addWidget(self.ensemble_combo2)
        ensemble_layout.addWidget(QLabel("Color Space 3:"))
        ensemble_layout.addWidget(self.ensemble_combo3)
        
        # Tie-breaking strategy for ensemble
        ensemble_layout.addWidget(QLabel("Tie Strategy (3-way):"))
        self.ensemble_tie_combo = QComboBox()
        self.ensemble_tie_combo.addItem("First", "first")
        self.ensemble_tie_combo.addItem("Second", "second")
        self.ensemble_tie_combo.addItem("Third", "third")
        self.ensemble_tie_combo.setCurrentIndex(0)
        ensemble_layout.addWidget(self.ensemble_tie_combo)
        
        self.ensemble_group.setLayout(ensemble_layout)
        seg_layout.addWidget(self.ensemble_group)
        self.ensemble_group.setVisible(False)  # Hidden by default
        
        # GrabCut iterations
        iters_group = QGroupBox("GrabCut Iterations")
        iters_layout = QVBoxLayout()
        iters_layout.setSpacing(8)
        
        iters_label = QLabel("Iterations per run:")
        iters_label.setObjectName("subHeaderLabel")
        iters_layout.addWidget(iters_label)
        
        self.iters_spinbox = QSpinBox()
        self.iters_spinbox.setMinimum(1)
        self.iters_spinbox.setMaximum(20)
        self.iters_spinbox.setValue(5)
        iters_layout.addWidget(self.iters_spinbox)
        
        iters_hint = QLabel("Higher = better quality, but slower (5 is good default)")
        iters_hint.setObjectName("hintLabel")
        iters_hint.setWordWrap(True)
        iters_layout.addWidget(iters_hint)
        
        iters_group.setLayout(iters_layout)
        seg_layout.addWidget(iters_group)
        
        # Overlay opacity
        opacity_group = QGroupBox("Overlay Opacity")
        opacity_layout = QVBoxLayout()
        opacity_layout.setSpacing(8)
        
        self.opacity_label = QLabel("Opacity: 50%")
        opacity_layout.addWidget(self.opacity_label)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        opacity_group.setLayout(opacity_layout)
        seg_layout.addWidget(opacity_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(8)
        
        self.seed_refine_checkbox = QCheckBox("Apply Seed Refinement")
        self.seed_refine_checkbox.setChecked(True)
        self.seed_refine_checkbox.setToolTip("Apply MGC geodesic seed expansion (recommended)")
        
        self.post_smooth_checkbox = QCheckBox("Apply Post-Smoothing")
        self.post_smooth_checkbox.setChecked(True)
        self.post_smooth_checkbox.setToolTip("Apply MGC guided filter smoothing (recommended)")
        
        advanced_layout.addWidget(self.seed_refine_checkbox)
        advanced_layout.addWidget(self.post_smooth_checkbox)
        
        advanced_group.setLayout(advanced_layout)
        seg_layout.addWidget(advanced_group)
        
        seg_layout.addStretch()
        
        seg_widget.setLayout(seg_layout)
        seg_dock.setWidget(seg_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, seg_dock)
    
    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _update_ui_state(self):
        """Update UI element states based on application state."""
        has_image = self.image_array is not None
        mode = self.mode_combo.currentData()
        
        # Check for existing session based on mode
        if mode == "single":
            has_session = self.session is not None
            has_segmentation = has_session and np.any(self.session.final_mask > 0)
        else:  # ensemble
            has_session = self.ensemble_session is not None
            has_segmentation = has_session and np.any(self.ensemble_session.final_mask > 0)
        
        self.segment_btn.setEnabled(has_image)
        
        # Refine button enabled when we have existing segmentation in either mode
        self.refine_btn.setEnabled(has_segmentation)
    
    def _open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load image using PIL
            pil_img = Image.open(file_path).convert("RGB")
            img_array = np.array(pil_img, dtype=np.uint8)
            
            # Set image
            self.image_array = img_array
            self.current_image_path = Path(file_path)
            self.canvas.set_image(img_array)
            
            # Clear previous segmentation mask
            self.canvas.clear_segmentation()
            
            # Create new sessions (both single and ensemble)
            color_space = self.color_space_combo.currentData()
            gc_iters = self.iters_spinbox.value()
            
            # Single color space session
            self.session = MultiClassSegmentationSession(
                img_array,
                color_space=color_space,
                gc_iters=gc_iters,
                apply_seed_refinement=self.seed_refine_checkbox.isChecked(),
                apply_post_smoothing=self.post_smooth_checkbox.isChecked()
            )
            
            # Ensemble session
            ensemble_spaces = [
                self.ensemble_combo1.currentData(),
                self.ensemble_combo2.currentData(),
                self.ensemble_combo3.currentData()
            ]
            tie_strategy = self.ensemble_tie_combo.currentData()
            tie_map = {"first": 0, "second": 1, "third": 2}
            
            self.ensemble_session = EnsembleSegmentationSession(
                img_array,
                color_spaces=ensemble_spaces,
                gc_iters=gc_iters,
                apply_seed_refinement=self.seed_refine_checkbox.isChecked(),
                apply_post_smoothing=self.post_smooth_checkbox.isChecked(),
                label_tie_pref=tie_map.get(tie_strategy, 0)
            )
            
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def _run_segmentation(self, refine: bool = False):
        """Run segmentation."""
        if self.image_array is None:
            return
        
        # Get current mode
        mode = self.mode_combo.currentData()
        
        # Check that appropriate session exists
        if mode == "single" and self.session is None:
            return
        if mode == "ensemble" and self.ensemble_session is None:
            return
        
        # Get current annotations from canvas
        annotations = self.canvas.get_annotation_map()
        
        # Check if we have scribbles
        classes = sorted([int(x) for x in np.unique(annotations) if x > 1])
        if not classes:
            QMessageBox.warning(
                self,
                "No Scribbles",
                "Please draw some foreground scribbles before running segmentation."
            )
            return
        
        # Update session parameters
        if mode == "single":
            self.session.color_space = self.color_space_combo.currentData()
            self.session.gc_iters = self.iters_spinbox.value()
            self.session.apply_seed_refinement = self.seed_refine_checkbox.isChecked()
            self.session.apply_post_smoothing = self.post_smooth_checkbox.isChecked()
            
            # Update feature space if changed
            if self.session.color_space != self.session.color_space:
                self.session.img_feats = convert_color_space(self.image_array, self.session.color_space)
            
            self.session.update_annotations(annotations)
        else:  # ensemble
            self.ensemble_session.update_settings(
                gc_iters=self.iters_spinbox.value(),
                apply_seed_refinement=self.seed_refine_checkbox.isChecked(),
                apply_post_smoothing=self.post_smooth_checkbox.isChecked()
            )
            self.ensemble_session.update_annotations(annotations)
        
        # Create progress dialog
        progress = QProgressDialog("Running segmentation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Create worker thread
        if mode == "single":
            force_reinit = not refine
            self.segmentation_worker = SegmentationWorker(
                self.session, force_reinit, mode="single"
            )
        else:  # ensemble
            force_reinit = not refine
            self.segmentation_worker = EnsembleSegmentationWorker(
                self.ensemble_session, force_reinit
            )
        
        # Connect signals
        self.segmentation_worker.progress.connect(
            lambda pct, msg: (progress.setValue(pct), progress.setLabelText(msg))
        )
        self.segmentation_worker.finished.connect(
            lambda mask: self._on_segmentation_complete(mask, progress)
        )
        self.segmentation_worker.error.connect(
            lambda msg: self._on_segmentation_error(msg, progress)
        )
        progress.canceled.connect(self.segmentation_worker.terminate)
        
        # Start segmentation
        mode_str = "Ensemble" if mode == "ensemble" else ("Refining" if refine else "Running")
        self.status_bar.showMessage(f"{mode_str} segmentation...")
        self.segmentation_worker.start()
    
    def _on_segmentation_complete(self, mask: np.ndarray, progress: QProgressDialog):
        """Handle segmentation completion."""
        progress.close()
        
        # Update canvas
        self.canvas.set_segmentation_mask(mask)
        
        # Count classes
        unique_classes = np.unique(mask)
        num_classes = len([c for c in unique_classes if c > 0])
        
        self.status_bar.showMessage(
            f"Segmentation complete! Found {num_classes} class{'es' if num_classes != 1 else ''}"
        )
        
        self._update_ui_state()
    
    def _on_segmentation_error(self, error_msg: str, progress: QProgressDialog):
        """Handle segmentation error."""
        progress.close()
        QMessageBox.critical(self, "Segmentation Error", error_msg)
        self.status_bar.showMessage("Segmentation failed")
    
    def _reset_segmentation(self):
        """Reset segmentation and scribbles."""
        reply = QMessageBox.question(
            self,
            "Reset Segmentation",
            "This will clear all scribbles and segmentation results. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_scribbles(emit_signal=False)
            self.canvas.clear_segmentation()
            
            if self.session is not None and self.image_array is not None:
                # Reset session
                color_space = self.color_space_combo.currentData()
                gc_iters = self.iters_spinbox.value()
                
                self.session = MultiClassSegmentationSession(
                    self.image_array,
                    color_space=color_space,
                    gc_iters=gc_iters,
                    apply_seed_refinement=self.seed_refine_checkbox.isChecked(),
                    apply_post_smoothing=self.post_smooth_checkbox.isChecked()
                )
            
            self.status_bar.showMessage("Reset complete")
            self._update_ui_state()
    
    def _save_mask(self):
        """Save segmentation mask."""
        # Get the active session based on current mode
        mode = self.mode_combo.currentData()
        active_session = self.ensemble_session if mode == "ensemble" else self.session
        
        # Check if we have a valid mask to save
        if active_session is None or not np.any(active_session.final_mask > 0):
            QMessageBox.warning(self, "No Segmentation", "No segmentation to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation Mask",
            "",
            "NumPy Array (*.npy);;PNG Image (*.png);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith(".npy"):
                np.save(file_path, active_session.final_mask)
            else:
                # Save as indexed PNG with VOC palette
                img = Image.fromarray(active_session.final_mask, mode="P")
                img.putpalette(voc_palette().ravel().tolist())
                img.save(file_path)
            
            self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save mask:\n{str(e)}")
    
    def _save_session(self):
        """Save session state."""
        if self.session is None:
            QMessageBox.warning(self, "No Session", "No session to save.")
            return
        
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Session"
        )
        
        if not dir_path:
            return
        
        try:
            self.session.save_session(Path(dir_path))
            self.status_bar.showMessage(f"Session saved to: {dir_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save session:\n{str(e)}")
    
    def _load_session(self):
        """Load session state."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Load Session From"
        )
        
        if not dir_path:
            return
        
        try:
            if self.session is None:
                QMessageBox.warning(
                    self,
                    "No Image",
                    "Please load an image first before loading a session."
                )
                return
            
            self.session.load_session(Path(dir_path))
            
            # Update canvas
            self.canvas.clear_scribbles(emit_signal=False)
            self.canvas.set_segmentation_mask(self.session.final_mask)
            
            self.status_bar.showMessage(f"Session loaded from: {dir_path}")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load session:\n{str(e)}")
    
    def _clear_scribbles(self):
        """Clear all scribbles."""
        reply = QMessageBox.question(
            self,
            "Clear Scribbles",
            "Clear all scribbles?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_scribbles()
            self.status_bar.showMessage("Scribbles cleared")
    
    def _initialize_default_classes(self):
        """Initialize with just the background class."""
        # Add background class (label 1, black color)
        self.classes[1] = {
            "name": "Background",
            "color": QColor(0, 0, 0)
        }
        self.next_class_id = 2
        
        # Update combo box
        self.class_combo.addItem("Background", 1)
        self.canvas.add_class_color(1, QColor(0, 0, 0))
        
        # Enable add button, disable edit/remove
        self.add_class_btn.setEnabled(True)
        self.edit_class_btn.setEnabled(False)
        self.remove_class_btn.setEnabled(False)
    
    def _on_class_changed(self, index: int):
        """Handle class selection change."""
        if self.class_combo.count() == 0:
            return
        
        class_id = self.class_combo.currentData()
        if class_id is None:
            return
            
        self.canvas.set_current_class(class_id)
        
        # Update button states
        # Can't remove background (class 1)
        can_edit = class_id is not None
        can_remove = class_id != 1
        
        self.edit_class_btn.setEnabled(can_edit)
        self.remove_class_btn.setEnabled(can_remove)
        
        # Update status
        if class_id in self.classes:
            class_name = self.classes[class_id]["name"]
            self.status_bar.showMessage(f"Drawing: {class_name} (label {class_id - 1})")
    
    def _on_brush_size_changed(self, value: int):
        """Handle brush size change."""
        self.canvas.set_brush_size(value)
        self.brush_label.setText(f"Size: {value} px")
    
    def _on_eraser_toggled(self, checked: bool):
        """Handle eraser toggle."""
        self.canvas.set_eraser_mode(checked)
        self.status_bar.showMessage("Eraser mode: " + ("ON" if checked else "OFF"))
    
    def _on_opacity_changed(self, value: int):
        """Handle opacity change."""
        opacity = value / 100.0
        self.canvas.set_segmentation_opacity(opacity)
        self.opacity_label.setText(f"Opacity: {value}%")
    
    def _on_scribbles_changed(self):
        """Handle scribbles change."""
        self.status_bar.showMessage("Scribbles updated")
    
    def _on_mode_changed(self, index: int):
        """Handle segmentation mode change."""
        mode = self.mode_combo.currentData()
        
        # Show/hide appropriate controls
        if mode == "single":
            self.single_color_space_group.setVisible(True)
            self.ensemble_group.setVisible(False)
            self.status_bar.showMessage("Mode: Single Color Space (with refinement)")
        else:  # ensemble
            self.single_color_space_group.setVisible(False)
            self.ensemble_group.setVisible(True)
            self.status_bar.showMessage("Mode: Ensemble (Majority Voting with refinement)")
        
        self._update_ui_state()
    
    def _add_new_class(self):
        """Add a new foreground class."""
        # Limit to reasonable number
        if self.next_class_id > 50:
            QMessageBox.warning(self, "Warning", "Maximum number of classes (50) reached.")
            return
        
        # Ask for class name
        name, ok = QInputDialog.getText(
            self, 
            "Add New Class", 
            f"Enter class name (will be assigned label {self.next_class_id - 1}):",
            text=f"Object_{self.next_class_id - 1}"
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Generate a nice default color (cycling through distinct colors)
        color_palette = [
            QColor(0, 128, 0),      # Green
            QColor(128, 128, 0),    # Olive
            QColor(0, 0, 128),      # Navy
            QColor(128, 0, 128),    # Purple
            QColor(0, 128, 128),    # Teal
            QColor(192, 0, 0),      # Red
            QColor(192, 128, 0),    # Orange
            QColor(64, 0, 128),     # Indigo
        ]
        default_color = color_palette[(self.next_class_id - 2) % len(color_palette)]
        
        # Pick a color
        color = QColorDialog.getColor(default_color, self, f"Choose color for '{name}'")
        
        if not color.isValid():
            return
        
        # Add to class registry
        class_id = self.next_class_id
        self.classes[class_id] = {
            "name": name,
            "color": color
        }
        self.next_class_id += 1
        
        # Add to combo box
        display_name = f"{name} (label {class_id - 1})"
        self.class_combo.addItem(display_name, class_id)
        self.canvas.add_class_color(class_id, color)
        
        # Select the new class
        self.class_combo.setCurrentIndex(self.class_combo.count() - 1)
        
        self.status_bar.showMessage(f"Added class: {name} (label {class_id - 1})")
    
    def _edit_current_class(self):
        """Edit the name or color of the currently selected class."""
        current_index = self.class_combo.currentIndex()
        class_id = self.class_combo.currentData()
        
        if class_id is None or class_id not in self.classes:
            return
        
        current_name = self.classes[class_id]["name"]
        current_color = self.classes[class_id]["color"]
        
        # Ask for new name
        name, ok = QInputDialog.getText(
            self,
            "Edit Class",
            f"Class name (label {class_id - 1}):",
            text=current_name
        )
        
        if not ok:
            return
        
        if name.strip():
            name = name.strip()
        else:
            name = current_name
        
        # Ask for new color
        color = QColorDialog.getColor(current_color, self, f"Choose color for '{name}'")
        
        if not color.isValid():
            color = current_color
        
        # Update class info
        self.classes[class_id]["name"] = name
        self.classes[class_id]["color"] = color
        
        # Update combo box
        display_name = f"{name} (label {class_id - 1})"
        self.class_combo.setItemText(current_index, display_name)
        
        # Update canvas color
        self.canvas.add_class_color(class_id, color)
        
        self.status_bar.showMessage(f"Updated: {name}")
    
    def _remove_current_class(self):
        """Remove the currently selected class."""
        current_index = self.class_combo.currentIndex()
        class_id = self.class_combo.currentData()
        
        # Can't remove background
        if class_id == 1:
            QMessageBox.information(self, "Info", "Cannot remove the background class.")
            return
        
        if class_id not in self.classes:
            return
        
        class_name = self.classes[class_id]["name"]
        
        reply = QMessageBox.question(
            self,
            "Remove Class",
            f"Remove class '{class_name}' (label {class_id - 1})?\n\n"
            "Existing scribbles for this class will remain but won't be segmented.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from registry
            del self.classes[class_id]
            
            # Remove from combo box
            self.class_combo.removeItem(current_index)
            
            # Select previous class if possible
            if self.class_combo.count() > 0:
                new_index = min(current_index, self.class_combo.count() - 1)
                self.class_combo.setCurrentIndex(new_index)
            
            self.status_bar.showMessage(f"Removed: {class_name}")
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Interactive GrabCut",
            "<h3>Interactive GrabCut Segmentation</h3>"
            "<p>A tool for multi-class image segmentation with iterative refinement.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Draw scribbles to define foreground/background</li>"
            "<li>Support for up to 20 foreground classes</li>"
            "<li>Single color space mode with iterative refinement</li>"
            "<li>Ensemble mode with majority voting (3 color spaces)</li>"
            "<li>16+ color spaces including Ruderman LAB, OKLAB, JzCzHz</li>"
            "<li>Advanced seed refinement and post-smoothing</li>"
            "</ul>"
            "<p><b>Default Settings:</b></p>"
            "<ul>"
            "<li>Single mode: Ruderman LAB (top performer)</li>"
            "<li>Ensemble mode: Ruderman LAB + OKLAB + JzCzHz</li>"
            "</ul>"
            "<p><b>Controls:</b></p>"
            "<ul>"
            "<li>Left Mouse: Draw scribbles</li>"
            "<li>Middle Mouse: Pan</li>"
            "<li>Mouse Wheel: Zoom</li>"
            "<li>Ctrl+Z: Undo</li>"
            "<li>Ctrl+Y: Redo</li>"
            "<li>R: Reset view</li>"
            "<li>T: Toggle segmentation overlay</li>"
            "</ul>"
        )
