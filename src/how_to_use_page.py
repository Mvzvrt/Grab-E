#!/usr/bin/env python3
# Filename: how_to_use_page.py
# -*- coding: utf-8 -*-
"""How-to-use template page for Grab-E."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils import enable_windows_dark_title_bar


class HowToUsePage(QWidget):
    """Structured how-to-use template with placeholder instructional content."""

    def __init__(self, on_back=None, back_icon_path=None):
        super().__init__()
        self.on_back = on_back
        self.back_icon_path = Path(back_icon_path) if back_icon_path else None
        self.setWindowTitle("How to Use - Grab-E")
        self.setMinimumSize(560, 360)
        self._apply_stylesheet()
        self._build_ui()
        self.winId()
        enable_windows_dark_title_bar(self)

    def _apply_stylesheet(self):
        """Apply app-consistent dark styling for readability and hierarchy."""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            QLabel#pageTitle {
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#pageSubtitle {
                font-size: 13px;
                color: #a9a9a9;
            }
            QLabel#h1 {
                font-size: 40px;
                font-weight: 700;
                color: #ffffff;
                margin-top: 12px;
                margin-bottom: 18px;
            }
            QLabel#h2 {
                font-size: 25px;
                font-weight: 600;
                color: #ffffff;
                margin-top: 16px;
                margin-bottom: 12px;
            }
            QLabel#bodyText {
                font-size: 15px;
                color: #d0d0d0;
                line-height: 1.5;
                margin-bottom: 16px;
            }
            QPushButton#backButton {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 6px 12px;
                background-color: #252526;
                color: #cccccc;
                font-size: 11pt;
            }
            QPushButton#backButton:hover {
                background-color: #2d2d30;
                color: #ffffff;
            }
            QPushButton#backButton:pressed {
                background-color: #3a3a3d;
            }
            QFrame#sectionCard {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 8px;
            }
            QLabel#sectionKicker {
                font-size: 11px;
                font-weight: 700;
                color: #4fc1ff;
                letter-spacing: 0.5px;
            }
            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: 600;
                color: #ffffff;
            }
            QLabel#sectionBody {
                font-size: 14px;
                color: #d0d0d0;
                line-height: 1.4;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #1e1e1e;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3e3e42;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _build_ui(self):
        """Build a scannable, structured help page with title and content sections."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(14)

        # Top row with back button
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("backButton")
        if self.back_icon_path and self.back_icon_path.exists():
            self.back_btn.setIcon(QIcon(str(self.back_icon_path)))
        self.back_btn.clicked.connect(self._handle_back)
        top_row.addWidget(self.back_btn, alignment=Qt.AlignLeft)
        top_row.addStretch()

        # H1 title
        h1_title = QLabel("How To Use Grab-E")
        h1_title.setObjectName("h1")
        h1_title.setAlignment(Qt.AlignLeft)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # H2: Open new image
        h2_open_image = QLabel("1. Open new image")
        h2_open_image.setObjectName("h2")

        # Body text
        body_text = QLabel("Click the open new image button.")
        body_text.setObjectName("bodyText")
        body_text.setWordWrap(True)

        # Load and display open_image.png at 50% size
        image_widget = self._load_screenshot("open_image.png", 960, 516)

        # Additional body text
        body_text_2 = QLabel("Find file in file explorer to open")
        body_text_2.setObjectName("bodyText")
        body_text_2.setWordWrap(True)

        # Load and display open_image_file_explorer.png
        image_widget_2 = self._load_screenshot("open_image_file_explorer.png", 960, 516)

        add_new_classes = QLabel("2. Add new classes")
        add_new_classes.setObjectName("h2")

        anc_body = QLabel("Click the add new class button")
        anc_body.setObjectName("bodyText")
        anc_body.setWordWrap(True)
        anc_image_widget_1 = self._load_screenshot("add_new_class_1.png", 960, 516)

        anc_body_2 = QLabel("Enter new class name")
        anc_body_2.setObjectName("bodyText")
        anc_body_2.setWordWrap(True)
        anc_image_widget_2 = self._load_screenshot("add_new_class_2.png", 960, 516)

        anc_body_3 = QLabel("Enter new class ID")
        anc_body_3.setObjectName("bodyText")
        anc_body_3.setWordWrap(True)
        anc_image_widget_3 = self._load_screenshot("add_new_class_3.png", 960, 516)

        anc_body_4 = QLabel("Choose color for new class")
        anc_body_4.setObjectName("bodyText")
        anc_body_4.setWordWrap(True)
        anc_image_widget_4 = self._load_screenshot("add_new_class_4.png", 960, 516)

        drawing_scribbles = QLabel("3. Draw scribbles to segment")
        drawing_scribbles.setObjectName("h2")

        ds_body = QLabel("Click the class selection button")
        ds_body.setObjectName("bodyText")
        ds_body.setWordWrap(True)
        ds_image_widget_1 = self._load_screenshot("ds_1.png", 960, 516)

        ds_body_2 = QLabel("Choose class to draw scribbles for")
        ds_body_2.setObjectName("bodyText")
        ds_body_2.setWordWrap(True)
        ds_image_widget_2 = self._load_screenshot("ds_2.png", 960, 516)

        ds_body_3 = QLabel("Draw scribbles to segment the image")
        ds_body_3.setObjectName("bodyText")
        ds_body_3.setWordWrap(True)
        ds_image_widget_3 = self._load_screenshot("ds_3.png", 960, 516)

        ds_body_4 = QLabel("Use scribble size slider to adjust brush size")
        ds_body_4.setObjectName("bodyText")
        ds_body_4.setWordWrap(True)
        ds_image_widget_4 = self._load_screenshot("ds_4.png", 960, 516)

        ds_body_5 = QLabel("Use eraser mode to remove a scribble")
        ds_body_5.setObjectName("bodyText")
        ds_body_5.setWordWrap(True)
        ds_image_widget_5 = self._load_screenshot("ds_5.png", 960, 516)

        ds_body_6 = QLabel("Click clear all scribbles to remove all scribbles")
        ds_body_6.setObjectName("bodyText")
        ds_body_6.setWordWrap(True)
        ds_image_widget_6 = self._load_screenshot("ds_6.png", 960, 516)

        content_layout.addWidget(h2_open_image)
        content_layout.addWidget(body_text)
        content_layout.addWidget(image_widget)
        content_layout.addSpacing(24)
        content_layout.addWidget(body_text_2)
        content_layout.addWidget(image_widget_2)
        content_layout.addSpacing(48)
        content_layout.addWidget(add_new_classes)
        content_layout.addWidget(anc_body)
        content_layout.addWidget(anc_image_widget_1)
        content_layout.addSpacing(24)
        content_layout.addWidget(anc_body_2)
        content_layout.addWidget(anc_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(anc_body_3)
        content_layout.addWidget(anc_image_widget_3)
        content_layout.addSpacing(24)
        content_layout.addWidget(anc_body_4)
        content_layout.addWidget(anc_image_widget_4)
        content_layout.addSpacing(48)
        content_layout.addWidget(drawing_scribbles)
        content_layout.addWidget(ds_body)
        content_layout.addWidget(ds_image_widget_1)
        content_layout.addSpacing(24)
        content_layout.addWidget(ds_body_2)
        content_layout.addWidget(ds_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(ds_body_3)
        content_layout.addWidget(ds_image_widget_3)
        content_layout.addSpacing(24)
        content_layout.addWidget(ds_body_4)
        content_layout.addWidget(ds_image_widget_4)
        content_layout.addSpacing(24)
        content_layout.addWidget(ds_body_5)
        content_layout.addWidget(ds_image_widget_5)
        content_layout.addSpacing(24)
        content_layout.addWidget(ds_body_6)
        content_layout.addWidget(ds_image_widget_6)
        content_layout.addStretch()

        scroll_area.setWidget(content)

        root_layout.addLayout(top_row)
        root_layout.addWidget(h1_title)
        root_layout.addWidget(scroll_area)

    def _handle_back(self):
        """Return to splash screen."""
        if callable(self.on_back):
            self.on_back()
        self.close()

    def _load_screenshot(self, filename: str, target_width: int, target_height: int) -> QLabel | None:
        """Load and display a screenshot from src/public/ at specified dimensions.
        
        Args:
            filename: Name of the image file in src/public/
            target_width: Target width in pixels
            target_height: Target height in pixels
            
        Returns:
            QLabel with scaled image, or None if file not found.
        """
        src_dir = Path(__file__).parent
        image_path = src_dir / "public" / filename
        
        if not image_path.exists():
            return None
        
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return None
        
        scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        
        image_label = QLabel()
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        
        return image_label
