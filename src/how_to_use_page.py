#!/usr/bin/env python3
# Filename: how_to_use_page.py
# -*- coding: utf-8 -*-
"""How-to-use template page for Grab-E."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
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
                font-size: 12px;
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
        """Build a scannable, sectioned help page template."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("backButton")
        if self.back_icon_path and self.back_icon_path.exists():
            self.back_btn.setIcon(QIcon(str(self.back_icon_path)))
        self.back_btn.clicked.connect(self._handle_back)
        top_row.addWidget(self.back_btn, alignment=Qt.AlignLeft)
        top_row.addStretch()

        page_title = QLabel("How to Use Grab-E")
        page_title.setObjectName("pageTitle")
        page_title.setAlignment(Qt.AlignLeft)

        page_subtitle = QLabel(
            "A quick walkthrough template for interactive segmentation workflows."
        )
        page_subtitle.setObjectName("pageSubtitle")
        page_subtitle.setWordWrap(True)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        sections = [
            (
                "Step 1",
                "Open an Image",
                (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    "Vestibulum consequat, nibh ac ultrices elementum, urna sem "
                    "tempor libero, non posuere mi orci non nulla."
                ),
                "TODO: Insert screenshot of the splash action for opening a new image.",
            ),
            (
                "Step 2",
                "Draw Scribbles",
                (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    "Proin bibendum sapien in mauris ultricies, vitae pretium "
                    "orci consectetur. Integer varius odio at dignissim bibendum."
                ),
                "TODO: Insert screenshot of class selection and canvas scribble tools.",
            ),
            (
                "Step 3",
                "Run Segmentation",
                (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    "Mauris hendrerit, purus id sollicitudin posuere, sem magna "
                    "pharetra nunc, in fringilla turpis risus nec est."
                ),
                "TODO: Insert screenshot of Segment/Refine controls and resulting mask.",
            ),
            (
                "Step 4",
                "Save Outputs",
                (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    "Suspendisse potenti. Nam ut facilisis tortor, vitae tempor "
                    "mauris. Morbi posuere velit et est sollicitudin, at feugiat mi luctus."
                ),
                "TODO: Insert screenshot of Save Mask and Save Scribbles actions.",
            ),
        ]

        for kicker, title, body, screenshot_note in sections:
            content_layout.addWidget(self._build_section_card(kicker, title, body, screenshot_note))

        content_layout.addStretch()
        scroll_area.setWidget(content)

        root_layout.addLayout(top_row)
        root_layout.addWidget(page_title)
        root_layout.addWidget(page_subtitle)
        root_layout.addWidget(scroll_area)

    def _handle_back(self):
        """Return to splash screen."""
        if callable(self.on_back):
            self.on_back()
        self.close()

    def _build_section_card(self, kicker: str, title: str, body: str, screenshot_note: str) -> QFrame:
        """Create one instruction section card."""
        card = QFrame()
        card.setObjectName("sectionCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)

        kicker_label = QLabel(kicker)
        kicker_label.setObjectName("sectionKicker")

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")

        body_label = QLabel(body)
        body_label.setObjectName("sectionBody")
        body_label.setWordWrap(True)

        card_layout.addWidget(kicker_label)
        card_layout.addWidget(title_label)
        card_layout.addWidget(body_label)

        # Screenshot placeholder (uncomment when real screenshots are available):
        # screenshot_placeholder = QLabel(screenshot_note)
        # screenshot_placeholder.setObjectName("sectionBody")
        # screenshot_placeholder.setWordWrap(True)
        # screenshot_placeholder.setMinimumHeight(180)
        # screenshot_placeholder.setAlignment(Qt.AlignCenter)
        # screenshot_placeholder.setStyleSheet(
        #     "border: 1px dashed #3e3e42; border-radius: 6px; color: #8a8a8a;"
        # )
        # card_layout.addWidget(screenshot_placeholder)

        return card
