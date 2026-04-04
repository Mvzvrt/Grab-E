#!/usr/bin/env python3
# Filename: splash_screen.py
# -*- coding: utf-8 -*-
"""Simple startup splash screen for Grab-E."""

from pathlib import Path
from datetime import date

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from how_to_use_page import HowToUsePage
from main_window import MainWindow
from utils import enable_windows_dark_title_bar, get_public_dir


class SplashScreen(QWidget):
    """Splash screen with a single start action."""

    def __init__(self):
        super().__init__()
        self.main_window = None
        self.how_to_use_window = None
        self.hero_logo = None
        self.assets_dir = get_public_dir()

        self.setWindowTitle("Grab-E")
        self.setMinimumSize(560, 360)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            QPushButton {
                border: none;
                background: transparent;
                color: #cccccc;
                font-size: 16px;
                font-weight: 400;
                padding: 6px 10px;
            }
            QPushButton#actionTextButton {
                text-align: left;
            }
            QPushButton:hover {
                color: #ffffff;
            }
            QPushButton:pressed {
                color: #f0f0f0;
            }
            QLabel#creditsText {
                color: #9aa0a6;
                font-size: 12px;
            }
            QPushButton#creditsIcon {
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 36, 32, 32)
        layout.setSpacing(8)

        content_group = QWidget()
        content_layout = QVBoxLayout(content_group)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.setAlignment(Qt.AlignHCenter)

        logo_path = self.assets_dir / "splash-screen-logo.svg"
        if logo_path.exists():
            self.hero_logo = QSvgWidget(str(logo_path))
            # Keep square dimensions to avoid distorting the 2000x2000 SVG.
            self.hero_logo.setFixedSize(840, 420)
            content_layout.addWidget(self.hero_logo, alignment=Qt.AlignHCenter)

        self.start_btn = QPushButton("Open new image")
        self.start_btn.setObjectName("actionTextButton")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFlat(True)
        self.start_btn.setFixedSize(290, 38)
        self.start_btn.clicked.connect(self._launch_main_window)
        self.start_icon_btn = self._create_icon_button(
            self.assets_dir / "start_with_new_image_logo.svg",
            self._launch_main_window
        )

        self.how_to_use_btn = QPushButton("How to Use")
        self.how_to_use_btn.setObjectName("actionTextButton")
        self.how_to_use_btn.setCursor(Qt.PointingHandCursor)
        self.how_to_use_btn.setFlat(True)
        self.how_to_use_btn.setFixedSize(290, 38)
        self.how_to_use_btn.clicked.connect(self._open_how_to_use)
        self.how_to_use_icon_btn = self._create_icon_button(
            self.assets_dir / "how_to_use_logo.svg",
            self._open_how_to_use
        )

        button_group = QWidget()
        button_layout = QVBoxLayout(button_group)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        button_layout.addLayout(self._create_inline_action_row(self.start_icon_btn, self.start_btn))
        button_layout.addLayout(self._create_inline_action_row(self.how_to_use_icon_btn, self.how_to_use_btn))

        content_layout.addWidget(button_group, alignment=Qt.AlignHCenter)

        layout.addStretch()
        layout.addWidget(content_group, alignment=Qt.AlignCenter)
        layout.addStretch()
        layout.addLayout(self._create_credits_row())

        self.winId()
        enable_windows_dark_title_bar(self)

    def _create_icon_button(self, icon_path: Path, callback):
        """Create a flat clickable icon button for a splash action."""
        icon_btn = QPushButton()
        icon_btn.setCursor(Qt.PointingHandCursor)
        icon_btn.setFlat(True)
        icon_btn.setFixedSize(32, 32)
        icon_btn.setIconSize(QSize(20, 20))
        if icon_path.exists():
            icon_btn.setIcon(QIcon(str(icon_path)))
        icon_btn.clicked.connect(callback)
        return icon_btn

    def _create_inline_action_row(self, icon_button: QPushButton, text_button: QPushButton):
        """Create a centered row with icon + text action controls."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(icon_button)
        row.addWidget(text_button)
        row.setAlignment(Qt.AlignHCenter)
        return row

    def _create_credits_row(self):
        """Create a muted credits footer with optional GitHub icon."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(6)

        github_icon_path = self.assets_dir / "github_logo.svg"
        if github_icon_path.exists():
            icon_btn = QPushButton()
            icon_btn.setObjectName("creditsIcon")
            icon_btn.setFlat(True)
            icon_btn.setFocusPolicy(Qt.NoFocus)
            icon_btn.setFixedSize(16, 16)
            icon_btn.setIcon(QIcon(str(github_icon_path)))
            icon_btn.setIconSize(QSize(14, 14))
            icon_btn.setEnabled(False)
            row.addWidget(icon_btn, alignment=Qt.AlignVCenter)

        credits = QLabel(f"© {date.today().year} Mzvzvrt. All rights reserved.")
        credits.setObjectName("creditsText")
        row.addWidget(credits, alignment=Qt.AlignVCenter)
        row.addStretch()
        return row

    def _launch_main_window(self):
        """Open main window and immediately prompt for image selection."""
        self.main_window = MainWindow()
        self.main_window.setWindowIcon(self.windowIcon())
        self.main_window.show()
        self.main_window.start_with_new_image()
        self.close()

    def _open_how_to_use(self):
        """Open how-to-use page and keep splash available for back navigation."""
        self.hide()
        self.how_to_use_window = HowToUsePage(
            on_back=self._return_from_how_to_use,
            back_icon_path=self.assets_dir / "arrow_back.svg",
        )
        self.how_to_use_window.setWindowIcon(self.windowIcon())
        self.how_to_use_window.showMaximized()

    def _return_from_how_to_use(self):
        """Show splash screen again after closing the how-to-use page."""
        self.showMaximized()
        self.raise_()
        self.activateWindow()
