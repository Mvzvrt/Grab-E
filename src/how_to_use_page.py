#!/usr/bin/env python3
# Filename: how_to_use_page.py
# -*- coding: utf-8 -*-
"""Placeholder How to Use page for Grab-E."""

from PySide6.QtWidgets import QVBoxLayout, QWidget


class HowToUsePage(QWidget):
    """Placeholder page for future usage instructions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("How to Use - Grab-E")
        self.setMinimumSize(560, 360)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            """
        )

        # Intentionally left empty for now.
        self.setLayout(QVBoxLayout())
