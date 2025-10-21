#!/usr/bin/env python3
# Filename: main.py
# -*- coding: utf-8 -*-
"""
Interactive GrabCut Segmentation Application Entry Point
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Interactive GrabCut Segmentation")
    app.setOrganizationName("GrabCut Research")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
