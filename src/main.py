#!/usr/bin/env python3
# Filename: main.py
# -*- coding: utf-8 -*-
"""
Interactive Grab-E Application Entry Point
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from splash_screen import SplashScreen


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Grab-E")
    app.setOrganizationName("University of the Philippines Tacloban College")

    # Prefer logo.svg, then fallback to app-logo.svg.
    assets_dir = Path(__file__).parent / "public"
    icon_candidates = [assets_dir / "logo.svg", assets_dir / "app-logo.svg"]
    for icon_path in icon_candidates:
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                app.setWindowIcon(icon)
                break
    
    # Create and show splash screen
    window = SplashScreen()
    window.setWindowIcon(app.windowIcon())
    window.showMaximized()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
