#!/usr/bin/env python3
# Filename: how_to_use_page.py
# -*- coding: utf-8 -*-
"""How-to-use template page for Grab-E."""

from pathlib import Path
from datetime import date

from PySide6.QtCore import QSize, Qt
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
                font-size: 38px;
                font-weight: 700;
                color: #ffffff;
                margin-top: 8px;
                margin-bottom: 10px;
            }
            QLabel#h2 {
                font-size: 25px;
                font-weight: 600;
                color: #ffffff;
                margin-top: 18px;
                margin-bottom: 12px;
            }
            QLabel#h3 {
                font-size: 20px;
                font-weight: 600;
                color: #e8f0fe;
                margin-top: 12px;
                margin-bottom: 8px;
            }
            QLabel#guideKicker {
                font-size: 12px;
                font-weight: 700;
                color: #8ab4f8;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            QLabel#leadText {
                font-size: 16px;
                color: #e1e3e6;
                line-height: 1.5;
                margin-bottom: 6px;
            }
            QLabel#metaChip {
                font-size: 12px;
                color: #d2e3fc;
                background-color: #1f2a37;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 4px 10px;
            }
            QFrame#tocCard {
                background-color: transparent;
                border: none;
                border-radius: 0px;
            }
            QLabel#tocTitle {
                font-size: 25px;
                font-weight: 700;
                color: #ffffff;
            }
            QPushButton#tocLink {
                border: none;
                background-color: transparent;
                color: #ffffff;
                font-size: 15px;
                text-align: left;
                padding: 0px;
                margin: 0px;
            }
            QPushButton#tocSubLink {
                border: none;
                background-color: transparent;
                color: #d0d0d0;
                font-size: 15px;
                text-align: left;
                padding: 0px 0px 0px 18px;
                margin: 0px;
            }
            QPushButton#tocLink:hover,
            QPushButton#tocSubLink:hover {
                color: #d8ecff;
                text-decoration: underline;
            }
            QPushButton#tocLink:pressed,
            QPushButton#tocSubLink:pressed,
            QPushButton#tocLink:focus,
            QPushButton#tocSubLink:focus {
                background-color: transparent;
                border: none;
                color: #ffffff;
                outline: none;
            }
            QLabel#bodyText {
                font-size: 15px;
                color: #d0d0d0;
                line-height: 1.5;
                margin-bottom: 16px;
            }
            QLabel#creditsText {
                font-size: 12px;
                color: #9aa0a6;
            }
            QPushButton#creditsIcon {
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
            QFrame#calloutCard {
                background-color: #1f1f1f;
                border: 1px solid #2f3640;
                border-radius: 10px;
            }
            QLabel#calloutTitle {
                font-size: 14px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#calloutBody {
                font-size: 14px;
                color: #d0d0d0;
                line-height: 1.5;
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
            QLabel#imageLabel {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                padding: 6px;
            }
            QFrame#sectionDivider {
                background-color: #2e2e2e;
                min-height: 1px;
                max-height: 1px;
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
        self.h1_title = h1_title

        guide_kicker = QLabel("Product guide")
        guide_kicker.setObjectName("guideKicker")

        lead_text = QLabel(
            "Follow this step-by-step guide to load an image, create classes, draw scribbles, "
            "run segmentation, and save results with confidence."
        )
        lead_text.setObjectName("leadText")
        lead_text.setWordWrap(True)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(8)
        eta_chip = QLabel("Estimated time: 5-10 min")
        eta_chip.setObjectName("metaChip")
        audience_chip = QLabel("Audience: New users")
        audience_chip.setObjectName("metaChip")
        meta_row.addWidget(eta_chip)
        meta_row.addWidget(audience_chip)
        meta_row.addStretch()

        callout_card = QFrame()
        callout_card.setObjectName("calloutCard")
        callout_layout = QVBoxLayout(callout_card)
        callout_layout.setContentsMargins(12, 10, 12, 10)
        callout_layout.setSpacing(4)

        callout_title = QLabel("Before you begin")
        callout_title.setObjectName("calloutTitle")
        callout_body = QLabel(
            "Prepare a sample image and decide class labels first. This reduces rework while drawing "
            "scribbles and helps keep segmentation consistent across runs."
        )
        callout_body.setObjectName("calloutBody")
        callout_body.setWordWrap(True)

        callout_layout.addWidget(callout_title)
        callout_layout.addWidget(callout_body)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.scroll_area = scroll_area

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

        anc_body_5 = QLabel("Click the edit class button to edit existing class names and colors")
        anc_body_5.setObjectName("bodyText")
        anc_body_5.setWordWrap(True)
        anc_image_widget_5 = self._load_screenshot("add_new_class_5.png", 960, 516)

        anc_body_6 = QLabel("Click the delete class button to delete existing classes")
        anc_body_6.setObjectName("bodyText")
        anc_body_6.setWordWrap(True)
        anc_image_widget_6 = self._load_screenshot("add_new_class_6.png", 960, 516)

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

        run_segmentation = QLabel("4. Run segmentation")
        run_segmentation.setObjectName("h2")

        rs_body = QLabel("Select single color space mode or multiple color space mode")
        rs_body.setObjectName("bodyText")
        rs_body.setWordWrap(True)
        rs_image_widget_1 = self._load_screenshot("rs_1.png", 960, 516)

        single_color_space_mode = QLabel("4.1. Single color space mode")
        single_color_space_mode.setObjectName("h3")

        scsm_body = QLabel("Select color space for segmentation. Ruderman LAB is selected by default due to achieving the highest performance.")
        scsm_body.setObjectName("bodyText")
        scsm_body.setWordWrap(True)
        scsm_image_widget = self._load_screenshot("rs_2.png", 960, 516)

        ensemble_mode = QLabel("4.2. Multicolor Space Ensemble Mode")
        ensemble_mode.setObjectName("h3")

        em_body = QLabel("Select the color spaces to include in the ensemble. Ruderman LAB, JzAzBz, and Oklch are selected by default.")
        em_body.setObjectName("bodyText")
        em_body.setWordWrap(True)
        em_image_widget = self._load_screenshot("rs_5.png", 960, 516)

        em_body_2 = QLabel("Select the model to resolve ties in voting. The first color space model is selected by default due to having the highest performance.")
        em_body_2.setObjectName("bodyText")
        em_body_2.setWordWrap(True)
        em_image_widget_2 = self._load_screenshot("rs_6.png", 960, 516)

        scsm_body_2 = QLabel("Select the number of iterations for GrabCut. Five iterations is selected by default.")
        scsm_body_2.setObjectName("bodyText")
        scsm_body_2.setWordWrap(True)
        scsm_image_widget_2 = self._load_screenshot("rs_3.png", 960, 516)

        scsm_body_3 = QLabel("Adjust the opacity of overlaying the generated mask over the image. 50%% opacity is selected by default.")
        scsm_body_3.setObjectName("bodyText")
        scsm_body_3.setWordWrap(True)
        scsm_image_widget_3 = self._load_screenshot("rs_4.png", 960, 516)

        rs_body_2 = QLabel("Click the Segment button to generate segmentation mask")
        rs_body_2.setObjectName("bodyText")
        rs_body_2.setWordWrap(True)
        rs_image_widget_2 = self._load_screenshot("rs_7.png", 960, 516)

        rs_body_3 = QLabel("For refinement, click the Refine button to use the current color models from GrabCut to fix errors after drawing correction scribbles. Otherwise, click the Segment button to use new color models after drawing correction scribbles.")
        rs_body_3.setObjectName("bodyText")
        rs_body_3.setWordWrap(True)
        rs_image_widget_3 = self._load_screenshot("rs_8.png", 960, 516)

        file_saving = QLabel("5. Save files")
        file_saving.setObjectName("h2")

        fs_body = QLabel("Click the Save Mask button to save the generated masks as a .png file")
        fs_body.setObjectName("bodyText")
        fs_body.setWordWrap(True)
        fs_image_widget = self._load_screenshot("fs_1.png", 960, 367)

        fs_body_2 = QLabel("Click the Save Scribbles button to save the drawn scribbles as a .png file")
        fs_body_2.setObjectName("bodyText")
        fs_body_2.setWordWrap(True)
        fs_image_widget_2 = self._load_screenshot("fs_2.png", 960, 367)

        fs_body_3 = QLabel("Click the Reset All button to clear all drawn scribbles and generated masks.")
        fs_body_3.setObjectName("bodyText")
        fs_body_3.setWordWrap(True)
        fs_image_widget_3 = self._load_screenshot("fs_3.png", 960, 516)

        # Table of contents for quick in-page navigation.
        self._toc_targets = {
            "h1": h1_title,
            "h2_open_image": h2_open_image,
            "h2_add_new_classes": add_new_classes,
            "h2_drawing_scribbles": drawing_scribbles,
            "h2_run_segmentation": run_segmentation,
            "h3_single_color_space_mode": single_color_space_mode,
            "h3_ensemble_mode": ensemble_mode,
            "h2_file_saving": file_saving,
        }

        toc_card = QFrame()
        toc_card.setObjectName("tocCard")
        toc_layout = QVBoxLayout(toc_card)
        toc_layout.setContentsMargins(0, 6, 0, 10)
        toc_layout.setSpacing(6)

        toc_title = QLabel("Table of contents")
        toc_title.setObjectName("tocTitle")

        toc_layout.addWidget(toc_title)

        toc_items = [
            ("How To Use Grab-E", "h1", False),
            ("1. Open new image", "h2_open_image", True),
            ("2. Add new classes", "h2_add_new_classes", True),
            ("3. Draw scribbles to segment", "h2_drawing_scribbles", True),
            ("4. Run segmentation", "h2_run_segmentation", True),
            ("4.1. Single color space mode", "h3_single_color_space_mode", True),
            ("4.2. Multicolor Space Ensemble Mode", "h3_ensemble_mode", True),
            ("5. Save files", "h2_file_saving", True),
        ]

        for text, section_key, is_sub_item in toc_items:
            toc_btn = QPushButton(text)
            toc_btn.setCursor(Qt.PointingHandCursor)
            toc_btn.setFlat(True)
            toc_btn.setFocusPolicy(Qt.NoFocus)
            toc_btn.setObjectName("tocSubLink" if is_sub_item else "tocLink")
            toc_btn.clicked.connect(lambda _checked=False, key=section_key: self._handle_toc_link(key))
            toc_layout.addWidget(toc_btn, alignment=Qt.AlignLeft)

        content_layout.addWidget(toc_card)
        content_layout.addWidget(guide_kicker)
        content_layout.addWidget(h1_title)
        content_layout.addWidget(lead_text)
        content_layout.addLayout(meta_row)
        content_layout.addWidget(callout_card)
        content_layout.addSpacing(16)

        content_layout.addWidget(h2_open_image)
        content_layout.addWidget(body_text)
        content_layout.addWidget(image_widget)
        content_layout.addSpacing(24)
        content_layout.addWidget(body_text_2)
        content_layout.addWidget(image_widget_2)
        content_layout.addSpacing(30)
        content_layout.addWidget(self._create_section_divider())
        content_layout.addSpacing(20)
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
        content_layout.addSpacing(24)
        content_layout.addWidget(anc_body_5)
        content_layout.addWidget(anc_image_widget_5)
        content_layout.addSpacing(24)
        content_layout.addWidget(anc_body_6)
        content_layout.addWidget(anc_image_widget_6)
        content_layout.addSpacing(30)
        content_layout.addWidget(self._create_section_divider())
        content_layout.addSpacing(20)
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
        content_layout.addSpacing(30)
        content_layout.addWidget(self._create_section_divider())
        content_layout.addSpacing(20)
        content_layout.addWidget(run_segmentation)
        content_layout.addWidget(rs_body)
        content_layout.addWidget(rs_image_widget_1)
        content_layout.addSpacing(24)
        content_layout.addWidget(single_color_space_mode)
        content_layout.addWidget(scsm_body)
        content_layout.addWidget(scsm_image_widget)
        content_layout.addSpacing(24)
        content_layout.addWidget(ensemble_mode)
        content_layout.addWidget(em_body)
        content_layout.addWidget(em_image_widget)
        content_layout.addSpacing(24)
        content_layout.addWidget(em_body_2)
        content_layout.addWidget(em_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(scsm_body_2)
        content_layout.addWidget(scsm_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(scsm_body_3)
        content_layout.addWidget(scsm_image_widget_3)
        content_layout.addSpacing(24)
        content_layout.addWidget(rs_body_2)
        content_layout.addWidget(rs_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(rs_body_3)
        content_layout.addWidget(rs_image_widget_3)
        content_layout.addSpacing(30)
        content_layout.addWidget(self._create_section_divider())
        content_layout.addSpacing(20)
        content_layout.addWidget(file_saving)
        content_layout.addWidget(fs_body)
        content_layout.addWidget(fs_image_widget)
        content_layout.addSpacing(24)
        content_layout.addWidget(fs_body_2)
        content_layout.addWidget(fs_image_widget_2)
        content_layout.addSpacing(24)
        content_layout.addWidget(fs_body_3)
        content_layout.addWidget(fs_image_widget_3)
        content_layout.addStretch()

        scroll_area.setWidget(content)

        root_layout.addLayout(top_row)
        root_layout.addWidget(scroll_area, 1)
        root_layout.addLayout(self._create_credits_row())

    def _handle_toc_link(self, key: str):
        """Scroll the help content to the requested section from the TOC."""
        if key == "h1":
            self.scroll_area.verticalScrollBar().setValue(0)
            return

        target = self._toc_targets.get(key)
        if target is None:
            return

        content_widget = self.scroll_area.widget()
        if content_widget is None:
            return

        target_y = target.mapTo(content_widget, target.rect().topLeft()).y()
        self.scroll_area.verticalScrollBar().setValue(max(0, target_y - 12))

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
        image_label.setObjectName("imageLabel")
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        
        return image_label

    def _create_section_divider(self) -> QFrame:
        """Create a subtle divider between major step groups."""
        divider = QFrame()
        divider.setObjectName("sectionDivider")
        return divider

    def _create_credits_row(self) -> QHBoxLayout:
        """Create a muted credits footer with optional GitHub icon."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 0)
        row.setSpacing(6)

        github_icon_path = Path(__file__).parent / "public" / "github_logo.svg"
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
