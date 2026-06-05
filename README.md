# Grab-E

<p align="center">
  <img src="src/public/splash-screen-logo.svg" alt="Grab-E logo" width="220" />
</p>

<p align="center">
  <strong>An interactive segmentation tool using a multi-color-space ensemble GrabCut workflow.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13" />
  </a>
  <a href="https://doc.qt.io/qtforpython/">
    <img src="https://img.shields.io/badge/PySide6-Qt_Desktop-41CD52?logo=qt&logoColor=white" alt="PySide6" />
  </a>
  <a href="https://opencv.org/">
    <img src="https://img.shields.io/badge/OpenCV-GrabCut-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV" />
  </a>
  <a href="https://pyinstaller.org/">
    <img src="https://img.shields.io/badge/PyInstaller-Packaging-3776AB?logo=pyinstaller&logoColor=white" alt="PyInstaller" />
  </a>
</p>

## Overview

Grab-E is a desktop image segmentation application built around scribble-guided GrabCut refinement. It lets you mark foreground and background regions, runs segmentation in the GUI, and also supports batch processing from the command line.

The project combines:

- An interactive PySide6 desktop UI
- Multi-class scribble-based segmentation
- Multi-color-space ensemble processing
- Batch GrabCut tooling for offline runs
- PyInstaller packaging for a redistributable executable

## Features

- Load an image and draw segmentation scribbles directly in the app
- Refine masks iteratively without starting over
- Run segmentation across multiple color spaces for stronger results
- Export segmentation outputs for downstream use
- Process images in batch mode from the command line
- Build a standalone Windows executable with PyInstaller

## Project Branding

The repository includes app assets under `src/public/`, including:

- `src/public/splash-screen-logo.svg`
- `src/public/how_to_use_logo.svg`
- `src/public/github_logo.svg`
- `src/public/start_with_new_image_logo.svg`

These assets are used by the application UI and can also be reused when extending the documentation or product pages.

## Requirements

- Windows x64
- Python 3.13, or the same Python minor version used to build any included native extension
- Visual C++ Build Tools if you need to compile `mgc_core`

## Installation

The recommended setup is a virtual environment plus `pip`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer, `src/run.bat` and `src/run.sh` provide launcher scripts for local use.

## Run the GUI

From the repository root:

```powershell
python src\main.py
```

On Windows, you can also start the app from `src\run.bat`.

## Batch Processing

The repository also includes a batch GrabCut CLI.

```powershell
python grabcut.py --images_dir PATH_TO_IMAGES --anns_dir PATH_TO_ANNOTATIONS --output_dir PATH_TO_OUTPUT
```

Optional batch controls include:

- `--num_images` to limit how many images are processed
- `--start_one` to choose the first 1-based item to process
- `--color_space` to select the feature space
- `--enable_majority_vote` to enable ensemble voting over a trio of color spaces

## Build a Redistributable

To generate a packaged executable:

```powershell
python -m PyInstaller GrabE.spec
```

The build output is written to `dist/`.

## Repository Layout

```text
color_space.py        Color-space conversion utilities
grabcut.py            Batch GrabCut command-line entry point
GrabE.spec            PyInstaller build spec
io_utils.py           Image and annotation helpers
mgc_api.py            Multi-color-space GrabCut API surface
mgc_core/             Native core and structured edge support
scripts/              Automation scripts for experiments and builds
src/                  PySide6 GUI application
```

## Setup Notes

The file `setup.md` contains the fuller step-by-step setup and build guide. Use it if you need the native extension build path, dependency notes, or the exact packaging workflow.

## Attribution

Grab-E is configured in the application metadata with the organization name `University of the Philippines Tacloban College`.

If you add a formal author or project page later, place it here along with a citation, lab, or institutional acknowledgment section.

## License

No license has been selected yet. Add one before distributing the project publicly.
