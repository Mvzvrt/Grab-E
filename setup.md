# Grab-E — Build & Run

This document lists the steps to build and run the project from the USB image containing the selected files.

**Disclaimer:** This repository includes both the backend processing (batch GrabCut and core native extensions) and the frontend GUI (PySide6 application), which are integrated together as a desktop based application.

## Instruction scope (Frontend vs Backend)

- **Both (shared setup):** Prerequisites, Quick setup, Build redistributable (PyInstaller)
- **Backend-specific:** Native extension (`mgc_core`) guidance, Model file, Run batch GrabCut
- **Frontend-specific:** Run the GUI

## Contents expected on directory

- `dist/`
- `mgc_core/`
- `scripts/`
- `src/`
- `color_space.py`
- `grabcut.py`
- `GrabE.spec`
- `io_utils.py`
- `mgc_api.py`
- `requirements.txt`

## Prerequisites (Both: Frontend + Backend)

- Windows x64, Python 3.13 (or the same Python minor used to build any included `.pyd`)
- Visual C++ Build Tools (only required to compile native extensions)

> Note: The steps below assume you will run commands from the repository root on the USB drive.

## Quick setup (Both: Frontend + Backend)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Native extension (`mgc_core`) guidance (Backend)

- If `mgc_core` already contains a matching compiled extension (for example a `*.pyd` built for Python 3.13/Win64), you do not need to build anything.
- If a compiled binary is not present or the target Python/OS differs, build the extension:

```powershell
cd mgc_core
python -m pip install --upgrade build setuptools wheel pybind11 Cython
python setup.py build_ext --inplace
```

You must have Visual C++ Build Tools installed for compilation on Windows.

## Model file (Backend)

Ensure the structured edge model is present:

```
mgc_core/third_party/sed/model.yml.gz
```

## Run the GUI (Frontend)

From the repository root (with the venv activated):

```powershell
python src\main.py
```

## Run batch GrabCut (Backend)

Example command:

```powershell
python grabcut.py --images_dir PATH_TO_IMAGES --anns_dir PATH_TO_ANNOTATIONS --output_dir PATH_TO_OUTPUT
```

## Build redistributable (PyInstaller) (Both: Frontend + Backend)

If you want an EXE and `dist/` is empty:

```powershell
python -m PyInstaller GrabE.spec
# resulting executable(s) appear in `dist/`
```
