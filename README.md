# Grab-E — Build & Run

This document lists the steps to build and run the project from the USB image containing the selected files.

**Disclaimer:** This repository includes both the backend processing (batch GrabCut and core native extensions) and the frontend GUI (PySide6 application), which are integrated together as a desktop based application.

## Contents expected on USB

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

## Prerequisites

- Windows x64, Python 3.13 (or the same Python minor used to build any included `.pyd`)
- Visual C++ Build Tools (only required to compile native extensions)

> Note: The steps below assume you will run commands from the repository root on the USB drive.

## Quick setup

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

## Native extension (`mgc_core`) guidance

- If `mgc_core` already contains a matching compiled extension (for example a `*.pyd` built for Python 3.13/Win64), you do not need to build anything.
- If a compiled binary is not present or the target Python/OS differs, build the extension:

```powershell
cd mgc_core
python -m pip install --upgrade build setuptools wheel pybind11 Cython
python setup.py build_ext --inplace
```

You must have Visual C++ Build Tools installed for compilation on Windows.

## Model file

Ensure the structured edge model is present:

```
mgc_core/third_party/sed/model.yml.gz
```

## Run the GUI

From the repository root (with the venv activated):

```powershell
python src\main.py
```

## Run batch GrabCut

Example command:

```powershell
python grabcut.py --images_dir PATH_TO_IMAGES --anns_dir PATH_TO_ANNOTATIONS --output_dir PATH_TO_OUTPUT
```

## Build redistributable (PyInstaller)

If you want an EXE and `dist/` is empty:

```powershell
python -m PyInstaller GrabE.spec
# resulting executable(s) appear in `dist/`
```

## Notes & caveats

- Use the same Python minor version as any prebuilt `.pyd` included in `mgc_core` — mismatched Python/ABI will cause import failures.
- `opencv-contrib-python` (listed in `requirements.txt`) provides `cv.ximgproc` on supported wheels; if no wheel exists for the target platform, you must build OpenCV+contrib from source.
- Large data files under `cs/` or `diagram/` are not included by default — copy them if required for experiments or demos.
- For offline usage on machines without internet, include the compiled `.pyd` and the `dist/` EXE(s) to avoid rebuilding and downloading packages.

## Troubleshooting (short)

- ImportError for `mgc_core.fastgeo` — the `.pyd` is missing or built for a different Python; either include matching `.pyd` or build from source.
- `cv.ximgproc` missing — install `opencv-contrib-python` or build OpenCV with contrib modules.
- Permission errors running scripts — ensure the venv is activated and PowerShell execution policy allows running scripts.

---

If you want, I can also:

- Add an automated `setup.ps1` that creates the venv, installs dependencies, and attempts to build `mgc_core` if needed.
- Remove the original `README.txt` and replace it with this file.
