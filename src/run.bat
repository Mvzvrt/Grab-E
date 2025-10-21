@echo off
REM Quick start script for Interactive GrabCut on Windows

echo ========================================
echo Interactive GrabCut Segmentation
echo ========================================
echo.

REM Check if we're in the src directory
if not exist "main.py" (
    echo Error: main.py not found!
    echo Please run this script from the src directory.
    pause
    exit /b 1
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo PySide6 not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting Interactive GrabCut Application...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo Application exited with error
    pause
)
