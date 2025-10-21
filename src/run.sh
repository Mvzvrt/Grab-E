#!/bin/bash
# Quick start script for Interactive GrabCut on Linux/Mac

echo "========================================"
echo "Interactive GrabCut Segmentation"
echo "========================================"
echo ""

# Check if we're in the src directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found!"
    echo "Please run this script from the src directory."
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Show Python version
echo "Python version:"
python3 --version
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! python3 -c "import PySide6" &> /dev/null; then
    echo "PySide6 not found. Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

echo ""
echo "Starting Interactive GrabCut Application..."
echo ""

python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with error"
    read -p "Press Enter to continue..."
fi
