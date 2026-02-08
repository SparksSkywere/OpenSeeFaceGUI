#!/bin/bash
# =============================================================================
# OpenSeeFace GUI Launcher
# =============================================================================
#
# This script activates the Python virtual environment and launches the
# OpenSeeFace graphical user interface.
#
# Usage:
#     ./run_gui.sh
#
# Requirements:
#     - Python virtual environment at ./venv
#     - All dependencies installed (see requirements.txt)
#
# First-time setup:
#     ./setup.sh
#
# Or manually:
#     python -m venv venv
#     source venv/bin/activate
#     pip install -r requirements.txt
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at ./venv"
    echo "Run ./setup.sh to create it"
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Check and install requirements if needed
echo "Checking requirements..."
python -c "import cv2, PIL, numpy, onnxruntime" 2>/dev/null || {
    echo "Installing requirements..."
    pip install -r requirements.txt
}

# Run the GUI
echo "Starting OpenSeeFace GUI..."
python gui.py