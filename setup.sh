#!/bin/bash
# =============================================================================
# OpenSeeFace Setup Script
# =============================================================================
#
# This script sets up the OpenSeeFace environment automatically:
# - Creates a Python virtual environment
# - Installs all required dependencies
# - Validates the installation
#
# Usage:
#     ./setup.sh
#
# Requirements:
#     - Python 3.7+
#     - bash shell
#
# After running this script, you can:
#     source venv/bin/activate  # Activate the virtual environment
#     python gui.py            # Launch the GUI
#
# =============================================================================

set -e  # Exit on any error

echo "========================================"
echo "OpenSeeFace Setup Script"
echo "========================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $PYTHON_VERSION detected, but Python $REQUIRED_VERSION or higher is required"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Do you want to recreate it? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Validate installation
echo "Validating installation..."
python3 -c "
import sys
try:
    import cv2
    import numpy
    import onnxruntime
    import PIL
    from pythonosc import osc_message_builder
    print('✓ All dependencies installed successfully')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use OpenSeeFace:"
echo "  source venv/bin/activate    # Activate virtual environment"
echo "  python gui.py              # Launch GUI"
echo "  ./run_gui.sh               # Or use the launcher script"
echo ""
echo "For command-line usage:"
echo "  python facetracker.py --help"
echo "  python facetracker_vmc.py --help"
echo ""
echo "Happy tracking! 🎭"