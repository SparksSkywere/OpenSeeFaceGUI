![OSF.png](https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/Images/OSF.png)

# OpenSeeFace GUI

This is a fork of [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) by emilianavt, customized for personal use with a focus on the graphical user interface for camera-based face tracking.

**Note**: This project is specifically tailored for GUI-based face tracking with camera input. It implements a facial landmark detection model based on MobileNetV3, optimized for real-time performance using ONNX Runtime.

## Features

- **Graphical User Interface**: Easy-to-use GUI for camera selection and tracking configuration
- **Camera Support**: Live webcam input with preview
- **Real-time Tracking**: 15-60 FPS face landmark detection
- **Multiple Models**: Four tracking models with different speed/quality trade-offs
- **Cross-platform**: Works on Windows, Linux, and macOS

## Quick Start

### Automated Setup (Recommended)

```bash
# Run the setup script to create virtual environment and install dependencies
./setup.sh

# Launch the GUI
./run_gui.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch GUI
python gui.py
```

## GUI Features

The GUI provides:
- **Camera Selection**: Choose from available cameras with live preview
- **FPS Options**: Select tracking speed (15, 30, or 60 FPS)
- **Model Selection**: Choose between different tracking quality models
- **Real-time Preview**: Visual feedback of face tracking

## Requirements

- Python 3.7+
- Webcam or camera device
- Dependencies listed in `requirements.txt`

## Dependencies

Core dependencies:
- ONNX Runtime - Neural network inference
- OpenCV - Camera capture and image processing
- Pillow - Image handling
- NumPy - Numerical operations
- Tkinter - GUI framework (included with Python)

Install with:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the setup script: `./setup.sh`
2. Launch the GUI: `./run_gui.sh` or `python gui.py`
3. Select your camera and preferred settings
4. Start tracking

## Models

Four pretrained face landmark models are included:
- **Model 0**: Fast, low accuracy (68fps)
- **Model 1**: Balanced speed/accuracy (59fps)
- **Model 2**: Good accuracy (50fps)
- **Model 3** (default): High accuracy (44fps)

## License

This project is distributed under the BSD 2-clause license. See the original [OpenSeeFace repository](https://github.com/emilianavt/OpenSeeFace) for full license details.

## Original Project

This is a fork of [emilianavt/OpenSeeFace](https://github.com/emilianavt/OpenSeeFace). The original project includes additional features like Unity integration, VMC protocol support, and command-line tracking that are not included in this GUI-focused fork.

