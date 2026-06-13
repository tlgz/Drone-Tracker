# Drone Tracker Demo (SiamFC)

A real-time visual object tracking script built in Python, using OpenCV and the **SiamFC** (Fully-Convolutional Siamese Networks) PyTorch model. It is specifically designed to track fast-moving objects (like drones) in video files.

## Features
- **Accurate Deep Learning Tracking:** Utilizes a PyTorch implementation of the SiamFC model for robust, state-of-the-art tracking capabilities.
- **Smart Frame Navigation:** Allows you to pause the video and navigate frame-by-frame (forwards/backwards) to find the exact moment the object is clearly visible before initializing the tracker.
- **Automatic Frame Resizing:** Dynamically scales down high-resolution videos (like 4K drone footage) to fit your screen properly while dramatically speeding up the tracking performance.
- **Graceful Hardware Fallback:** Automatically detects newer, unsupported GPUs (like the RTX 50-series Blackwell architecture) and seamlessly falls back to CPU processing to prevent CUDA crashes.

## Installation

1. **Install dependencies:**
   This project is managed using `uv`. To install the required packages (PyTorch with CUDA support, OpenCV, etc.), run:
   ```bash
   uv pip install torch torchvision torchaudio opencv-python got10k numpy
   ```

2. **Download Pretrained Weights:**
   The Siamese network requires pretrained weights to function. 
   - Download the file `siamfc_alexnet_e50.pth` from the `siamfc-pytorch` documentation.
   - Place the file inside the following directory structure:
     `siamfc-pytorch/pretrained/siamfc_alexnet_e50.pth`

## Usage

1. Place your video file (named `test.mp4` by default) in the root directory.
2. Run the tracker:
   ```bash
   uv run drone_tracker.py
   ```

### Interactive Controls

**1. Navigation Phase:**
When the video first opens, it will be paused.
- `D` - Next frame
- `A` - Previous frame
- `S` - Initialize tracker on the current frame
- `Q` - Quit program

**2. Selection Phase:**
After pressing `S`, the video will pause in selection mode.
- Use your mouse to click and drag a bounding box around the drone/object.
- Press `SPACE` or `ENTER` to confirm your selection.
- Press `C` to cancel and try drawing again.

**3. Tracking Phase:**
- The script will automatically follow the object and draw a green bounding box around it.
- If the object is lost or moves off-screen, a red "KOHDE KADONNUT" (Target Lost) warning will appear.
- Press `Q` at any time to exit the viewer.
