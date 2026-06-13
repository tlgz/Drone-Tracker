# Drone Tracker Demo

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
