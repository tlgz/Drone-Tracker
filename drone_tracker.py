import cv2
import sys
import os
import numpy as np
import time
import argparse

# Add the cloned siamfc-pytorch repository to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'siamfc-pytorch'))

try:
    from siamfc import TrackerSiamFC
except ImportError:
    print("Error: Could not import TrackerSiamFC.")
    print("Please ensure you have installed the requirements: pip install torch torchvision got10k")
    sys.exit()

def resize_frame(frame, max_width=1280, max_height=720):
    """Resizes the frame to fit within the given max_width and max_height while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def main():
    parser = argparse.ArgumentParser(description="Drone Tracker using SiamFC")
    parser.add_argument('--cam', action='store_true', help="Use webcam instead of video file")
    args = parser.parse_args()

    # 1. Open the video using OpenCV
    if args.cam:
        video_path = 0 # Default webcam index
        print("Opening webcam...")
    else:
        video_path = 'test.mp4'
        print(f"Opening video file '{video_path}'...")
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_path}'.")
        print("Please make sure the video file exists or the webcam is connected.")
        sys.exit()

    # 2. Let the user navigate frames to find the object
    if args.cam:
        print("Live webcam feed active.")
        print("  's' - Select object on the current frame")
        print("  'q' - Quit program")
    else:
        print("Navigation instructions:")
        print("  'a' - Previous frame")
        print("  'd' - Next frame")
        print("  's' - Select object on the current frame")
        print("  'q' - Quit program")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    frame_idx = 0
    
    while True:
        if not args.cam:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            sys.exit()
        
        # Resize frame to fit on screen
        frame = resize_frame(frame)
            
        display_frame = frame.copy()
        if args.cam:
            cv2.putText(display_frame, "Live Feed | S: Select | Q: Quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames} | A: Prev | D: Next | S: Select | Q: Quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        cv2.imshow("Tracking Demo", display_frame)
        
        # In live cam mode we use waitKey(1) to keep the feed moving.
        # In video mode we use waitKey(0) to pause on the frame.
        key = cv2.waitKey(1 if args.cam else 0) & 0xFF
        
        if key == ord('q'):
            print("Program terminated by user.")
            sys.exit()
        elif key == ord('a') and not args.cam:
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('d') and not args.cam:
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('s') or key == 13 or key == 32:  # 's', ENTER or SPACE
            print("Please draw a bounding box around the object you want to track.")
            print("Press SPACE or ENTER to confirm the selection.")
            bbox = cv2.selectROI("Tracking Demo", frame, fromCenter=False, showCrosshair=True)
            if bbox[2] > 0 and bbox[3] > 0:
                break
            else:
                print("No object selected. Please try again or navigate.")

    # Define the path to the pretrained PyTorch weights
    # Note: You MUST download 'siamfc_alexnet_e50.pth' from the siamfc-pytorch README
    # and place it in the 'siamfc-pytorch/pretrained/' folder.
    pretrained_dir = os.path.join(os.path.dirname(__file__), 'siamfc-pytorch', 'pretrained')
    net_path = os.path.join(pretrained_dir, 'siamfc_alexnet_e50.pth')
    
    if not os.path.exists(net_path):
        print(f"Error: Pretrained model weights not found at {net_path}")
        print("Please download 'siamfc_alexnet_e50.pth' from the link in siamfc-pytorch/README.md")
        print("Google Drive link is in the README. Place the file inside the 'pretrained' folder.")
        # Create the directory to make it easier for the user
        os.makedirs(pretrained_dir, exist_ok=True)
        sys.exit()

    # Create the SiamFC tracker using the PyTorch implementation
    tracker = TrackerSiamFC(net_path=net_path)
    
    print(f"Tracking device initialized: {tracker.device}")
    if not tracker.cuda:
        print("Note: PyTorch is running on CPU.")
    
    # 3. Initialize the tracker with the selected bounding box
    # The siamfc init method takes the image array and bounding box list/tuple
    tracker.init(frame, bbox)

    # 4. Create a while loop to read the video frame by frame and update tracker
    while True:
        timer_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break

        # Resize frame to fit on screen
        frame = resize_frame(frame)

        # Update the tracker's position for the current frame
        # TrackerSiamFC returns a bounding box as an array: [x, y, w, h]
        box = tracker.update(frame)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - timer_start)

        # 5. Handle tracking success and failure
        # The standard SiamFC implementation does not return a confidence score natively, 
        # so it always returns a bounding box prediction. We assume success unless the 
        # box dimensions become invalid (e.g. tracking drifts off screen heavily).
        if box[2] > 0 and box[3] > 0 and not np.isnan(box).any():
            # Tracking was successful: draw a green rectangle around the object
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Tracking failed: the target is lost (box invalid)
            # Display "Target Lost" in red text on the screen
            cv2.putText(frame, "Target Lost", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 6. Display the video in real-time in a window
        cv2.imshow("Tracking Demo", frame)

        # Wait for 1 millisecond between frames and check if 'q' was pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Program terminated by user.")
            break

    # Release resources: close the video file and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
