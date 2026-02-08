#!/usr/bin/env python3
# =============================================================================
# OpenSeeFace VMC Protocol Wrapper
# =============================================================================
#
# This script wraps the OpenSeeFace face tracker to output tracking data
# using the VMC (Virtual Motion Capture) protocol instead of the native
# OpenSeeFace binary protocol.
#
# The native OpenSeeFace protocol is optimized for VSeeFace and VTube Studio,
# but many other applications (like Warudo) require VMC protocol. This wrapper
# bridges that gap by:
#
# 1. Running the standard OpenSeeFace face detection and tracking
# 2. Converting tracking data to VMC-compatible format
# 3. Sending data via OSC over UDP to the target application
#
# Usage:
#     python facetracker_vmc.py -c 0 --vmc-port 39539
#     
#     Common options:
#         -c, --capture     Camera ID or video file (default: 0)
#         --vmc-ip         Target IP address (default: 127.0.0.1)
#         --vmc-port       Target port (default: 39539 for Warudo)
#         -W, --width      Camera width (default: 640)
#         -H, --height     Camera height (default: 480)
#         -F, --fps        Frames per second (default: 24)
#         --model          Tracking model quality 0-3 (default: 3)
#         -v, --visualize  Show tracking visualization
#
# Example for Warudo:
#     python facetracker_vmc.py -c 0 --vmc-port 39539 --model 3
#
# Example for VMagicMirror:
#     python facetracker_vmc.py -c 0 --vmc-port 39540 --model 3
#
# Author: OpenSeeFace Contributors
# License: BSD 2-clause
# =============================================================================

import copy
import os
import sys
import argparse
import time
import cv2
import numpy as np

from input_reader import InputReader, VideoReader, try_int
from tracker import Tracker, get_model_base_path
from vmc_sender import VMCSender

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--vmc-ip", help="Set IP address for sending VMC data", default="127.0.0.1")
parser.add_argument("--vmc-port", type=int, help="Set port for sending VMC data", default=39539)
parser.add_argument("-W", "--width", type=int, help="Set camera width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set camera height", default=480)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 0 for verbose output (default: silent)", default=1)
parser.add_argument("--model", type=int, help="Tracking model (higher = better quality, slower)", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="Path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, gaze tracking is enabled", default=1)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces", default=1)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="Seconds after which feature values stop updating", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)

if sys.platform == 'linux':
    parser.add_argument("--dformat", type=str, help="Set device format (MJPG, YUYV, RGB3, ...)", default=None)

args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

# Initialize VMC sender
vmc_sender = VMCSender(args.vmc_ip, args.vmc_port)
print(f"VMC Protocol sender initialized: {args.vmc_ip}:{args.vmc_port}")

# Initialize input reader
fps = args.fps
if sys.platform == 'linux' and hasattr(args, 'dformat') and args.dformat:
    input_reader = InputReader(args.capture, 0, args.width, args.height, fps, dcap=args.dformat)
else:
    input_reader = InputReader(args.capture, 0, args.width, args.height, fps)

if type(input_reader.reader) == VideoReader:
    fps = 0

first = True
height = 0
width = 0
tracker = None
frame_count = 0

is_camera = args.capture == str(try_int(args.capture))

print("Starting VMC face tracking...")
print("Press Ctrl+C to stop")

try:
    while input_reader.is_open():
        if not input_reader.is_ready():
            time.sleep(0.001)
            continue

        ret, frame = input_reader.read()
        if not ret:
            if is_camera:
                time.sleep(0.001)
                continue
            else:
                break

        frame_count += 1

        if args.mirror_input:
            frame = cv2.flip(frame, 1)

        if first:
            first = False
            height, width, channels = frame.shape
            tracker = Tracker(
                width, height,
                threshold=args.threshold,
                max_threads=args.max_threads,
                max_faces=args.faces,
                discard_after=args.discard_after,
                scan_every=args.scan_every,
                silent=False if args.silent == 0 else True,
                model_type=args.model,
                model_dir=args.model_dir,
                no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True,
                detection_threshold=args.detection_threshold,
                use_retinaface=False,
                max_feature_updates=args.max_feature_updates,
                static_model=True if args.no_3d_adapt == 1 else False,
                try_hard=args.try_hard == 1
            )
            print(f"Tracker initialized: {width}x{height}")

        # Run face detection/tracking
        assert tracker is not None, "Tracker should be initialized"
        faces = tracker.predict(frame)

        # Send VMC data for each detected face
        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            
            if f.eye_blink is None:
                f.eye_blink = [1, 1]
            
            # Send tracking data via VMC (includes eye gaze)
            vmc_sender.send_tracking_data(f)

        # Visualization
        if args.visualize != 0:
            for f in faces:
                for pt_num, (x, y, c) in enumerate(f.lms):
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    color = (0, 255, 0)
                    if pt_num >= 66:
                        color = (255, 255, 0)
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        cv2.circle(frame, (y, x), 1, color, -1)
            
            cv2.imshow('OpenSeeFace VMC', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    input_reader.close()
    if args.visualize != 0:
        cv2.destroyAllWindows()
    print("VMC face tracking stopped")
