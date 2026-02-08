#!/usr/bin/env python3
# =============================================================================
# OpenSeeFace GUI - Graphical User Interface for Face Tracking
# =============================================================================
#
# This GUI provides an easy-to-use interface for configuring and running
# OpenSeeFace face tracking. It supports both the native OpenSeeFace protocol
# and VMC (Virtual Motion Capture) protocol for compatibility with various
# avatar applications.
#
# Features:
#     - Camera selection and live preview
#     - Configurable FPS, resolution, and tracking model
#     - Protocol selection (OpenSeeFace or VMC)
#     - Preset configurations for popular applications:
#         * VSeeFace (OpenSeeFace protocol)
#         * VTube Studio (OpenSeeFace protocol)
#         * Warudo (VMC protocol)
#         * VMagicMirror (VMC protocol)
#     - Real-time status display
#
# Usage:
#     python gui.py
#     
#     Or use the provided shell script:
#     ./run_gui.sh
#
# Requirements:
#     - Python 3.6+
#     - tkinter (usually included with Python)
#     - opencv-python
#     - pillow
#     - All OpenSeeFace dependencies (onnxruntime, numpy)
#     - python-osc (for VMC protocol support)
#
# Author: OpenSeeFace Contributors
# License: BSD 2-clause
# =============================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import threading
import time
import glob

try:
    import cv2
    from PIL import Image, ImageTk
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Missing Dependencies", f"Required packages are not installed: {e}\nPlease run: pip install -r requirements.txt")
    sys.exit(1)

# Suppress OpenCV warnings during camera detection
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

preview_running = False
preview_thread = None
cap = None
tracking_process = None

def get_cameras():
    # Detect available cameras on Linux using /dev/video* devices
    cameras = []
    try:
        # On Linux, enumerate /dev/video* devices
        video_devices = sorted(glob.glob('/dev/video*'))
        for device in video_devices:
            try:
                # Extract device number
                dev_num = int(device.replace('/dev/video', ''))
                
                # Check if this is a capture device (not metadata)
                # Metadata devices usually have "Metadata" in their name or 
                # don't have a "device" capability
                caps_path = f'/sys/class/video4linux/video{dev_num}/device/video4linux/video{dev_num}/dev'
                index_path = f'/sys/class/video4linux/video{dev_num}/index'
                
                # Skip if index > 0 (usually means it's a secondary/metadata device)
                if os.path.exists(index_path):
                    with open(index_path, 'r') as f:
                        idx = int(f.read().strip())
                        if idx > 0:
                            continue
                
                # Try to get device name from v4l2
                name_path = f'/sys/class/video4linux/video{dev_num}/name'
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                        # Skip metadata devices
                        if 'metadata' in name.lower():
                            continue
                else:
                    name = f"Camera {dev_num}"
                
                # Test if the device can actually capture video
                test_cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
                if test_cap.isOpened():
                    # Try to read a frame to confirm it's a real camera
                    ret = test_cap.grab()
                    if ret:
                        cameras.append((dev_num, name))
                    test_cap.release()
            except Exception:
                pass
    except Exception as e:
        print(f"Error detecting cameras: {e}")
    
    if not cameras:
        # Fallback: try camera indices 0-4
        for i in range(5):
            try:
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    ret = test_cap.grab()
                    if ret:
                        cameras.append((i, f"Camera {i}"))
                    test_cap.release()
            except Exception:
                pass
    
    return cameras

def get_camera_modes_v4l2(cam_id):
    """Use v4l2-ctl to get actual supported camera modes on Linux."""
    modes = []
    try:
        import subprocess
        result = subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{cam_id}', '--list-formats-ext'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            current_resolution = None
            for line in result.stdout.split('\n'):
                line = line.strip()
                # Parse resolution lines like "Size: Discrete 1920x1080"
                if 'Size:' in line and 'x' in line:
                    parts = line.split()
                    for part in parts:
                        if 'x' in part and part[0].isdigit():
                            try:
                                w, h = part.split('x')
                                current_resolution = (int(w), int(h))
                            except:
                                pass
                # Parse FPS lines like "Interval: Discrete 0.033s (30.000 fps)"
                elif 'Interval:' in line and 'fps' in line and current_resolution:
                    try:
                        # Extract fps from "(30.000 fps)" pattern
                        fps_start = line.find('(')
                        fps_end = line.find(' fps')
                        if fps_start != -1 and fps_end != -1:
                            fps = int(float(line[fps_start+1:fps_end]))
                            if fps > 0:
                                w, h = current_resolution
                                mode_key = (w, h, fps)
                                if mode_key not in [(m[0], m[1], m[2]) for m in modes]:
                                    label = f"{w}x{h} @ {fps}fps"
                                    modes.append((w, h, fps, label))
                    except:
                        pass
    except Exception as e:
        print(f"v4l2-ctl detection failed: {e}")
    return modes

def get_camera_modes(cam_id):
    """Detect available camera resolutions and FPS combinations dynamically."""
    modes = []
    
    # On Linux, try v4l2-ctl first for accurate detection
    if sys.platform == 'linux':
        modes = get_camera_modes_v4l2(cam_id)
        if modes:
            # Sort by resolution (highest first), then by fps (highest first)
            modes.sort(key=lambda x: (x[0] * x[1], x[2]), reverse=True)
            return modes
    
    # Fallback: probe camera with OpenCV for common resolutions
    # Test a wider range of resolutions and framerates
    test_resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 1440p
        (1920, 1080),  # 1080p
        (1600, 900),
        (1280, 720),   # 720p
        (1024, 576),
        (960, 540),    # 540p
        (864, 480),
        (800, 600),
        (800, 448),
        (640, 480),    # 480p
        (640, 360),    # 360p
        (424, 240),
        (352, 288),
        (320, 240),    # 240p
        (320, 180),
        (176, 144),
    ]
    test_fps = [60, 30, 15]
    
    try:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2 if sys.platform == 'linux' else cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        
        if cap.isOpened():
            tested_modes = set()
            
            for width, height in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Skip if this resolution was already tested
                if (actual_w, actual_h) in tested_modes:
                    continue
                tested_modes.add((actual_w, actual_h))
                
                # Test different framerates for this resolution
                for fps in test_fps:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Accept if actual fps is close to requested (within 20%)
                    if actual_fps > 0 and abs(actual_fps - fps) <= fps * 0.2:
                        actual_fps_int = int(round(actual_fps))
                        mode_key = (actual_w, actual_h, actual_fps_int)
                        if mode_key not in [(m[0], m[1], m[2]) for m in modes]:
                            label = f"{actual_w}x{actual_h} @ {actual_fps_int}fps"
                            modes.append((actual_w, actual_h, actual_fps_int, label))
            
            cap.release()
    except Exception as e:
        print(f"Error detecting camera modes: {e}")
    
    # Fallback if no modes detected
    if not modes:
        modes = [
            (640, 480, 30, "640x480 @ 30fps"),
            (1280, 720, 30, "1280x720 @ 30fps"),
        ]
    
    # Sort by resolution (highest first), then fps (highest first)
    modes.sort(key=lambda x: (x[0] * x[1], x[2]), reverse=True)
    return modes

def on_camera_select(event=None):
    """Update quality dropdown when camera changes."""
    global camera_modes
    if not camera_var.get():
        return
    try:
        cam_str = camera_var.get()
        cam_id = int(cam_str.split(':')[0])
        status_label.config(text="Detecting camera modes...")
        root.update()
        camera_modes = get_camera_modes(cam_id)
        quality_options = [mode[3] for mode in camera_modes]
        quality_combo['values'] = quality_options
        if quality_options:
            quality_combo.current(0)
        status_label.config(text=f"Found {len(camera_modes)} quality mode(s)")
    except Exception as e:
        print(f"Error updating camera modes: {e}")

def refresh_cameras():
    # Refresh the camera list
    global cameras, camera_options, camera_modes
    cameras = get_cameras()
    camera_options = [f"{id}: {name}" for id, name in cameras]
    camera_combo['values'] = camera_options
    if camera_options:
        camera_combo.current(0)
        on_camera_select()  # Also update quality modes
        status_label.config(text=f"Found {len(cameras)} camera(s)")
    else:
        status_label.config(text="No cameras found")
        camera_modes = []

def get_selected_quality():
    """Get the currently selected quality settings (width, height, fps)."""
    global camera_modes
    try:
        selected = quality_var.get()
        for mode in camera_modes:
            if mode[3] == selected:
                return mode[0], mode[1], mode[2]
    except:
        pass
    # Default fallback
    return 640, 480, 30

def start_preview():
    # Start the camera preview in a separate thread.
    # Opens the selected camera and displays frames in the preview area.
    # The preview runs at approximately 30 FPS to balance responsiveness with CPU usage.
    global preview_running, preview_thread, cap
    if preview_running:
        return
    
    if not camera_var.get():
        messagebox.showerror("Error", "No camera selected")
        return
    
    try:
        cam_str = camera_var.get()
        cam_id = int(cam_str.split(':')[0])
        width, height, fps = get_selected_quality()
        
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cam_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        preview_running = True
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
        status_label.config(text=f"Preview running ({width}x{height} @ {fps}fps)...")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start preview: {e}")

def stop_preview():
    # Stop the camera preview and release resources.
    # Signals the preview thread to stop, waits briefly for cleanup,
    # then releases the camera capture object.
    global preview_running, cap
    preview_running = False
    time.sleep(0.1)  # Give thread time to stop
    if cap:
        cap.release()
        cap = None
    # Clear preview
    preview_label.config(image='')
    status_label.config(text="Preview stopped")

def preview_loop():
    # Main preview loop that captures and displays camera frames.
    # Runs in a separate thread to avoid blocking the GUI.
    # Resizes frames to 320x180 for efficient display.
    global cap
    while preview_running and cap:
        ret, frame = cap.read()
        if ret:
            # Resize for preview - smaller size
            frame = cv2.resize(frame, (320, 180))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(img)
            preview_label.config(image=photo, text='')
            setattr(preview_label, 'image', photo)
        time.sleep(0.03)  # ~30 fps

def start_tracking():
    # Start the face tracking process with current settings.
    # Launches either facetracker.py (native protocol) or facetracker_vmc.py
    # (VMC protocol) based on the selected protocol. The tracking process
    # runs as a subprocess to avoid blocking the GUI.
    global tracking_process
    stop_preview()  # Stop preview before starting tracking
    
    if tracking_process and tracking_process.poll() is None:
        messagebox.showwarning("Warning", "Tracking is already running!")
        return
    
    try:
        cam_str = camera_var.get()
        cam_id = int(cam_str.split(':')[0])
        width, height, fps = get_selected_quality()
        port = int(port_var.get())
        ip = ip_var.get()
        visualize = visualize_var.get()
        model = model_var.get()
        protocol = protocol_var.get()

        if protocol == "VMC":
            # Use VMC protocol wrapper
            cmd = [sys.executable, 'facetracker_vmc.py', 
                   '-c', str(cam_id), 
                   '-F', str(fps), 
                   '-W', str(width), 
                   '-H', str(height), 
                   '--vmc-port', str(port),
                   '--vmc-ip', ip,
                   '--model', str(model),
                   '--silent', '1']
        else:
            # Use native OpenSeeFace protocol
            cmd = [sys.executable, 'facetracker.py', 
                   '-c', str(cam_id), 
                   '-F', str(fps), 
                   '-W', str(width), 
                   '-H', str(height), 
                   '-p', str(port),
                   '-i', ip,
                   '--model', str(model),
                   '--silent', '1']
        
        if visualize:
            cmd.extend(['-v', '3'])

        tracking_process = subprocess.Popen(cmd, cwd=os.getcwd())
        status_label.config(text=f"Tracking ({protocol}) → {ip}:{port}", fg="green")
        start_btn.config(state='disabled')
        stop_tracking_btn.config(state='normal')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start tracking: {e}")

def stop_tracking():
    # Stop the currently running face tracking process.
    # Terminates the subprocess and updates the UI state.
    global tracking_process
    if tracking_process:
        tracking_process.terminate()
        tracking_process = None
        status_label.config(text="Tracking stopped", fg="blue")
        start_btn.config(state='normal')
        stop_tracking_btn.config(state='disabled')

def on_closing():
    # Handle window close event - cleanup resources before exit.
    stop_preview()
    stop_tracking()
    root.destroy()

# Create main window
root = tk.Tk()
root.title("OpenSeeFace GUI")
root.minsize(480, 520)
root.resizable(True, True)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Main frame with padding
main_frame = ttk.Frame(root, padding="5")
main_frame.grid(row=0, column=0, sticky='nsew')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)

# Title
title_label = tk.Label(main_frame, text="OpenSeeFace Tracker", font=('Arial', 14, 'bold'))
title_label.grid(row=0, column=0, pady=5, sticky='ew')

# Camera selection frame
cam_frame = ttk.LabelFrame(main_frame, text="Camera Settings", padding="3")
cam_frame.grid(row=1, column=0, sticky='ew', pady=3, padx=5)
cam_frame.columnconfigure(1, weight=1)

tk.Label(cam_frame, text="Camera:").grid(row=0, column=0, sticky='w', padx=3, pady=2)
camera_var = tk.StringVar()
camera_combo = ttk.Combobox(cam_frame, textvariable=camera_var, state='readonly')
camera_combo.grid(row=0, column=1, padx=3, pady=2, sticky='ew')
camera_combo.bind('<<ComboboxSelected>>', on_camera_select)
refresh_btn = tk.Button(cam_frame, text="↻", command=refresh_cameras, width=3)
refresh_btn.grid(row=0, column=2, padx=3, pady=2)

settings_row = tk.Frame(cam_frame)
settings_row.grid(row=1, column=0, columnspan=3, sticky='w', pady=2)
tk.Label(settings_row, text="Quality:").pack(side=tk.LEFT, padx=3)
quality_var = tk.StringVar()
quality_combo = ttk.Combobox(settings_row, textvariable=quality_var, state='readonly', width=22)
quality_combo.pack(side=tk.LEFT, padx=3)

# Network settings frame
net_frame = ttk.LabelFrame(main_frame, text="Network", padding="3")
net_frame.grid(row=2, column=0, sticky='ew', pady=3, padx=5)

# Protocol and program presets
PROTOCOL_PROGRAMS = {
    "OpenSeeFace": {
        "VSeeFace": "11573",
        "VTube Studio": "21412",
        "Custom": ""
    },
    "VMC": {
        "Warudo": "39539",
        "VMagicMirror": "39540",
        "VSeeFace": "39539",
        "Custom": ""
    }
}

def on_protocol_change(event=None):
    protocol = protocol_var.get()
    programs = list(PROTOCOL_PROGRAMS[protocol].keys())
    program_combo['values'] = programs
    program_combo.current(0)
    on_program_select(None)

def on_program_select(event):
    protocol = protocol_var.get()
    program = program_var.get()
    if program in PROTOCOL_PROGRAMS[protocol]:
        port = PROTOCOL_PROGRAMS[protocol][program]
        if port:
            port_var.set(port)

# Protocol selection
proto_row = tk.Frame(net_frame)
proto_row.pack(fill='x', pady=2)
tk.Label(proto_row, text="Protocol:").pack(side=tk.LEFT, padx=3)
protocol_var = tk.StringVar(value="OpenSeeFace")
protocol_combo = ttk.Combobox(proto_row, textvariable=protocol_var, values=["OpenSeeFace", "VMC"], state='readonly', width=12)
protocol_combo.pack(side=tk.LEFT, padx=3)
protocol_combo.bind('<<ComboboxSelected>>', on_protocol_change)

tk.Label(proto_row, text="App:").pack(side=tk.LEFT, padx=3)
program_var = tk.StringVar(value="VSeeFace")
program_combo = ttk.Combobox(proto_row, textvariable=program_var, values=list(PROTOCOL_PROGRAMS["OpenSeeFace"].keys()), state='readonly', width=12)
program_combo.pack(side=tk.LEFT, padx=3)
program_combo.bind('<<ComboboxSelected>>', on_program_select)

net_row2 = tk.Frame(net_frame)
net_row2.pack(fill='x', pady=2)
tk.Label(net_row2, text="IP:").pack(side=tk.LEFT, padx=3)
ip_var = tk.StringVar(value="127.0.0.1")
ip_entry = tk.Entry(net_row2, textvariable=ip_var, width=12)
ip_entry.pack(side=tk.LEFT, padx=3)
tk.Label(net_row2, text="Port:").pack(side=tk.LEFT, padx=3)
port_var = tk.StringVar(value="11573")
port_entry = tk.Entry(net_row2, textvariable=port_var, width=7)
port_entry.pack(side=tk.LEFT, padx=3)

# Tracking settings frame
track_frame = ttk.LabelFrame(main_frame, text="Tracking", padding="3")
track_frame.grid(row=3, column=0, sticky='ew', pady=3, padx=5)

track_row = tk.Frame(track_frame)
track_row.pack(fill='x', pady=2)
tk.Label(track_row, text="Model:").pack(side=tk.LEFT, padx=3)
model_var = tk.IntVar(value=3)
model_combo = ttk.Combobox(track_row, textvariable=model_var, values=[-3, -2, -1, 0, 1, 2, 3, 4], state='readonly', width=4)  # type: ignore
model_combo.pack(side=tk.LEFT, padx=3)
tk.Label(track_row, text="(Higher=better)", fg="gray").pack(side=tk.LEFT, padx=3)
visualize_var = tk.BooleanVar(value=False)
visualize_check = tk.Checkbutton(track_row, text="Show Viz", variable=visualize_var)
visualize_check.pack(side=tk.LEFT, padx=10)

# Preview frame
preview_outer = ttk.LabelFrame(main_frame, text="Preview", padding="3")
preview_outer.grid(row=4, column=0, sticky='nsew', pady=3, padx=5)
main_frame.rowconfigure(4, weight=1)

btn_frame = tk.Frame(preview_outer)
btn_frame.pack(fill='x', pady=3)
start_preview_btn = tk.Button(btn_frame, text="▶ Preview", command=start_preview)
start_preview_btn.pack(side=tk.LEFT, padx=5, expand=True, fill='x')
stop_preview_btn = tk.Button(btn_frame, text="■ Stop", command=stop_preview)
stop_preview_btn.pack(side=tk.LEFT, padx=5, expand=True, fill='x')

preview_label = tk.Label(preview_outer, bg='#222', text="No preview", fg='#666')
preview_label.pack(fill='both', expand=True, pady=3)

# Status label
status_label = tk.Label(main_frame, text="Ready", fg="blue", font=('Arial', 9))
status_label.grid(row=5, column=0, pady=3, sticky='ew')

# Control buttons frame
control_frame = tk.Frame(main_frame)
control_frame.grid(row=6, column=0, sticky='ew', pady=5, padx=5)
control_frame.columnconfigure(0, weight=1)
control_frame.columnconfigure(1, weight=1)

start_btn = tk.Button(control_frame, text="▶ Start Tracking", command=start_tracking, 
                      bg='#228B22', fg='white', font=('Arial', 11, 'bold'))
start_btn.grid(row=0, column=0, sticky='ew', padx=3, pady=3, ipady=8)

stop_tracking_btn = tk.Button(control_frame, text="■ Stop Tracking", command=stop_tracking,
                               bg='#B22222', fg='white', font=('Arial', 11, 'bold'),
                               state='disabled')
stop_tracking_btn.grid(row=0, column=1, sticky='ew', padx=3, pady=3, ipady=8)

# Load cameras on startup
cameras = get_cameras()
camera_modes = []  # Will be populated when camera is selected
camera_options = [f"{id}: {name}" for id, name in cameras]
camera_combo['values'] = camera_options
if camera_options:
    camera_combo.current(0)
    # Detect quality modes for the first camera
    on_camera_select()
    status_label.config(text=f"Found {len(cameras)} camera(s) - Ready to track")

root.mainloop()