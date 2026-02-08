#!/usr/bin/env python3
# =============================================================================
# VMC Protocol Sender for OpenSeeFace
# =============================================================================
#
# This module implements the VMC (Virtual Motion Capture) protocol for sending
# face tracking data over OSC (Open Sound Control) via UDP. This enables
# OpenSeeFace to work with VMC-compatible applications that don't support
# the native OpenSeeFace protocol.
#
# Supported Applications:
#     - Warudo (default port 39539)
#     - VMagicMirror (default port 39540)
#     - VSeeFace (via VMC receiver)
#     - Any VMC-compatible application
#
# VMC Protocol Overview:
#     VMC uses OSC messages over UDP to transmit:
#     - Blend shapes (facial expressions): /VMC/Ext/Blend/Val
#     - Bone transforms (head/neck position/rotation): /VMC/Ext/Bone/Pos
#     - Availability status: /VMC/Ext/OK
#     - Time synchronization: /VMC/Ext/T
#
# Blend Shape Mappings:
#     This sender outputs both ARKit-style blend shapes (eyeBlinkLeft, etc.)
#     and VRM-style blend shapes (Blink_L, A, etc.) for broad compatibility.
#
# Usage:
#     from vmc_sender import VMCSender
#     
#     sender = VMCSender(ip="127.0.0.1", port=39539)
#     sender.send_tracking_data(face_info)
#
# Author: OpenSeeFace Contributors
# License: BSD 2-clause
# =============================================================================

from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder
import math
import time


class VMCSender:
    def __init__(self, ip="127.0.0.1", port=39539):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port
        self.start_time = time.time()
        
    def send_blend_shape(self, name, value):
        # Send a single blend shape value
        self.client.send_message("/VMC/Ext/Blend/Val", [name, float(value)])
    
    def send_blend_shape_apply(self):
        # Signal that blend shape updates are complete
        self.client.send_message("/VMC/Ext/Blend/Apply", [])
    
    def send_bone_transform(self, bone_name, px, py, pz, qx, qy, qz, qw):
        # Send bone position and rotation
        self.client.send_message("/VMC/Ext/Bone/Pos", [
            bone_name,
            float(px), float(py), float(pz),
            float(qx), float(qy), float(qz), float(qw)
        ])
    
    def send_available(self, available=1, calibration_state=0, tracking_status=0):
        # Send availability status - VMC/Ext/OK format
        self.client.send_message("/VMC/Ext/OK", [available, calibration_state, tracking_status])
    
    def send_time(self, time_val):
        # Send current time
        self.client.send_message("/VMC/Ext/T", [float(time_val)])
    
    def euler_to_quaternion(self, pitch, yaw, roll):
        # Convert Euler angles (in degrees) to quaternion for Unity
        # Convert to radians
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        roll = math.radians(roll)
        
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def send_root_transform(self):
        # Send root transform (required by some apps)
        # Send a neutral root position
        self.client.send_message("/VMC/Ext/Root/Pos", [
            "root",
            0.0, 0.0, 0.0,  # position
            0.0, 0.0, 0.0, 1.0  # quaternion (identity)
        ])
    
    def send_tracking_data(self, face):
        # Send face tracking data in VMC format
        # Args: face - FaceInfo object from OpenSeeFace tracker
        if face is None:
            return
        
        # Send availability (loaded=1, calibration_state=3 means calibrated, tracking=1)
        self.send_available(1 if face.success else 0, 3, 1)
        
        # Send time since start
        self.send_time(time.time() - self.start_time)
        
        # Send root transform first (some apps require this)
        self.send_root_transform()
        
        # Head bone transform using Euler angles (more reliable)
        if face.success and face.euler is not None:
            # OpenSeeFace euler is [x, y, z] in degrees (pitch, yaw, roll)
            pitch, yaw, roll = face.euler
            
            # Convert to quaternion with Unity coordinate adjustments
            qx, qy, qz, qw = self.euler_to_quaternion(-pitch, -yaw, roll)
            
            # Translation - scale for Unity units (meters)
            if face.translation is not None:
                tx = face.translation[0] / 1000.0  # Convert mm to m
                ty = -face.translation[1] / 1000.0  # Invert Y
                tz = face.translation[2] / 1000.0
            else:
                tx, ty, tz = 0, 0, 0
            
            # Send head bone
            self.send_bone_transform("Head", tx, ty, tz, qx, qy, qz, qw)
            
            # Neck follows head at reduced intensity
            neck_qx, neck_qy, neck_qz, neck_qw = self.euler_to_quaternion(-pitch * 0.3, -yaw * 0.3, roll * 0.3)
            self.send_bone_transform("Neck", tx * 0.5, ty * 0.5, tz * 0.5, neck_qx, neck_qy, neck_qz, neck_qw)
        
        # Eye gaze tracking via bone transforms
        # eye_state format: [open, eye_y, eye_x, conf] for each eye (0=right, 1=left)
        if face.eye_state is not None and len(face.eye_state) >= 2:
            # Right eye gaze
            right_eye = face.eye_state[0]
            if len(right_eye) >= 3:
                # eye_x, eye_y are pixel offsets from eye center
                # Convert to rotation angles (approximate)
                gaze_x_r = (right_eye[2] - 16) / 16.0  # Normalize to -1 to 1
                gaze_y_r = (right_eye[1] - 16) / 16.0
                # Clamp values
                gaze_x_r = max(-1, min(1, gaze_x_r)) * 15  # Max 15 degrees
                gaze_y_r = max(-1, min(1, gaze_y_r)) * 10  # Max 10 degrees
                eye_qx, eye_qy, eye_qz, eye_qw = self.euler_to_quaternion(gaze_y_r, gaze_x_r, 0)
                self.send_bone_transform("RightEye", 0, 0, 0, eye_qx, eye_qy, eye_qz, eye_qw)
            
            # Left eye gaze
            left_eye = face.eye_state[1]
            if len(left_eye) >= 3:
                gaze_x_l = (left_eye[2] - 16) / 16.0
                gaze_y_l = (left_eye[1] - 16) / 16.0
                gaze_x_l = max(-1, min(1, gaze_x_l)) * 15
                gaze_y_l = max(-1, min(1, gaze_y_l)) * 10
                eye_qx, eye_qy, eye_qz, eye_qw = self.euler_to_quaternion(gaze_y_l, gaze_x_l, 0)
                self.send_bone_transform("LeftEye", 0, 0, 0, eye_qx, eye_qy, eye_qz, eye_qw)
        
        # Eye blinks
        if face.eye_blink is not None:
            # eye_blink[0] = right eye, eye_blink[1] = left eye
            # OpenSeeFace: 1 = open, 0 = closed
            # VMC: 0 = open, 1 = closed
            right_blink = 1.0 - min(1.0, max(0.0, face.eye_blink[0]))
            left_blink = 1.0 - min(1.0, max(0.0, face.eye_blink[1]))
            
            # ARKit/Perfect Sync blend shapes
            self.send_blend_shape("eyeBlinkRight", right_blink)
            self.send_blend_shape("eyeBlinkLeft", left_blink)
            
            # VRM blend shapes
            self.send_blend_shape("Blink_R", right_blink)
            self.send_blend_shape("Blink_L", left_blink)
            self.send_blend_shape("Blink", (right_blink + left_blink) / 2)
        
        # Facial features from current_features
        if face.current_features is not None:
            features = face.current_features
            
            # Mouth
            if "mouth_open" in features:
                mouth_open = min(1.0, max(0.0, features["mouth_open"]))
                self.send_blend_shape("jawOpen", mouth_open)
                # VRM vowel shapes
                self.send_blend_shape("A", mouth_open)
                self.send_blend_shape("Aa", mouth_open)
                self.send_blend_shape("A", mouth_open)  # VRM blend shape
            
            if "mouth_wide" in features:
                mouth_wide = features["mouth_wide"]
                if mouth_wide > 0:
                    self.send_blend_shape("mouthSmileLeft", min(1.0, mouth_wide))
                    self.send_blend_shape("mouthSmileRight", min(1.0, mouth_wide))
                else:
                    self.send_blend_shape("mouthFrownLeft", min(1.0, -mouth_wide))
                    self.send_blend_shape("mouthFrownRight", min(1.0, -mouth_wide))
            
            # Mouth corners
            if "mouth_corner_updown_l" in features:
                val = features["mouth_corner_updown_l"]
                if val > 0:
                    self.send_blend_shape("mouthSmileLeft", min(1.0, val))
                else:
                    self.send_blend_shape("mouthFrownLeft", min(1.0, -val))
            
            if "mouth_corner_updown_r" in features:
                val = features["mouth_corner_updown_r"]
                if val > 0:
                    self.send_blend_shape("mouthSmileRight", min(1.0, val))
                else:
                    self.send_blend_shape("mouthFrownRight", min(1.0, -val))
            
            # Eyebrows
            if "eyebrow_updown_l" in features:
                val = features["eyebrow_updown_l"]
                if val > 0:
                    self.send_blend_shape("browInnerUp", min(1.0, val))
                else:
                    self.send_blend_shape("browDownLeft", min(1.0, -val))
            
            if "eyebrow_updown_r" in features:
                val = features["eyebrow_updown_r"]
                if val > 0:
                    self.send_blend_shape("browInnerUp", min(1.0, val))
                else:
                    self.send_blend_shape("browDownRight", min(1.0, -val))
        
        # Eye gaze blend shapes from eye_state (more accurate than features)
        # eye_state format: [open, eye_y, eye_x, conf] for each eye
        # Sends both ARKit-style and alternative naming for broad compatibility
        if face.eye_state is not None and len(face.eye_state) >= 2:
            # Right eye
            right_eye = face.eye_state[0]
            if len(right_eye) >= 3:
                gaze_x_r = (right_eye[2] - 16) / 16.0  # -1 to 1
                gaze_y_r = (right_eye[1] - 16) / 16.0
                # ARKit-style horizontal gaze
                self.send_blend_shape("eyeLookOutRight", max(0, gaze_x_r))
                self.send_blend_shape("eyeLookInRight", max(0, -gaze_x_r))
                # ARKit-style vertical gaze
                self.send_blend_shape("eyeLookUpRight", max(0, -gaze_y_r))
                self.send_blend_shape("eyeLookDownRight", max(0, gaze_y_r))
                # Alternative names (some apps use these)
                self.send_blend_shape("EyeLookOutRight", max(0, gaze_x_r))
                self.send_blend_shape("EyeLookInRight", max(0, -gaze_x_r))
                self.send_blend_shape("EyeLookUpRight", max(0, -gaze_y_r))
                self.send_blend_shape("EyeLookDownRight", max(0, gaze_y_r))
            
            # Left eye
            left_eye = face.eye_state[1]
            if len(left_eye) >= 3:
                gaze_x_l = (left_eye[2] - 16) / 16.0
                gaze_y_l = (left_eye[1] - 16) / 16.0
                # ARKit-style horizontal gaze (inverted for left eye)
                self.send_blend_shape("eyeLookInLeft", max(0, gaze_x_l))
                self.send_blend_shape("eyeLookOutLeft", max(0, -gaze_x_l))
                # ARKit-style vertical gaze
                self.send_blend_shape("eyeLookUpLeft", max(0, -gaze_y_l))
                self.send_blend_shape("eyeLookDownLeft", max(0, gaze_y_l))
                # Alternative names (some apps use these)
                self.send_blend_shape("EyeLookInLeft", max(0, gaze_x_l))
                self.send_blend_shape("EyeLookOutLeft", max(0, -gaze_x_l))
                self.send_blend_shape("EyeLookUpLeft", max(0, -gaze_y_l))
                self.send_blend_shape("EyeLookDownLeft", max(0, gaze_y_l))
        
        # Signal blend shape update complete
        self.send_blend_shape_apply()


if __name__ == "__main__":
    # Test the VMC sender
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VMC sender")
    parser.add_argument("--ip", default="127.0.0.1", help="Target IP")
    parser.add_argument("--port", type=int, default=39539, help="Target port")
    args = parser.parse_args()
    
    sender = VMCSender(args.ip, args.port)
    
    print(f"Sending test VMC data to {args.ip}:{args.port}")
    
    # Send test data
    for i in range(100):
        sender.send_available(1)
        sender.send_time(time.time())
        
        # Animate a blink
        blink_val = abs(math.sin(i * 0.1))
        sender.send_blend_shape("eyeBlinkLeft", blink_val)
        sender.send_blend_shape("eyeBlinkRight", blink_val)
        sender.send_blend_shape("Blink_L", blink_val)
        sender.send_blend_shape("Blink_R", blink_val)
        
        # Animate mouth
        mouth_val = abs(math.sin(i * 0.05))
        sender.send_blend_shape("jawOpen", mouth_val)
        sender.send_blend_shape("A", mouth_val)
        
        sender.send_blend_shape_apply()
        
        # Head rotation
        qx, qy, qz, qw = sender.euler_to_quaternion(0, math.sin(i * 0.02) * 0.3, 0)
        sender.send_bone_transform("Head", 0, 0, 0, qx, qy, qz, qw)
        
        time.sleep(0.033)  # ~30 fps
    
    print("Test complete")
