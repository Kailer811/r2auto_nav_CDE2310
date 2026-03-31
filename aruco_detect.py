#!/usr/bin/env python3
"""
aruco_detect.py — ArUco marker detection using RPi CSI camera (OpenCV only)
============================================================================
Runs on the Raspberry Pi.
Continuously reads frames from the CSI camera and prints
to the terminal whenever an ArUco marker is detected.

Install dependencies on RPi:
    pip install opencv-contrib-python

Run on RPi:
    python3 aruco_detect.py
"""

import cv2
import numpy as np
import time

# ── settings ──────────────────────────────────────────────────────────────

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # must match your printed markers

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

HEARTBEAT_INTERVAL = 5.0   # print "still scanning" every 5 seconds


# ── open camera ───────────────────────────────────────────────────────────

def open_camera():
    """
    For RPi CSI camera with OpenCV, use the GStreamer pipeline.
    Falls back to regular VideoCapture if GStreamer isn't available.
    """

    # ── Method 1: GStreamer pipeline (best for RPi CSI camera) ───────────
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=%d, height=%d, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    ) % (FRAME_WIDTH, FRAME_HEIGHT)

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Camera opened using GStreamer (CSI).")
        return cap

    raise RuntimeError("Could not open CSI camera via GStreamer. Check camera connection.")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  ArUco Detector — OpenCV only")
    print("=" * 50)
    print("  Press Ctrl+C to stop.")
    print("=" * 50)

    # open camera
    cap = open_camera()

    # set up ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

    last_heartbeat    = time.time()
    last_detected_ids = None

    try:
        while True:
            # grab frame
            success, frame = cap.read()

            if not success or frame is None:
                print("WARNING: Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # convert to greyscale and detect markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # print results
            if ids is not None and len(ids) > 0:
                detected = [int(i[0]) for i in ids]

                # only print when IDs change — avoids spamming same message
                if detected != last_detected_ids:
                    print(f"\n>>> MARKER DETECTED! IDs: {detected}")

                    for i, corner in enumerate(corners):
                        pts = corner[0]   # 4 corner points of this marker
                        cx  = int(np.mean(pts[:, 0]))   # center x
                        cy  = int(np.mean(pts[:, 1]))   # center y
                        print(f"    Marker {int(ids[i][0])}: center at pixel ({cx}, {cy})")

                    last_detected_ids = detected
                    last_heartbeat    = time.time()

            else:
                if last_detected_ids is not None:
                    print("    Marker gone.")
                    last_detected_ids = None

                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    print("  [scanning... no marker seen]")
                    last_heartbeat = time.time()

            time.sleep(0.03)   # ~30 fps

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        print("Camera released. Done.")


if __name__ == '__main__':
    main()