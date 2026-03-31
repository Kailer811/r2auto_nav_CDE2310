#!/usr/bin/env python3
"""
aruco_node.py — ArUco Detection Node
=====================================
Runs ON YOUR LAPTOP.
Subscribes to /camera/image_raw published by camera_node on the RPi.
Runs OpenCV ArUco detection on your faster laptop CPU.
Publishes detection results to /aruco_detected.
Prints detections to terminal.

ros2 run auto_nav aruco_node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np
import time


# ==========================================
# SETTINGS
# ==========================================

ARUCO_DICT  = cv2.aruco.DICT_4X4_50

MARKER_SIZE = 0.05   # physical marker size in metres (5cm)

# camera intrinsic matrix
# these are rough defaults — adjust if you have calibrated values
camera_matrix = np.array([
    [500,   0, 160],   # scaled for 320x240 resolution
    [  0, 500, 120],
    [  0,   0,   1]
], dtype=float)
dist_coeffs = np.zeros((5, 1))

HEARTBEAT_INTERVAL = 5.0


class ArucoNode(Node):

    def __init__(self):
        super().__init__('aruco_node')

        # subscriber — receives raw frames from RPi camera_node
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # publisher — sends detection results to auto_nav2
        # message format: "DETECTED:id:x:y:distance" or "NONE"
        self.pub = self.create_publisher(String, '/aruco_detected', 10)

        # set up ArUco detector (compatible with older OpenCV)
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.last_detected_ids = None
        self.last_heartbeat    = time.time()
        self.frame_count       = 0

        self.get_logger().info(
            'ArUco node started. Waiting for frames on /camera/image_raw...'
        )

    # ==========================================
    # CALLED EVERY TIME A FRAME ARRIVES
    # ==========================================

    def image_callback(self, msg):
        self.frame_count += 1

        # convert ROS2 Image message back to OpenCV frame
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        # convert to greyscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0:
            detected = [int(i[0]) for i in ids]

            # estimate 3D pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, camera_matrix, dist_coeffs
            )

            # only print + publish when IDs change
            if detected != self.last_detected_ids:
                self.get_logger().info(f'MARKER DETECTED — IDs: {detected}')

                for i, marker_id in enumerate(ids.flatten()):
                    x = tvecs[i][0][0]   # left/right
                    y = tvecs[i][0][1]   # up/down
                    z = tvecs[i][0][2]   # distance

                    self.get_logger().info(
                        f'  ID {marker_id}: '
                        f'x={x:+.3f}m  y={y:+.3f}m  distance={z:.3f}m'
                    )

                    # publish detection to /aruco_detected
                    # format: "DETECTED:id:x:y:distance"
                    detection_msg = String()
                    detection_msg.data = (
                        f'DETECTED:{marker_id}:{x:.4f}:{y:.4f}:{z:.4f}'
                    )
                    self.pub.publish(detection_msg)

                self.last_detected_ids = detected
                self.last_heartbeat    = time.time()

        else:
            # no marker — publish NONE so auto_nav2 knows it's clear
            if self.last_detected_ids is not None:
                self.get_logger().info('Marker gone.')
                none_msg = String()
                none_msg.data = 'NONE'
                self.pub.publish(none_msg)
                self.last_detected_ids = None

            # heartbeat
            if time.time() - self.last_heartbeat > HEARTBEAT_INTERVAL:
                self.get_logger().info(
                    f'[scanning... no marker seen] '
                    f'frames processed: {self.frame_count}'
                )
                self.last_heartbeat = time.time()

        # show camera feed
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ArUco node.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
