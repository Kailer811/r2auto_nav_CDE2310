#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ArucoPrecisionDock(Node):
    def __init__(self):
        super().__init__('aruco_precision_dock')

        self.bridge = CvBridge()

        self.image_topic = self.declare_parameter(
            'image_topic', '/image_raw/compressed').value
        self.cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', 'cmd_vel').value
        self.target_marker_id = int(self.declare_parameter('target_marker_id', -1).value)
        self.marker_size = float(self.declare_parameter('marker_size', 0.05).value)
        self.dock_distance = float(self.declare_parameter('dock_distance', 0.18).value)

        self.linear_gain = float(self.declare_parameter('linear_gain', 0.8).value)
        self.lateral_gain = float(self.declare_parameter('lateral_gain', 3.2).value)

        self.max_linear_speed = float(
            self.declare_parameter('max_linear_speed', 0.12).value)
        self.max_angular_speed = float(
            self.declare_parameter('max_angular_speed', 1.0).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.25).value)
        self.angular_direction = float(
            self.declare_parameter('angular_direction', -1.0).value)

        self.x_threshold = float(self.declare_parameter('x_threshold', 0.01).value)
        self.realign_x_threshold = float(
            self.declare_parameter('realign_x_threshold', 0.025).value)
        self.distance_threshold = float(
            self.declare_parameter('distance_threshold', 0.015).value)
        self.loss_timeout = float(self.declare_parameter('loss_timeout', 0.75).value)
        self.show_debug_window = bool(
            self.declare_parameter('show_debug_window', False).value)

        camera_matrix_default = [
            1000.0, 0.0, 640.0,
            0.0, 1000.0, 360.0,
            0.0, 0.0, 1.0,
        ]
        dist_coeffs_default = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_matrix = np.array(
            self.declare_parameter('camera_matrix', camera_matrix_default).value,
            dtype=float).reshape((3, 3))
        self.dist_coeffs = np.array(
            self.declare_parameter('dist_coeffs', dist_coeffs_default).value,
            dtype=float).reshape((-1, 1))

        dictionary_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10)

        self.locked_id = None
        self.last_seen_time = None
        self.docked = False
        self.phase = 'align_x'

        self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(
            'ArucoPrecisionDock started. Waiting for camera images on '
            f'"{self.image_topic}".')

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to decode image: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            self.handle_marker_loss(frame)
            return

        ids_flat = ids.flatten().astype(int)
        marker_index = self.select_target_marker(ids_flat, corners)
        if marker_index is None:
            self.handle_marker_loss(frame)
            return

        selected_id = int(ids_flat[marker_index])
        if self.locked_id != selected_id:
            self.locked_id = selected_id
            self.docked = False
            self.phase = 'align_x'
            self.get_logger().info(f'Locked on ArUco marker ID {self.locked_id}')

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        rvec = rvecs[marker_index][0]
        tvec = tvecs[marker_index][0]

        self.last_seen_time = self.get_clock().now()

        depth = float(tvec[2])
        x_error = float(tvec[0])
        distance_error = depth - self.dock_distance

        self.get_logger().info(
            f'Marker {self.locked_id} pose: tvec.x={x_error:.4f}, tvec.z={depth:.4f}')

        if self.phase == 'align_x':
            if abs(x_error) <= self.x_threshold:
                self.phase = 'forward'
                self.stop_robot()
                self.get_logger().info(
                    f'X aligned for marker {self.locked_id}. Starting forward approach.')
                return

            cmd = Twist()
            cmd.angular.z = self.clamp(
                self.angular_direction * self.lateral_gain * x_error,
                -self.max_angular_speed,
                self.max_angular_speed)
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(
                f'Aligning X for marker {self.locked_id}: tx={x_error:.4f} z={depth:.4f}')
            return

        if abs(x_error) > self.realign_x_threshold:
            self.phase = 'align_x'
            self.stop_robot()
            self.get_logger().info(
                f'Marker drifted off-center (tx={x_error:.4f}). Re-aligning X.')
            return

        if abs(distance_error) <= self.distance_threshold:
            if not self.docked:
                self.get_logger().info(
                    f'Docked with marker {self.locked_id}: tx={x_error:.4f} z={depth:.4f}')
            self.docked = True
            self.stop_robot()
            return

        self.docked = False
        cmd = Twist()
        cmd.linear.x = self.clamp(
            self.linear_gain * distance_error,
            0.0,
            self.max_linear_speed)
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(
            f'Forward approach to marker {self.locked_id}: tx={x_error:.4f} '
            f'z={depth:.4f} dist_err={distance_error:.4f}')

        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow('aruco_precision_dock', frame)
            cv2.waitKey(1)

    def select_target_marker(self, ids_flat, corners):
        if self.target_marker_id >= 0:
            matches = np.where(ids_flat == self.target_marker_id)[0]
            if len(matches) == 0:
                return None
            return int(matches[0])

        areas = [cv2.contourArea(corner.astype(np.float32)) for corner in corners]
        return int(np.argmax(areas))

    def handle_marker_loss(self, frame):
        if self.last_seen_time is None:
            self.publish_search_rotation()
        else:
            elapsed = (
                self.get_clock().now() - self.last_seen_time).nanoseconds / 1e9
            if elapsed > self.loss_timeout:
                if self.locked_id is not None:
                    self.get_logger().info(
                        'Lost locked ArUco marker. Stopping so it can come back into view.')
                    self.docked = False
                    self.phase = 'align_x'
                    self.stop_robot()
                else:
                    self.publish_search_rotation()
            else:
                self.stop_robot()

        if self.show_debug_window:
            cv2.imshow('aruco_precision_dock', frame)
            cv2.waitKey(1)

    def publish_search_rotation(self):
        cmd = Twist()
        cmd.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(cmd)

    def watchdog_callback(self):
        if self.last_seen_time is None:
            return

        elapsed = (self.get_clock().now() - self.last_seen_time).nanoseconds / 1e9
        if elapsed > self.loss_timeout:
            if self.locked_id is None:
                self.publish_search_rotation()
            else:
                self.docked = False
                self.phase = 'align_x'
                self.stop_robot()

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(value, upper))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPrecisionDock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
