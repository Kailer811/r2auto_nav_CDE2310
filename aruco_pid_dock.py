#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class ArucoPidDock(Node):
    def __init__(self):
        super().__init__('aruco_pid_dock')

        self.bridge = CvBridge()

        self.image_topic = self.declare_parameter(
            'image_topic', '/image_raw/compressed').value
        self.cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', 'cmd_vel').value
        self.primary_marker_id = int(
            self.declare_parameter('primary_marker_id', 2).value)
        self.secondary_marker_id = int(
            self.declare_parameter('secondary_marker_id', 4).value)
        self.marker_size = float(
            self.declare_parameter('marker_size', 0.04).value)
        self.final_distance = float(
            self.declare_parameter('final_distance', 0.05).value)

        self.linear_gain = float(self.declare_parameter('linear_gain', 4.0).value)
        self.angular_gain_x = float(self.declare_parameter('angular_gain_x', 4.0).value)
        self.angular_gain_yaw = float(self.declare_parameter('angular_gain_yaw', 0.5).value)
        self.angular_direction = float(
            self.declare_parameter('angular_direction', 1.0).value)
        self.search_direction_gain = float(
            self.declare_parameter('search_direction_gain', 1.0).value)
        self.max_linear_speed = float(
            self.declare_parameter('max_linear_speed', 0.20).value)
        self.max_angular_speed = float(
            self.declare_parameter('max_angular_speed', 0.9).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.35).value)
        self.loss_timeout = float(
            self.declare_parameter('loss_timeout', 0.75).value)
        self.stop_nav_loss_timeout = float(
            self.declare_parameter('stop_nav_loss_timeout', 5.0).value)
        self.stop_nav_post_dock_timeout = float(
            self.declare_parameter('stop_nav_post_dock_timeout', 20.0).value)

        self.align_x_threshold = float(
            self.declare_parameter('align_x_threshold', 0.05).value)
        self.align_yaw_threshold = float(
            self.declare_parameter('align_yaw_threshold', 0.08).value)
        self.distance_threshold = float(
            self.declare_parameter('distance_threshold', 0.05).value)
        self.show_debug_window = bool(
            self.declare_parameter('show_debug_window', True).value)

        self.camera_matrix = np.array([
            [475, 0.0, 320],
            [0.0, 475, 240],
            [0.0, 0.0, 1.0]
            ], dtype=float)
        self.dist_coeffs = np.zeros((5,1))

        dictionary_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.detected_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.objA_trigger_pub = self.create_publisher(Bool, '/trigger_objA', 10)
        self.objB_trigger_pub = self.create_publisher(Bool, '/trigger_objB', 10)
        self.stop_nav_pub = self.create_publisher(Bool, '/stop_nav', 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10)

        self.locked_id = None
        self.last_seen_time = 0.0
        self.search_direction = 1.0
        self.docked = False
        self.stop_nav_active = False
        self.docked_at = 0.0

        self.create_timer(0.1, self.watchdog_callback)
        self.publish_stop_nav(False)

        self.get_logger().info(
            'ArucoPidDock started. '
            f'primary_marker_id={self.primary_marker_id}, '
            f'secondary_marker_id={self.secondary_marker_id}, '
            f'final_distance={self.final_distance:.2f} m.')

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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        ids_flat = ids.flatten().astype(int)
        target_index = self.select_target_marker(ids_flat, tvecs)
        if target_index is None:
            self.handle_marker_loss(frame)
            return

        marker_id = int(ids_flat[target_index])
        tvec = tvecs[target_index][0]
        rvec = rvecs[target_index][0]

        if self.locked_id != marker_id:
            self.locked_id = marker_id
            self.docked = False
            self.get_logger().info(f'Locked on ArUco marker ID {marker_id}')

        self.last_seen_time = time.time()
        self.search_direction = (
            -1.0 if float(tvec[0]) < 0.0 else 1.0
        ) * self.search_direction_gain
        self.publish_detected(True)
        self.publish_stop_nav(True)
        self.get_logger().info(
            f'ArUco ID {marker_id}: x={float(tvec[0]):.3f}, '
            f'y={float(tvec[1]):.3f}, z={float(tvec[2]):.3f}')

        self.visual_servo_to_marker(tvec, rvec)

        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow('aruco_pid_dock', frame)
            cv2.waitKey(1)

    def marker_heading_error(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        marker_normal = rotation_matrix[:, 2]
        return math.atan2(
            float(marker_normal[0]),
            abs(float(marker_normal[2])) + 1e-6,
        )

    def visual_servo_to_marker(self, tvec, rvec):
        x_error = float(tvec[0])
        distance_error = float(tvec[2] - self.final_distance)
        yaw_error = self.marker_heading_error(rvec)

        aligned = (
            abs(x_error) <= self.align_x_threshold
            and abs(yaw_error) <= self.align_yaw_threshold
            and abs(distance_error) <= self.distance_threshold
        )

        if aligned:
            if not self.docked:
                self.get_logger().info(
                    f'Docked with ArUco ID {self.locked_id}: '
                    f'x={x_error:.3f}, z={tvec[2]:.3f}, yaw_err={yaw_error:.3f}')
                trigger_msg = Bool()
                trigger_msg.data = True
                if self.locked_id == self.primary_marker_id:
                    self.objA_trigger_pub.publish(trigger_msg)
                elif self.locked_id == self.secondary_marker_id:
                    self.objB_trigger_pub.publish(trigger_msg)
                self.docked_at = time.time()
            self.docked = True
            self.stop_robot()
            return

        self.docked = False
        cmd = Twist()
        cmd.linear.x = self.linear_gain * distance_error
        cmd.angular.z = self.angular_direction * (
            self.angular_gain_x * x_error
            + self.angular_gain_yaw * yaw_error
        )

        if abs(x_error) > 0.03 or abs(yaw_error) > 0.12:
            cmd.linear.x = min(cmd.linear.x, 0.04)

        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed),
                           -self.max_linear_speed)
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed),
                            -self.max_angular_speed)
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().debug(
            f'PID dock ID {self.locked_id}: '
            f'x_err={x_error:.3f}, dist_err={distance_error:.3f}, '
            f'yaw_err={yaw_error:.3f}')

    def handle_marker_loss(self, frame):
        self.publish_detected(False)

        if self.last_seen_time <= 0.0:
            self.publish_search_rotation()
        else:
            elapsed = time.time() - self.last_seen_time
            if elapsed > self.stop_nav_loss_timeout:
                self.publish_stop_nav(False)
            if elapsed > self.loss_timeout:
                self.docked = False
                self.stop_robot()
            else:
                self.publish_search_rotation()

        if self.show_debug_window:
            cv2.imshow('aruco_pid_dock', frame)
            cv2.waitKey(1)

    def publish_search_rotation(self):
        cmd = Twist()
        cmd.angular.z = self.search_direction * self.search_angular_speed
        self.cmd_vel_pub.publish(cmd)

    def watchdog_callback(self):
        if self.last_seen_time <= 0.0:
            self.publish_stop_nav(False)
            return

        elapsed = time.time() - self.last_seen_time
        if elapsed > self.stop_nav_loss_timeout:
            self.publish_stop_nav(False)
        else:
            self.publish_stop_nav(True)
        if elapsed > self.loss_timeout:
            self.docked = False
            self.stop_robot()
        if self.docked and self.docked_at > 0.0:
            if time.time() - self.docked_at > self.stop_nav_post_dock_timeout:
                self.publish_stop_nav(False)

    def select_target_marker(self, ids_flat, tvecs):
        if self.locked_id is not None:
            matches = np.where(ids_flat == self.locked_id)[0]
            if len(matches) == 0:
                return None
            return min(matches, key=lambda i: float(tvecs[i][0][2]))

        primary_matches = np.where(ids_flat == self.primary_marker_id)[0]
        secondary_matches = np.where(ids_flat == self.secondary_marker_id)[0]

        if len(primary_matches) > 0:
            return min(primary_matches, key=lambda i: float(tvecs[i][0][2]))
        if len(secondary_matches) > 0:
            return min(secondary_matches, key=lambda i: float(tvecs[i][0][2]))
        return None

    def publish_detected(self, state):
        msg = Bool()
        msg.data = bool(state)
        self.detected_pub.publish(msg)

    def publish_stop_nav(self, state):
        self.stop_nav_active = bool(state)
        msg = Bool()
        msg.data = self.stop_nav_active
        self.stop_nav_pub.publish(msg)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPidDock()
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
