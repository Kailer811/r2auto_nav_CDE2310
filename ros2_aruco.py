#!/usr/bin/env python3

import math
import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class State(Enum):
    SCANNING = auto()
    PERPENDICULAR = auto()
    STRAFE_X = auto()
    TURN_FACE = auto()
    APPROACH_Z = auto()
    DONE = auto()


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, current):
    delta = target - current
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


class ArucoStateMachineFsm(Node):
    def __init__(self):
        super().__init__('aruco_state_machine_fsm')

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

        self.target_z = float(self.declare_parameter('target_z', 0.01).value)
        self.x_thresh = float(self.declare_parameter('x_thresh', 0.02).value)
        self.z_thresh = float(self.declare_parameter('z_thresh', 0.02).value)
        self.angle_thresh = float(
            self.declare_parameter('angle_thresh', 0.02).value)
        self.linear_speed = float(
            self.declare_parameter('linear_speed', 0.15).value)
        self.angular_speed = float(
            self.declare_parameter('angular_speed', 0.4).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.35).value)
        self.search_direction_gain = float(
            self.declare_parameter('search_direction_gain', 1.0).value)
        self.recovery_delay = float(
            self.declare_parameter('recovery_delay', 0.5).value)
        self.show_debug_window = bool(
            self.declare_parameter('show_debug_window', True).value)

        dictionary_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = np.array([
            [475.0, 0.0, 320.0],
            [0.0, 475.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.state = State.SCANNING
        self.current_yaw = 0.0
        self.target_yaw = None
        self.target_x_dist = None
        self.target_z_dist = None
        self.strafe_turn_sign = 1.0

        self.locked_id = None
        self.last_seen_time = 0.0
        self.has_detected_marker_once = False
        self.search_direction = 1.0
        self.docked = False
        self.docking_enabled = False
        self.last_recovery_log_time = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.detected_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.objA_trigger_pub = self.create_publisher(Bool, '/trigger_objA', 10)
        self.objB_trigger_pub = self.create_publisher(Bool, '/trigger_objB', 10)

        self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            Bool, '/enable_docking', self.enable_docking_callback, 10)

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            'ros2_aruco4 ready. '
            f'primary_marker_id={self.primary_marker_id}, '
            f'secondary_marker_id={self.secondary_marker_id}')

    def enable_docking_callback(self, msg):
        enabled = bool(msg.data)
        if enabled == self.docking_enabled:
            return

        self.docking_enabled = enabled
        if enabled:
            self.reset_tracking_state()
            self.get_logger().info('Docking control enabled.')
            return

        self.get_logger().info('Docking control disabled.')
        self.reset_tracking_state()
        self.cmd_pub.publish(Twist())

    def reset_tracking_state(self):
        self.state = State.SCANNING
        self.target_yaw = None
        self.target_x_dist = None
        self.target_z_dist = None
        self.strafe_turn_sign = 1.0
        self.locked_id = None
        self.has_detected_marker_once = False
        self.docked = False

    def odom_callback(self, msg):
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Image callback error: {exc}')
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
        rvec = rvecs[target_index]
        tvec = tvecs[target_index][0]

        self.last_seen_time = time.time()
        self.has_detected_marker_once = True
        self.search_direction = (
            -1.0 if float(tvec[0]) < 0.0 else 1.0
        ) * self.search_direction_gain
        self.publish_detected(True)

        if self.locked_id != marker_id:
            self.locked_id = marker_id
            self.docked = False
            self.get_logger().info(f'Locked on ArUco marker ID {marker_id}')

        if self.docking_enabled and self.state == State.SCANNING:
            self.plan_from_detection(rvec, tvec)

        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow('ros2_aruco4', frame)
            cv2.waitKey(1)

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

    def plan_from_detection(self, rvec, tvec):
        cam_x = float(tvec[0])
        cam_z = float(tvec[2])

        rotation, _ = cv2.Rodrigues(rvec)
        rotation_inv = rotation.T
        cam_in_marker = -rotation_inv @ tvec
        marker_x = -float(cam_in_marker[0])
        marker_z = float(cam_in_marker[2])

        cam_yaw = math.atan2(cam_x, cam_z)
        marker_yaw = math.atan2(abs(marker_z), marker_x)
        if marker_yaw > math.pi / 2:
            marker_yaw = math.pi / 2 - marker_yaw
        perp_yaw = cam_yaw + marker_yaw

        self.target_yaw = self.current_yaw - perp_yaw
        self.target_x_dist = -marker_x
        self.target_z_dist = marker_z - self.target_z
        self.strafe_turn_sign = -1.0 if perp_yaw > 0.0 else 1.0
        self.state = State.PERPENDICULAR

        self.get_logger().info(
            f'[PLAN] id={self.locked_id} cam_x={cam_x:.3f} cam_z={cam_z:.3f} '
            f'marker_x={marker_x:.3f} marker_z={marker_z:.3f} '
            f'perp_yaw={math.degrees(perp_yaw):.1f}deg')

    def handle_marker_loss(self, frame):
        self.publish_detected(False)

        if not self.docking_enabled:
            pass
        elif not self.has_detected_marker_once:
            self.cmd_pub.publish(Twist())
        else:
            elapsed = time.time() - self.last_seen_time
            if elapsed < self.recovery_delay:
                self.cmd_pub.publish(Twist())
            else:
                if time.time() - self.last_recovery_log_time > 1.0:
                    self.get_logger().warn(
                        f'ArUco lost for {elapsed:.2f}s. Starting recovery search.')
                    self.last_recovery_log_time = time.time()
                self.state = State.SCANNING
                self.docked = False
                cmd = Twist()
                cmd.angular.z = self.search_direction * self.search_angular_speed
                self.publish_cmd(cmd)

        if self.show_debug_window:
            cv2.imshow('ros2_aruco4', frame)
            cv2.waitKey(1)

    def control_loop(self):
        cmd = Twist()

        if not self.docking_enabled:
            return

        if self.state == State.SCANNING:
            pass
        elif self.state == State.PERPENDICULAR:
            err = angle_diff(self.target_yaw, self.current_yaw)
            if abs(err) < self.angle_thresh:
                self.get_logger().info('[STATE] Perpendicular achieved -> STRAFE_X')
                self.state = State.STRAFE_X
            else:
                cmd.angular.z = self.angular_speed * np.sign(err)
        elif self.state == State.STRAFE_X:
            if abs(self.target_x_dist) < self.x_thresh:
                self.get_logger().info('[STATE] X aligned -> TURN_FACE')
                self.state = State.TURN_FACE
                self.target_yaw = self.current_yaw - math.copysign(
                    math.pi / 2, self.target_x_dist * self.strafe_turn_sign)
            else:
                direction = math.copysign(1.0, self.target_x_dist)
                cmd.linear.x = self.linear_speed * direction
                self.target_x_dist -= self.linear_speed * direction * 0.05
        elif self.state == State.TURN_FACE:
            err = angle_diff(self.target_yaw, self.current_yaw)
            if abs(err) < self.angle_thresh:
                self.get_logger().info('[STATE] Facing marker -> APPROACH_Z')
                self.state = State.APPROACH_Z
            else:
                cmd.angular.z = self.angular_speed * np.sign(err)
        elif self.state == State.APPROACH_Z:
            if self.target_z_dist <= self.z_thresh:
                self.get_logger().info('[STATE] Reached target Z -> DONE')
                self.state = State.DONE
            else:
                cmd.linear.x = self.linear_speed
                self.target_z_dist -= self.linear_speed * 0.05
        elif self.state == State.DONE:
            if not self.docked:
                self.get_logger().info(f'[DONE] Docked with ArUco ID {self.locked_id}')
                trigger_msg = Bool()
                trigger_msg.data = True
                if self.locked_id == self.primary_marker_id:
                    self.objA_trigger_pub.publish(trigger_msg)
                elif self.locked_id == self.secondary_marker_id:
                    self.objB_trigger_pub.publish(trigger_msg)
                self.docked = True

        self.publish_cmd(cmd)

    def publish_detected(self, state):
        msg = Bool()
        msg.data = bool(state)
        self.detected_pub.publish(msg)

    def publish_cmd(self, cmd):
        if not self.docking_enabled:
            return
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoStateMachineFsm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
