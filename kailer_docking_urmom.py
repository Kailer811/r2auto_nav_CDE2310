#!/usr/bin/env python3

import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


# ── Docking FSM states (mirrors the C++ repo's step enum) ──────────────────
STATE_DETECTION   = 0   # accumulate stable detections before moving
STATE_ALIGN_YAW   = 1   # rotate in place until laterally centred (x≈0)
STATE_APPROACH    = 2   # drive straight in until close enough
STATE_FINE_ALIGN  = 3   # final combined correction
STATE_DONE        = 4


class PdController:
    """Simple per-axis PD controller (mirrors PDController in the C++ repo)."""

    def __init__(self, kp: float, kd: float, dt: float = 0.05):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.kp * error + self.kd * derivative

    def reset(self):
        self._prev_error = 0.0


class ArucoPidDock(Node):
    def __init__(self):
        super().__init__('aruco_pid_dock')

        self.bridge = CvBridge()

        # ── ROS parameters ────────────────────────────────────────────────
        self.image_topic = self.declare_parameter(
            'image_topic', '/image_raw/compressed').value
        self.cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', 'cmd_vel').value
        self.primary_marker_id = int(
            self.declare_parameter('primary_marker_id', 2).value)
        self.secondary_marker_id = int(
            self.declare_parameter('secondary_marker_id', 4).value)
        self.marker_size = float(
            self.declare_parameter('marker_size', 0.05).value)
        self.final_distance = float(
            self.declare_parameter('final_distance', 0.20).value)

        # PD gains  (tune these — start conservative)
        self.kp_linear  = float(self.declare_parameter('kp_linear',  0.6).value)
        self.kd_linear  = float(self.declare_parameter('kd_linear',  0.05).value)
        self.kp_angular = float(self.declare_parameter('kp_angular', 2.0).value)
        self.kd_angular = float(self.declare_parameter('kd_angular', 0.1).value)
        self.kp_yaw     = float(self.declare_parameter('kp_yaw',     1.0).value)
        self.kd_yaw     = float(self.declare_parameter('kd_yaw',     0.05).value)

        self.max_linear_speed  = float(self.declare_parameter('max_linear_speed',  0.20).value)
        self.max_angular_speed = float(self.declare_parameter('max_angular_speed', 0.9).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.35).value)

        # Thresholds for state transitions
        self.align_x_threshold    = float(self.declare_parameter('align_x_threshold',    0.03).value)
        self.align_yaw_threshold  = float(self.declare_parameter('align_yaw_threshold',  0.05).value)
        self.distance_threshold   = float(self.declare_parameter('distance_threshold',   0.03).value)
        self.detection_gate       = int(  self.declare_parameter('detection_gate',       10  ).value)

        self.loss_timeout         = float(self.declare_parameter('loss_timeout',         0.75).value)
        self.stop_nav_loss_timeout= float(self.declare_parameter('stop_nav_loss_timeout',5.0 ).value)
        self.show_debug_window    = bool( self.declare_parameter('show_debug_window',    True).value)

        camera_matrix_default = [1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0]
        dist_coeffs_default   = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_matrix = np.array(
            self.declare_parameter('camera_matrix', camera_matrix_default).value,
            dtype=float).reshape((3, 3))
        self.dist_coeffs = np.array(
            self.declare_parameter('dist_coeffs', dist_coeffs_default).value,
            dtype=float).reshape((-1, 1))

        dictionary_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ── PD controllers ────────────────────────────────────────────────
        self.pd_linear  = PdController(self.kp_linear,  self.kd_linear)
        self.pd_angular = PdController(self.kp_angular, self.kd_angular)
        self.pd_yaw     = PdController(self.kp_yaw,     self.kd_yaw)

        # ── Publishers / subscribers ──────────────────────────────────────
        self.cmd_vel_pub       = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.detected_pub      = self.create_publisher(Bool,  '/aruco_detected',  10)
        self.objA_trigger_pub  = self.create_publisher(Bool,  '/trigger_objA',    10)
        self.objB_trigger_pub  = self.create_publisher(Bool,  '/trigger_objB',    10)
        self.stop_nav_pub      = self.create_publisher(Bool,  '/stop_nav',        10)
        self.image_sub         = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10)

        # ── State ─────────────────────────────────────────────────────────
        self.fsm_state       = STATE_DETECTION
        self.locked_id       = None
        self.detection_count = 0
        self.last_seen_time  = 0.0
        self.search_direction = 1.0
        self.docked          = False
        self.docked_at       = 0.0

        self.create_timer(0.1, self.watchdog_callback)
        self.publish_stop_nav(False)

        self.get_logger().info(
            f'ArucoPidDock ready. '
            f'primary={self.primary_marker_id}, '
            f'secondary={self.secondary_marker_id}, '
            f'final_distance={self.final_distance:.2f} m')

    # ── Image callback ────────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Image decode failed: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            self.handle_marker_loss(frame)
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        ids_flat     = ids.flatten().astype(int)
        target_index = self.select_target_marker(ids_flat, tvecs)
        if target_index is None:
            self.handle_marker_loss(frame)
            return

        marker_id = int(ids_flat[target_index])
        tvec      = tvecs[target_index][0]
        rvec      = rvecs[target_index][0]

        # Lock on
        if self.locked_id != marker_id:
            self.locked_id       = marker_id
            self.docked          = False
            self.detection_count = 0
            self.fsm_state       = STATE_DETECTION
            self._reset_pd()
            self.get_logger().info(f'Locked on ArUco ID {marker_id}')

        self.last_seen_time   = time.time()
        self.search_direction = -1.0 if float(tvec[0]) < 0.0 else 1.0
        self.publish_detected(True)
        self.publish_stop_nav(True)

        self.run_fsm(tvec, rvec)

        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow('aruco_pid_dock', frame)
            cv2.waitKey(1)

    # ── FSM ───────────────────────────────────────────────────────────────

    def run_fsm(self, tvec, rvec):
        x_error        = float(tvec[0])           # lateral offset (+ = right)
        distance_error = float(tvec[2]) - self.final_distance  # depth error
        yaw_error      = self.marker_heading_error(rvec)

        self.get_logger().info(
            f'[ID {self.locked_id}] state={self.fsm_state} '
            f'x={x_error:.3f} dist_err={distance_error:.3f} yaw={yaw_error:.3f}')

        # ── STATE 0: wait for N stable detections before moving ───────────
        if self.fsm_state == STATE_DETECTION:
            self.stop_robot()
            self.detection_count += 1
            if self.detection_count >= self.detection_gate:
                self.get_logger().info(
                    f'Detection gate passed ({self.detection_count} frames). '
                    'Moving to ALIGN_YAW.')
                self.fsm_state = STATE_ALIGN_YAW
            return

        # ── STATE 1: rotate until laterally centred ───────────────────────
        if self.fsm_state == STATE_ALIGN_YAW:
            cmd = Twist()
            cmd.angular.z = -self._clamp(
                self.pd_angular.update(x_error), self.max_angular_speed)
            self.cmd_vel_pub.publish(cmd)

            if abs(x_error) <= self.align_x_threshold:
                self.get_logger().info('Lateral aligned. Moving to APPROACH.')
                self.pd_angular.reset()
                self.fsm_state = STATE_APPROACH
            return

        # ── STATE 2: drive forward until within fine-align range ──────────
        if self.fsm_state == STATE_APPROACH:
            cmd = Twist()
            cmd.linear.x = self._clamp(
                self.pd_linear.update(distance_error), self.max_linear_speed)
            # small heading correction while approaching
            cmd.angular.z = -self._clamp(
                self.pd_yaw.update(yaw_error) * 0.5, self.max_angular_speed * 0.5)
            self.cmd_vel_pub.publish(cmd)

            # Switch to fine-align when close enough and drifted laterally
            if abs(distance_error) <= 0.15:
                self.get_logger().info('Close enough. Moving to FINE_ALIGN.')
                self.pd_linear.reset()
                self.pd_yaw.reset()
                self.fsm_state = STATE_FINE_ALIGN
            return

        # ── STATE 3: combined correction at close range ───────────────────
        if self.fsm_state == STATE_FINE_ALIGN:
            fully_aligned = (
                abs(x_error)        <= self.align_x_threshold
                and abs(yaw_error)  <= self.align_yaw_threshold
                and abs(distance_error) <= self.distance_threshold
            )

            if fully_aligned:
                self.fsm_state = STATE_DONE
                self.get_logger().info('Fine-aligned. Moving to DONE.')
                return

            cmd = Twist()
            cmd.linear.x  = self._clamp(
                self.pd_linear.update(distance_error), self.max_linear_speed)
            cmd.angular.z = -self._clamp(
                self.pd_angular.update(x_error) + self.pd_yaw.update(yaw_error),
                self.max_angular_speed)

            # Slow linear if still laterally off
            if abs(x_error) > 0.03 or abs(yaw_error) > 0.12:
                cmd.linear.x = min(cmd.linear.x, 0.04)

            self.cmd_vel_pub.publish(cmd)
            return

        # ── STATE 4: docked ───────────────────────────────────────────────
        if self.fsm_state == STATE_DONE:
            self.stop_robot()
            if not self.docked:
                self.get_logger().info(
                    f'Docked with ArUco ID {self.locked_id}')
                trigger_msg = Bool()
                trigger_msg.data = True
                if self.locked_id == self.primary_marker_id:
                    self.objA_trigger_pub.publish(trigger_msg)
                elif self.locked_id == self.secondary_marker_id:
                    self.objB_trigger_pub.publish(trigger_msg)
                self.docked    = True
                self.docked_at = time.time()

    # ── Helpers ───────────────────────────────────────────────────────────

    def marker_heading_error(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        marker_normal = rotation_matrix[:, 2]
        return math.atan2(
            float(marker_normal[0]),
            abs(float(marker_normal[2])) + 1e-6)

    def _clamp(self, value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    def _reset_pd(self):
        self.pd_linear.reset()
        self.pd_angular.reset()
        self.pd_yaw.reset()

    def select_target_marker(self, ids_flat, tvecs):
        if self.locked_id is not None:
            matches = np.where(ids_flat == self.locked_id)[0]
            if len(matches) == 0:
                return None
            return min(matches, key=lambda i: float(tvecs[i][0][2]))

        for mid in (self.primary_marker_id, self.secondary_marker_id):
            matches = np.where(ids_flat == mid)[0]
            if len(matches) > 0:
                return min(matches, key=lambda i: float(tvecs[i][0][2]))
        return None

    def handle_marker_loss(self, frame):
        self.publish_detected(False)
        if self.last_seen_time <= 0.0:
            self.publish_search_rotation()
        else:
            elapsed = time.time() - self.last_seen_time
            if elapsed > self.stop_nav_loss_timeout:
                self.publish_stop_nav(False)
            if elapsed > self.loss_timeout:
                self.docked    = False
                self.fsm_state = STATE_DETECTION
                self._reset_pd()
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
            self.docked    = False
            self.fsm_state = STATE_DETECTION
            self._reset_pd()
            self.stop_robot()

    def publish_detected(self, state: bool):
        msg = Bool()
        msg.data = state
        self.detected_pub.publish(msg)

    def publish_stop_nav(self, state: bool):
        msg = Bool()
        msg.data = bool(state)
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