#!/usr/bin/env python3
"""
aruco_docking_node.py
─────────────────────
Python port of dawan0111/Auto-Marker-Docking for a differential-drive robot.

Algorithm (mirrors the C++ repo's Step enum):
  DETECTION  – accumulate stable detections before moving
  WAYPOINT_1 – rotate in place until lateral offset (y) ≈ 0
  WAYPOINT_2 – drive forward until longitudinal offset (x) ≈ waypoint distance
  ARUCO_1    – fine rotate until lateral offset ≈ 0 again
  ARUCO_2    – combined approach + rotate until within marker_gap
  END        – stop and publish trigger

Coordinate convention (matches C++ repo's T_world_image_ rotation):
  After rotating from OpenCV optical frame →
    x  = forward  (was z in optical)
    y  = left     (was -x in optical)
    z  = up       (was -y in optical)
  So  x_cam→forward,  y_cam→left,  z_cam→up.

Topics published:
  /cmd_vel          geometry_msgs/Twist
  /aruco_detected   std_msgs/Bool
  /stop_nav         std_msgs/Bool
  /trigger_objA     std_msgs/Bool   (primary marker)
  /trigger_objB     std_msgs/Bool   (secondary marker)

Topics subscribed:
  /image_raw/compressed   sensor_msgs/CompressedImage
"""

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


# ── FSM step constants ──────────────────────────────────────────────────────
STEP_DETECTION  = 0
STEP_WAYPOINT_1 = 1   # align lateral (y → 0) using waypoint
STEP_WAYPOINT_2 = 2   # approach until x ≈ waypoint_distance
STEP_ARUCO_1    = 3   # fine-align lateral (y → 0) at close range
STEP_ARUCO_2    = 4   # final combined approach
STEP_END        = 5

STEP_NAMES = {
    STEP_DETECTION:  'DETECTION',
    STEP_WAYPOINT_1: 'WAYPOINT_1',
    STEP_WAYPOINT_2: 'WAYPOINT_2',
    STEP_ARUCO_1:    'ARUCO_1',
    STEP_ARUCO_2:    'ARUCO_2',
    STEP_END:        'END',
}


# ── Kalman filter (3-state: x, y, yaw) ─────────────────────────────────────
class ArucoKalmanFilter:
    """
    Constant-position Kalman filter for the marker pose (x, y, yaw).
    Mirrors aruco_kalman_filter.hpp from the C++ repo.
    """

    def __init__(
        self,
        initial_pose: np.ndarray,   # shape (3,)
        P: np.ndarray,              # initial covariance (3×3)
        Q: np.ndarray,              # process noise     (3×3)
        R: np.ndarray,              # measurement noise (3×3)
    ):
        self.x = initial_pose.copy().astype(float)  # state estimate
        self.P = P.copy().astype(float)
        self.Q = Q.copy().astype(float)
        self.R = R.copy().astype(float)
        self.H = np.eye(3)   # observation model
        self.F = np.eye(3)   # state transition (constant-position)

    def predict(self, delta: np.ndarray):
        """Add an external prediction delta (from robot velocity)."""
        self.x = self.F @ self.x + delta
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """Correct the estimate with a new measurement."""
        y  = measurement - self.H @ self.x          # innovation
        S  = self.H @ self.P @ self.H.T + self.R    # innovation covariance
        K  = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()


# ── PD controller (scalar) ──────────────────────────────────────────────────
class PDController:
    """Per-axis PD controller. Mirrors PD_controller.hpp."""

    def __init__(self, kp: float, kd: float, dt: float = 0.05):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self._prev = 0.0

    def update(self, error: float) -> float:
        out = self.kp * error + self.kd * (error - self._prev) / self.dt
        self._prev = error
        return out

    def reset(self):
        self._prev = 0.0


# ── Main node ───────────────────────────────────────────────────────────────
class ArucoDockingNode(Node):

    def __init__(self):
        super().__init__('aruco_docking_node')

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

        # Distance to stop in front of marker (matches repo's marker_gap)
        self.marker_gap = float(
            self.declare_parameter('marker_gap', 0.20).value)
        # Intermediate waypoint distance (stop here before fine-align)
        self.waypoint_distance = float(
            self.declare_parameter('waypoint_distance', 0.40).value)

        # Detection stability gate
        self.detection_gate = int(
            self.declare_parameter('detection_gate', 20).value)

        # Speed limits
        self.max_linear_speed  = float(
            self.declare_parameter('max_linear_speed',  0.20).value)
        self.max_angular_speed = float(
            self.declare_parameter('max_angular_speed', 0.35).value)
        self.search_speed = float(
            self.declare_parameter('search_speed', 0.25).value)

        # PD gains  ← tune these first
        kp_lin = float(self.declare_parameter('kp_linear',  0.25).value)
        kd_lin = float(self.declare_parameter('kd_linear',  0.01).value)
        kp_ang = float(self.declare_parameter('kp_angular', 0.25).value)
        kd_ang = float(self.declare_parameter('kd_angular', 0.01).value)

        # Alignment thresholds
        self.lateral_thr  = float(self.declare_parameter('lateral_threshold',  0.02).value)
        self.forward_thr  = float(self.declare_parameter('forward_threshold',  0.02).value)

        # Marker-loss timeout
        self.loss_timeout = float(self.declare_parameter('loss_timeout', 1.0).value)

        # Camera intrinsics
        cam_default = [1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0]
        dist_default = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_matrix = np.array(
            self.declare_parameter('camera_matrix', cam_default).value,
            dtype=float).reshape((3, 3))
        self.dist_coeffs = np.array(
            self.declare_parameter('dist_coeffs', dist_default).value,
            dtype=float).reshape((-1, 1))

        # ArUco dictionary
        dict_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.show_debug = bool(self.declare_parameter('show_debug_window', True).value)

        # ── PD controllers ────────────────────────────────────────────────
        dt = 0.05  # matches 20 Hz image callback expectation
        self.pd_linear  = PDController(kp_lin, kd_lin, dt)
        self.pd_angular = PDController(kp_ang, kd_ang, dt)

        # ── Kalman filter ─────────────────────────────────────────────────
        #   state = [x_forward, y_lateral, yaw]
        self.kf = ArucoKalmanFilter(
            initial_pose=np.zeros(3),
            P=np.diag([0.01, 0.01, 0.005]),
            Q=np.diag([0.005, 0.005, 0.001]),
            R=np.diag([0.005, 0.005, 0.18 ]),
        )
        self.kf_initialised = False

        # C++ repo T_world_image_ rotation matrix:
        #   x_world = z_cam, y_world = -x_cam, z_world = -y_cam
        self.T_world_image = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])

        # ── Publishers / subscribers ──────────────────────────────────────
        self.cmd_vel_pub      = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.detected_pub     = self.create_publisher(Bool,  '/aruco_detected',  10)
        self.stop_nav_pub     = self.create_publisher(Bool,  '/stop_nav',        10)
        self.objA_trigger_pub = self.create_publisher(Bool,  '/trigger_objA',    10)
        self.objB_trigger_pub = self.create_publisher(Bool,  '/trigger_objB',    10)

        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10)

        # ── State ─────────────────────────────────────────────────────────
        self.step             = STEP_DETECTION
        self.locked_id        = None
        self.detection_count  = 0
        self.triggered        = False
        self.last_seen_time   = 0.0
        self.search_direction = 1.0           # +1 = CCW, -1 = CW
        self.robot_linear     = 0.0           # last sent linear cmd
        self.robot_angular    = 0.0           # last sent angular cmd

        self.create_timer(0.1, self.watchdog_cb)
        self.publish_stop_nav(False)

        self.get_logger().info(
            'ArucoDockingNode ready. '
            f'primary={self.primary_marker_id}, '
            f'secondary={self.secondary_marker_id}, '
            f'marker_gap={self.marker_gap:.2f} m, '
            f'waypoint_distance={self.waypoint_distance:.2f} m')

    # ── Image callback ────────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Image decode error: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            self._on_marker_loss(frame)
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        ids_flat = ids.flatten().astype(int)
        idx = self._select_marker(ids_flat, tvecs)
        if idx is None:
            self._on_marker_loss(frame)
            return

        marker_id = int(ids_flat[idx])
        tvec = tvecs[idx][0]   # (x_cam, y_cam, z_cam)
        rvec = rvecs[idx][0]

        # Re-lock if changed
        if self.locked_id != marker_id:
            self.locked_id       = marker_id
            self.triggered       = False
            self.detection_count = 0
            self.step            = STEP_DETECTION
            self.kf_initialised  = False
            self.pd_linear.reset()
            self.pd_angular.reset()
            self.get_logger().info(f'Locked on ArUco ID {marker_id}')

        self.last_seen_time   = time.time()
        self.search_direction = -1.0 if tvec[0] < 0.0 else 1.0
        self.publish_detected(True)
        self.publish_stop_nav(True)

        # ── Convert to world/robot frame and run Kalman update ────────────
        raw_x, raw_y, raw_yaw = self._optical_to_robot(tvec, rvec)

        # Predict step (uses last sent velocity command)
        predict_delta = self._kalman_predict_delta()
        self.kf.predict(predict_delta)

        # Update step
        self.kf.update(np.array([raw_x, raw_y, raw_yaw]))
        self.kf_initialised = True

        state = self.kf.state   # [x_fwd, y_lat, yaw]
        self._run_fsm(state)

        # ── Debug visualisation ───────────────────────────────────────────
        if self.show_debug:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.putText(frame,
                f'ID={marker_id} step={STEP_NAMES[self.step]}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame,
                f'x={raw_x:.3f} y={raw_y:.3f} yaw={math.degrees(raw_yaw):.1f}deg',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('ArucoDocking', frame)
            cv2.waitKey(1)

    # ── Frame conversion ──────────────────────────────────────────────────

    def _optical_to_robot(self, tvec, rvec):
        """
        Convert OpenCV optical-frame tvec/rvec to robot-frame (x_fwd, y_lat, yaw).
        Matches the C++ repo's T_world_image_ rotation.
        """
        # Translation
        cam = np.array([tvec[0], 0.0, tvec[2]])  # ignore y (height)
        world = self.T_world_image @ cam
        x_fwd = float(world[0])
        y_lat = float(world[1])

        # Yaw from rvec (rotation around Z in robot frame)
        R, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        # Repo uses rvec[1] (pitch in optical = yaw in robot), sign-flipped
        marker_ry = float(rvec[1]) * -1.0
        yaw = self._normalize_angle(marker_ry)
        return x_fwd, y_lat, yaw

    # ── Kalman prediction from odometry ──────────────────────────────────

    def _kalman_predict_delta(self, dt: float = 0.05) -> np.ndarray:
        """
        Approximate how the marker pose changes given the robot's last velocity.
        Mirrors get_predict_vector() in the C++ node.
        """
        if not self.kf_initialised:
            return np.zeros(3)

        s = self.kf.state
        x, y = float(s[0]), float(s[1])
        r     = math.hypot(x, y)
        delta = math.atan2(y, x)

        lin = self.robot_linear
        ang = self.robot_angular

        new_delta = delta + ang * dt
        dx  = r * math.cos(new_delta) - x + lin * dt * math.cos(0)
        dy  = r * math.sin(new_delta) - y + lin * dt * math.sin(0)
        dyaw = ang * dt

        return np.array([dx, dy, dyaw])

    # ── FSM ───────────────────────────────────────────────────────────────

    def _run_fsm(self, state: np.ndarray):
        """
        5-step FSM mirroring the C++ repo's execute() loop.
        state = [x_fwd, y_lat, yaw]
        """
        x   = float(state[0])   # forward distance to marker
        y   = float(state[1])   # lateral offset (+ = left)
        yaw = float(state[2])   # heading error

        self.get_logger().debug(
            f'[{STEP_NAMES[self.step]}] x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°')

        # ── STEP 0: wait for stable detections ────────────────────────────
        if self.step == STEP_DETECTION:
            self.stop_robot()
            self.detection_count += 1
            if self.detection_count >= self.detection_gate:
                self.get_logger().info(
                    f'Detection stable ({self.detection_count} frames) → WAYPOINT_1')
                self.step = STEP_WAYPOINT_1
            return

        # ── STEP 1: rotate until y ≈ 0 (lateral centering at waypoint) ───
        if self.step == STEP_WAYPOINT_1:
            # Drive angular only — zero linear
            ang_cmd = self._clamp(self.pd_angular.update(y), self.max_angular_speed)
            self._publish_cmd(linear=0.0, angular=-ang_cmd)

            if abs(y) <= self.lateral_thr:
                self.get_logger().info('WAYPOINT_1 done → WAYPOINT_2')
                self.pd_angular.reset()
                self.step = STEP_WAYPOINT_2
            return

        # ── STEP 2: drive forward until x ≈ waypoint_distance ────────────
        if self.step == STEP_WAYPOINT_2:
            x_error = x - self.waypoint_distance
            lin_cmd = self._clamp(self.pd_linear.update(x_error), self.max_linear_speed)
            self._publish_cmd(linear=lin_cmd, angular=0.0)

            if abs(x_error) <= self.forward_thr:
                self.get_logger().info('WAYPOINT_2 done → ARUCO_1')
                self.pd_linear.reset()
                self.step = STEP_ARUCO_1
            return

        # ── STEP 3: fine-rotate until y ≈ 0 at close range ───────────────
        if self.step == STEP_ARUCO_1:
            ang_cmd = self._clamp(self.pd_angular.update(y), self.max_angular_speed)
            self._publish_cmd(linear=0.0, angular=-ang_cmd)

            if abs(y) <= self.lateral_thr:
                self.get_logger().info('ARUCO_1 done → ARUCO_2')
                self.pd_angular.reset()
                self.step = STEP_ARUCO_2
            return

        # ── STEP 4: combined final approach ───────────────────────────────
        if self.step == STEP_ARUCO_2:
            x_error = x - self.marker_gap
            lin_cmd = self._clamp(self.pd_linear.update(x_error), self.max_linear_speed)
            ang_cmd = self._clamp(self.pd_angular.update(y),       self.max_angular_speed)

            linear_done  = abs(x_error) <= self.forward_thr
            angular_done = abs(y)       <= self.lateral_thr

            if linear_done:
                lin_cmd = 0.0
            if angular_done:
                ang_cmd = 0.0

            self._publish_cmd(linear=lin_cmd, angular=-ang_cmd)

            if linear_done and angular_done:
                self.get_logger().info('ARUCO_2 done → END')
                self.step = STEP_END
            return

        # ── STEP 5: docked ────────────────────────────────────────────────
        if self.step == STEP_END:
            self.stop_robot()
            if not self.triggered:
                self.get_logger().info(
                    f'Docked with ArUco ID {self.locked_id}. Publishing trigger.')
                msg = Bool()
                msg.data = True
                if self.locked_id == self.primary_marker_id:
                    self.objA_trigger_pub.publish(msg)
                elif self.locked_id == self.secondary_marker_id:
                    self.objB_trigger_pub.publish(msg)
                self.triggered = True

    # ── Marker selection ──────────────────────────────────────────────────

    def _select_marker(self, ids_flat, tvecs):
        """Prefer locked ID; otherwise primary > secondary. Closest wins."""
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

    # ── Marker loss ───────────────────────────────────────────────────────

    def _on_marker_loss(self, frame):
        self.publish_detected(False)

        if self.last_seen_time <= 0.0:
            self._search_rotate()
        else:
            elapsed = time.time() - self.last_seen_time
            if elapsed < self.loss_timeout:
                # Briefly lost — keep searching
                self._search_rotate()
            else:
                # Truly lost — reset FSM
                self.get_logger().warn('Marker lost. Resetting to DETECTION.')
                self.step            = STEP_DETECTION
                self.detection_count = 0
                self.triggered       = False
                self.kf_initialised  = False
                self.locked_id       = None
                self.pd_linear.reset()
                self.pd_angular.reset()
                self.stop_robot()
                self.publish_stop_nav(False)

        if self.show_debug:
            cv2.imshow('ArucoDocking', frame)
            cv2.waitKey(1)

    def _search_rotate(self):
        self._publish_cmd(linear=0.0, angular=self.search_direction * self.search_speed)

    # ── Watchdog ──────────────────────────────────────────────────────────

    def watchdog_cb(self):
        if self.last_seen_time <= 0.0:
            return
        elapsed = time.time() - self.last_seen_time
        if elapsed > self.loss_timeout:
            self.publish_stop_nav(False)

    # ── Utility ───────────────────────────────────────────────────────────

    def _publish_cmd(self, linear: float, angular: float):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)
        self.robot_linear  = linear
        self.robot_angular = angular

    def stop_robot(self):
        self._publish_cmd(0.0, 0.0)

    def publish_detected(self, state: bool):
        msg = Bool()
        msg.data = bool(state)
        self.detected_pub.publish(msg)

    def publish_stop_nav(self, state: bool):
        msg = Bool()
        msg.data = bool(state)
        self.stop_nav_pub.publish(msg)

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


# ── Entry point ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDockingNode()
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
