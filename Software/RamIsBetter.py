#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import cv2
import numpy as np
import tf2_ros
import math
from tf2_ros import TransformException
from std_msgs.msg import Bool
import time


DEFAULT_CAMERA_FRAMES = [
    'camera_color_optical_frame',
    'camera_optical_frame',
    'camera_depth_optical_frame',
    'camera_link',
    'base_link',
]


class RamIsBetter(Node):
    def __init__(self):
        super().__init__('ram_is_better')

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera intrinsics — override via ROS params or load from calibration file
        self.camera_matrix = np.array(
            [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.05  # metres

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Parameters ────────────────────────────────────────────────────────
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.camera_frame = self.declare_parameter('camera_frame', '').value
        self.target_marker_id = int(
            self.declare_parameter('target_marker_id', 2).value)

        # How far IN FRONT of the marker face the robot should stop (metres).
        # Positive = in front of the marker (the side the marker is facing).
        self.standoff_distance = float(
            self.declare_parameter('standoff_distance', 0.15).value)
        self.min_nav2_standoff = float(
            self.declare_parameter('min_nav2_standoff', 0.15).value)
        self.external_detection_translation_x = float(
            self.declare_parameter('external_detection_translation_x', 0.0).value)
        self.external_detection_translation_y = float(
            self.declare_parameter('external_detection_translation_y', 0.0).value)
        self.external_detection_rotation_yaw = float(
            self.declare_parameter('external_detection_rotation_yaw', 0.0).value)
        self.staging_x_offset = float(
            self.declare_parameter('staging_x_offset', -self.standoff_distance).value)
        self.staging_y_offset = float(
            self.declare_parameter('staging_y_offset', 0.0).value)
        self.staging_yaw_offset = float(
            self.declare_parameter('staging_yaw_offset', 0.0).value)

        # Switch from Nav2 → visual servoing at this camera-frame Z distance.
        self.visual_takeover_distance = float(
            self.declare_parameter('visual_takeover_distance', 0.40).value)

        # Final approach distance for visual servoing (stop when this close).
        self.final_distance = float(
            self.declare_parameter('final_distance', 0.1).value)

        self.linear_gain = float(self.declare_parameter('linear_gain', 0.6).value)
        self.linear_integral_gain = float(
            self.declare_parameter('linear_integral_gain', 0.03).value)
        self.angular_gain_x = float(self.declare_parameter('angular_gain_x', 2.5).value)
        self.angular_gain_yaw = float(self.declare_parameter('angular_gain_yaw', 1.5).value)
        self.angular_d_gain_x = float(
            self.declare_parameter('angular_d_gain_x', 0.35).value)
        self.angular_d_gain_yaw = float(
            self.declare_parameter('angular_d_gain_yaw', 0.2).value)
        self.max_linear_speed = float(self.declare_parameter('max_linear_speed', 0.12).value)
        self.max_angular_speed = float(self.declare_parameter('max_angular_speed', 0.9).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.35).value)
        self.search_timeout = float(
            self.declare_parameter('search_timeout', 5.0).value)
        self.visual_loss_wait = float(
            self.declare_parameter('visual_loss_wait', 0.5).value)
        self.post_nav_goal_recovery_delay = float(
            self.declare_parameter('post_nav_goal_recovery_delay', 2.0).value)
        self.search_trigger_distance = float(
            self.declare_parameter('search_trigger_distance', 1.2).value)
        self.filter_alpha = float(
            self.declare_parameter('filter_alpha', 0.35).value)
        self.integral_limit = float(
            self.declare_parameter('integral_limit', 0.4).value)

        # Only resend a Nav2 goal when the desired position drifts by this much.
        self.goal_update_distance = float(
            self.declare_parameter('goal_update_distance', 0.08).value)
        self.goal_update_yaw = float(
            self.declare_parameter('goal_update_yaw', 0.2).value)

        self.show_debug_window = bool(
            self.declare_parameter('show_debug_window', True).value)

        # Visual-servo alignment thresholds (all must be satisfied to trigger).
        self.align_x_threshold = float(
            self.declare_parameter('align_x_threshold', 0.05).value)
        self.align_yaw_threshold = float(
            self.declare_parameter('align_yaw_threshold', 0.08).value)
        self.distance_threshold = float(
            self.declare_parameter('distance_threshold', 0.16).value)

        # Recovery / lost-marker settings
        self.missing_marker_threshold = int(
            self.declare_parameter('missing_marker_threshold', 10).value)
        self.recovery_timeout = float(
            self.declare_parameter('recovery_timeout', 3.0).value)

        # ── Internal state ────────────────────────────────────────────────────
        self.camera_frame_candidates = list(DEFAULT_CAMERA_FRAMES)
        if self.camera_frame:
            self.camera_frame_candidates.insert(0, self.camera_frame)
        self.active_source_frame = None
        self.last_tf_warn_time = 0.0

        self.locked_id = None
        self.goal_active = False
        self.goal_handle = None
        self.mode = 'search'
        self.last_goal_pose = None  # (x, y, yaw)
        self.search_start_time = 0.0
        self.search_direction = 1.0
        self.last_marker_seen_time = 0.0
        self.last_seen_x_error = 0.0
        self.visual_loss_start_time = 0.0
        self.last_nav_goal_end_time = 0.0

        self.missing_marker_frames = 0
        self.recovery_mode = False
        # Stores the last marker map pose data for recovery navigation.
        self.last_marker_map_data = None
        self.recovery_start_time = 0.0
        self.filtered_x_error = None
        self.filtered_distance_error = None
        self.filtered_yaw_error = None
        self.prev_filtered_x_error = None
        self.prev_filtered_yaw_error = None
        self.last_servo_time = None
        self.distance_integral = 0.0
        self.docked = False

        # ── ROS interfaces ────────────────────────────────────────────────────
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.objA_trigger_pub = self.create_publisher(Bool, '/trigger_objA', 10)
        self.aruco_detected_pub = self.create_publisher(Bool, '/aruco_detected', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10)

        self.get_logger().info(
            'RamIsBetter node initialised. '
            f'target_marker_id={self.target_marker_id}, '
            f'standoff={self.standoff_distance:.2f} m, '
            f'min_nav2_standoff={self.min_nav2_standoff:.2f} m, '
            f'visual_takeover_distance={self.visual_takeover_distance:.2f} m.')

    # ─────────────────────────────────────────────────────────────────────────
    # Image callback
    # ─────────────────────────────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

                target_candidates = [
                    i for i in range(len(ids))
                    if int(ids[i][0]) == self.target_marker_id
                ]
                if not target_candidates:
                    self._publish_detected(False)
                    return

                self._publish_detected(True)

                # Pick closest matching marker
                target_index = min(
                    target_candidates,
                    key=lambda i: float(tvecs[i][0][2]),
                )
                marker_id = int(ids[target_index][0])
                tvec = tvecs[target_index][0]
                rvec = rvecs[target_index][0]
                self.last_marker_seen_time = time.time()
                self.last_seen_x_error = float(tvec[0])
                self.visual_loss_start_time = 0.0
                self.last_nav_goal_end_time = 0.0

                self.missing_marker_frames = 0

                if self.recovery_mode:
                    self.get_logger().info('Marker reacquired — resuming normal tracking.')
                    self.recovery_mode = False
                    self.cancel_navigation_goal()
                if self.mode == 'search_reacquire':
                    self.get_logger().info('Marker reacquired during search.')
                    self.stop_robot()

                if self.locked_id is None:
                    self.locked_id = marker_id
                    self.docked = False
                    self.get_logger().info(f'Locked on to ArUco marker ID: {marker_id}')

                if marker_id != self.locked_id:
                    # Ignore non-target markers
                    return

                distance_to_marker = float(tvec[2])

                if distance_to_marker > self.visual_takeover_distance:
                    # ── Nav2 approach ─────────────────────────────────────────
                    marker_map_xy = self.transform_marker_to_map(tvec, rvec)
                    if marker_map_xy is not None:
                        self.last_marker_map_data = marker_map_xy
                        target_pose = self.calculate_target_pose(marker_map_xy)
                        self.mode = 'nav2'
                        self.send_navigation_goal(target_pose)
                        self.get_logger().info(
                            f'Nav2 → ID {marker_id} '
                            f'target=({target_pose.pose.position.x:.2f}, '
                            f'{target_pose.pose.position.y:.2f})')
                    else:
                        self.get_logger().warn(
                            'Cannot transform marker to map — TF not ready yet.')
                else:
                    # ── Visual servoing ───────────────────────────────────────
                    if self.mode != 'visual':
                        self.get_logger().info(
                            f'Switching to visual servo for ID {marker_id}.')
                        self.mode = 'visual'
                        self.cancel_navigation_goal()
                        self.reset_servo_state()
                    self.visual_servo_to_marker(tvec, rvec)

                if self.show_debug_window:
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(
                        frame, self.camera_matrix, self.dist_coeffs,
                        rvecs[target_index], tvecs[target_index], 0.05)

            else:
                # ── No marker visible ─────────────────────────────────────────
                self.missing_marker_frames += 1

                if self.missing_marker_frames >= self.missing_marker_threshold:
                    if self.locked_id is not None:
                        if self.mode == 'nav2' and self.goal_active:
                            self.get_logger().info(
                                'Marker temporarily lost during Nav2 approach — '
                                'continuing to current Nav2 goal.')
                            return
                        if self.should_start_search():
                            self.search_for_marker()
                            return
                        self.stop_robot()

            if self.show_debug_window:
                cv2.imshow('RamIsBetter — ArUco Detection', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate transforms
    # ─────────────────────────────────────────────────────────────────────────

    def transform_marker_to_map(self, tvec_camera, rvec_camera):
        """
        Return the marker position and facing normal in the map frame.

        We generate both possible standoff points in calculate_target_pose()
        and choose the one on the robot's side of the wall.
        """
        for source_frame in self.camera_frame_candidates:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, source_frame, rclpy.time.Time())

                translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ], dtype=float)

                R = self.quaternion_to_rotation_matrix([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ])

                marker_cam = self.marker_vector_in_source_frame(
                    tvec_camera, source_frame)
                marker_map = R @ marker_cam + translation

                marker_rot_optical, _ = cv2.Rodrigues(
                    np.array(rvec_camera, dtype=float))
                marker_rot_source = (
                    self.optical_to_source_rotation(source_frame)
                    @ marker_rot_optical
                )
                marker_normal_source = marker_rot_source[:, 2]
                marker_normal_map = R @ marker_normal_source

                if self.active_source_frame != source_frame:
                    self.active_source_frame = source_frame
                    self.get_logger().info(
                        f'Using TF source frame "{source_frame}" for ArUco map transform.')

                normal_xy = marker_normal_map[:2]
                normal_norm = np.linalg.norm(normal_xy)
                if normal_norm < 1e-6:
                    normal_xy = None
                else:
                    normal_xy = normal_xy / normal_norm

                return marker_map[:2], normal_xy

            except TransformException:
                continue

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_tf_warn_time > 2.0:
            self.last_tf_warn_time = now_sec
            self.get_logger().warn(
                f'No TF from any of {self.camera_frame_candidates} → "{self.map_frame}".')
        return None

    def marker_vector_in_source_frame(self, tvec_camera, source_frame):
        x_cam, y_cam, z_cam = tvec_camera
        if 'optical' in source_frame:
            return np.array([x_cam, y_cam, z_cam], dtype=float)
        # optical (x right, y down, z fwd) → robot (x fwd, y left, z up)
        return np.array([z_cam, -x_cam, -y_cam], dtype=float)

    def optical_to_source_rotation(self, source_frame):
        if 'optical' in source_frame:
            return np.eye(3, dtype=float)
        return np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

    def _get_robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, 'base_link', rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y])
        except TransformException:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Goal calculation
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_target_pose(self, marker_map_data):
        """
        Put the Nav2 goal directly on the robot-to-marker line.

        If the marker is straight ahead, the goal should also be straight ahead.
        This avoids side-offset staging behavior that makes Nav2 arc around the
        marker before visual servoing takes over.
        """
        robot_xy = self._get_robot_xy()
        marker_xy, _marker_normal_xy = marker_map_data
        marker_xy = np.array(marker_xy, dtype=float)
        nav2_standoff = max(self.standoff_distance, self.min_nav2_standoff)

        if robot_xy is None:
            self.get_logger().warn(
                'Cannot get robot position — using marker pose directly for Nav2 goal.')
            target_xy = marker_xy + np.array([-nav2_standoff, 0.0], dtype=float)
        else:
            approach_vector = marker_xy - robot_xy
            approach_distance = np.linalg.norm(approach_vector)

            if approach_distance < 1e-6:
                target_xy = marker_xy.copy()
            else:
                approach_unit = approach_vector / approach_distance
                target_xy = marker_xy - approach_unit * nav2_standoff

        face_vector = marker_xy - target_xy
        face_norm = np.linalg.norm(face_vector)
        if face_norm < 1e-6:
            yaw = 0.0
        else:
            yaw = math.atan2(float(face_vector[1]), float(face_vector[0]))

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(target_xy[0])
        pose.pose.position.y = float(target_xy[1])
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    # ─────────────────────────────────────────────────────────────────────────
    # Visual servoing
    # ─────────────────────────────────────────────────────────────────────────

    def marker_heading_error(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        marker_normal = rotation_matrix[:, 2]
        return math.atan2(float(marker_normal[0]),
                          abs(float(marker_normal[2])) + 1e-6)

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
                    f'Aligned with ArUco ID {self.locked_id}: '
                    f'x={x_error:.3f}, z={tvec[2]:.3f}, yaw_err={yaw_error:.3f}')
                trigger_msg = Bool()
                trigger_msg.data = True
                self.objA_trigger_pub.publish(trigger_msg)
            self.docked = True
            self.stop_robot()
            return

        self.docked = False
        cmd = Twist()
        cmd.linear.x = self.linear_gain * distance_error
        cmd.angular.z = (
            self.angular_gain_x * x_error
            + self.angular_gain_yaw * yaw_error
        )

        # Slow linear speed while still aligning laterally / rotationally
        if abs(x_error) > 0.03 or abs(yaw_error) > 0.12:
            cmd.linear.x = min(cmd.linear.x, 0.04)

        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed),
                           -self.max_linear_speed)
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed),
                            -self.max_angular_speed)
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().debug(
            f'Visual servo ID {self.locked_id}: '
            f'x_err={x_error:.3f}, dist_err={distance_error:.3f}, '
            f'yaw_err={yaw_error:.3f}')

    # ─────────────────────────────────────────────────────────────────────────
    # Nav2 helpers
    # ─────────────────────────────────────────────────────────────────────────

    def send_navigation_goal(self, pose):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available.')
            return

        if self.goal_active and self.last_goal_pose is not None:
            lx, ly, lyaw = self.last_goal_pose
            nx = pose.pose.position.x
            ny = pose.pose.position.y
            nyaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
            if (math.hypot(nx - lx, ny - ly) < self.goal_update_distance
                    and abs(self.normalize_angle(nyaw - lyaw)) < self.goal_update_yaw):
                return  # Goal hasn't moved enough — skip resend
            self.cancel_navigation_goal()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f'Sending Nav2 goal → ({pose.pose.position.x:.2f}, '
            f'{pose.pose.position.y:.2f})')

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
        self.goal_active = True
        self.last_goal_pose = (
            pose.pose.position.x,
            pose.pose.position.y,
            2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w),
        )

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Navigation goal rejected.')
            self.goal_active = False
            self.goal_handle = None
            return
        self.goal_handle = goal_handle
        self.get_logger().info('Navigation goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        status = result.status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation goal reached.')
        else:
            self.get_logger().warn(f'Navigation goal ended with status={status}.')
        self.goal_active = False
        self.goal_handle = None
        self.last_nav_goal_end_time = time.time()

        if self.mode == 'nav2' and self.locked_id is not None and not self.marker_recently_seen():
            self.mode = 'visual'
            self.search_start_time = 0.0
            self.visual_loss_start_time = 0.0
            self.docked = False
            self.stop_robot()
            self.get_logger().info(
                'Nav2 goal ended without a visible ArUco marker. '
                f'Waiting {self.post_nav_goal_recovery_delay:.1f} s before recovery.')

    def cancel_navigation_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.goal_active = False

    def should_start_search(self):
        if self.last_marker_map_data is None:
            return False
        if self.mode == 'search_reacquire':
            return True
        if self.mode == 'nav2':
            return False
        if self.mode != 'visual':
            return False
        now = time.time()
        if self.last_nav_goal_end_time > 0.0:
            return (now - self.last_nav_goal_end_time) >= self.post_nav_goal_recovery_delay
        if self.visual_loss_start_time == 0.0:
            self.visual_loss_start_time = now
            self.get_logger().info(
                f'ArUco lost during visual align. Waiting up to '
                f'{self.visual_loss_wait:.1f} s before recovery.')
            return False
        return (now - self.visual_loss_start_time) >= self.visual_loss_wait

    def search_for_marker(self):
        now = time.time()
        if self.mode != 'search_reacquire':
            self.mode = 'search_reacquire'
            self.search_start_time = now
            self.search_direction = -1.0 if self.last_seen_x_error < 0.0 else 1.0
            self.cancel_navigation_goal()
            self.reset_servo_state()
            self.docked = False
            self.last_nav_goal_end_time = 0.0
            self.get_logger().info(
                f'Searching for ArUco ID {self.locked_id} near last known pose.')

        if now - self.search_start_time > self.search_timeout:
            self.mode = 'nav2'
            self.recovery_mode = True
            self.recovery_start_time = now
            self.stop_robot()
            if self.last_marker_map_data is not None:
                recovery_pose = self.calculate_target_pose(self.last_marker_map_data)
                self.send_navigation_goal(recovery_pose)
            return

        cmd = Twist()
        cmd.angular.z = self.search_direction * self.search_angular_speed
        self.cmd_vel_pub.publish(cmd)

    def reset_servo_state(self):
        self.filtered_x_error = None
        self.filtered_distance_error = None
        self.filtered_yaw_error = None
        self.prev_filtered_x_error = None
        self.prev_filtered_yaw_error = None
        self.last_servo_time = None
        self.distance_integral = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def quaternion_to_rotation_matrix(self, quaternion):
        x, y, z, w = quaternion
        return np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y],
        ])

    def offset_pose_xy(self, origin_xy, yaw, offset_x, offset_y):
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([
            float(origin_xy[0]) + c * offset_x - s * offset_y,
            float(origin_xy[1]) + s * offset_x + c * offset_y,
        ], dtype=float)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def marker_recently_seen(self, timeout_sec=0.5):
        if self.last_marker_seen_time <= 0.0:
            return False
        return (time.time() - self.last_marker_seen_time) <= timeout_sec

    def _publish_detected(self, state: bool):
        msg = Bool()
        msg.data = state
        self.aruco_detected_pub.publish(msg)

    def _reset_tracking(self):
        self.locked_id = None
        self.active_source_frame = None
        self.mode = 'search'
        self.recovery_mode = False
        self.docked = False
        self.missing_marker_frames = 0
        self.last_marker_map_data = None
        self.search_start_time = 0.0
        self.visual_loss_start_time = 0.0
        self.last_nav_goal_end_time = 0.0
        self.cancel_navigation_goal()
        self.reset_servo_state()
        self.stop_robot()
        self._publish_detected(False)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RamIsBetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        if node.show_debug_window:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
