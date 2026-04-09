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
        
        # Initialize CV bridge for image processing
        self.bridge = CvBridge()
        
        # ArUco detection setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera parameters (adjust these based on your camera calibration)
        self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.05  # Size of the ArUco marker in meters
        
        # TF buffer and listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.camera_frame = self.declare_parameter('camera_frame', '').value
        self.camera_frame_candidates = list(DEFAULT_CAMERA_FRAMES)
        if self.camera_frame:
            self.camera_frame_candidates.insert(0, self.camera_frame)
        self.active_source_frame = None
        self.last_tf_warn_time = 0.0
        self.standoff_distance = float(self.declare_parameter('standoff_distance', 0.35).value)
        self.visual_takeover_distance = float(
            self.declare_parameter('visual_takeover_distance', 0.45).value)
        self.final_distance = float(self.declare_parameter('final_distance', 0.05).value)
        self.linear_gain = float(self.declare_parameter('linear_gain', 0.6).value)
        self.angular_gain_x = float(self.declare_parameter('angular_gain_x', 2.5).value)
        self.angular_gain_yaw = float(self.declare_parameter('angular_gain_yaw', 1.5).value)
        self.max_linear_speed = float(self.declare_parameter('max_linear_speed', 0.12).value)
        self.max_angular_speed = float(self.declare_parameter('max_angular_speed', 0.9).value)
        self.goal_update_distance = float(self.declare_parameter('goal_update_distance', 0.08).value)
        self.goal_update_yaw = float(self.declare_parameter('goal_update_yaw', 0.2).value)
        self.show_debug_window = bool(self.declare_parameter('show_debug_window', False).value)
        self.align_x_threshold = float(self.declare_parameter('align_x_threshold', 0.5).value)
        self.align_yaw_threshold = float(self.declare_parameter('align_yaw_threshold', 0.5).value)
        self.distance_threshold = float(self.declare_parameter('distance_threshold', 0.5).value)

        # Navigation client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.objA_trigger_pub = self.create_publisher(Bool, '/trigger_objA', 10)
        
        # Subscribe to compressed camera images
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10)
        
        # State variables
        self.locked_id = None
        self.goal_active = False
        self.goal_handle = None
        self.mode = 'search'
        self.last_goal_pose = None
        #self.objA = False

        self.aruco_detected_pub = self.create_publisher(
            Bool,
            '/aruco_detected',
            10)
        
        self.get_logger().info("RamIsBetter node initialized. Ready to detect ArUco markers and navigate towards them!")

    def image_callback(self, msg):
        try:
            # Convert compressed image to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                #if not self.objA:
                    msg = Bool()
                    msg.data = True
                    self.aruco_detected_pub.publish(msg)

                    # Estimate pose of markers
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                    
                    # Find the closest marker
                    distances = [tvecs[i][0][2] for i in range(len(ids))]
                    target_index = np.argmin(distances)
                    
                    marker_id = int(ids[target_index][0])
                    tvec = tvecs[target_index][0]  # Translation vector in camera frame
                    rvec = rvecs[target_index][0]
                    
                    # Lock on to the first detected marker
                    if self.locked_id is None:
                        self.locked_id = marker_id
                        self.get_logger().info(f"Locked on to ArUco marker ID: {marker_id}")
                    
                    if marker_id == self.locked_id:
                        distance_to_marker = float(tvec[2])
                        if distance_to_marker > self.visual_takeover_distance:
                            marker_pose_map = self.transform_marker_to_map(tvec)

                            if marker_pose_map is not None:
                                target_pose = self.calculate_target_pose(marker_pose_map)
                                self.mode = 'nav2'
                                self.send_navigation_goal(target_pose)
                                self.get_logger().info(
                                    f"Nav2 approach to ArUco ID {marker_id} at x={target_pose.pose.position.x:.2f}, y={target_pose.pose.position.y:.2f}")
                            else:
                                self.get_logger().warn("Could not transform marker position to map frame")
                        else:
                            if self.mode != 'visual':
                                self.get_logger().info(
                                    f'Switching to visual alignment for ArUco ID {marker_id}.')
                            self.mode = 'visual'
                            self.cancel_navigation_goal()
                            self.visual_servo_to_marker(tvec, rvec)
                    
                    # Draw detected markers for visualization
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[target_index], tvecs[target_index], 0.05)
                #else:
                    #self.get_logger().info("obj has been done alread")
                
            else:
                # No markers detected, reset lock
                if self.locked_id is not None:
                    self.get_logger().info("ArUco marker lost. Resetting lock.")
                    self.locked_id = None
                    self.active_source_frame = None
                    self.mode = 'search'
                    self.stop_robot()
                    msg = Bool()
                    msg.data = False
                    self.aruco_detected_pub.publish(msg)
            
            # Display the frame (optional, for debugging)
            cv2.imshow("RamIsBetter - ArUco Detection", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def transform_marker_to_map(self, tvec_camera):
        """
        Transform the marker's position from camera frame to map frame.
        """
        for source_frame in self.camera_frame_candidates:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    rclpy.time.Time())

                marker_camera = self.marker_vector_in_source_frame(tvec_camera, source_frame)
                translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                rotation_matrix = self.quaternion_to_rotation_matrix([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])

                marker_map = rotation_matrix @ marker_camera + translation

                if self.active_source_frame != source_frame:
                    self.active_source_frame = source_frame
                    self.get_logger().info(
                        f'Using TF source frame "{source_frame}" for ArUco map transform.')

                return marker_map
            except TransformException:
                continue

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_tf_warn_time > 2.0:
            self.last_tf_warn_time = now_sec
            self.get_logger().warn(
                f'Could not find a TF transform from any of {self.camera_frame_candidates} to "{self.map_frame}".')
        return None

    def marker_vector_in_source_frame(self, tvec_camera, source_frame):
        x_cam, y_cam, z_cam = tvec_camera

        if 'optical' in source_frame:
            return np.array([x_cam, y_cam, z_cam], dtype=float)

        # OpenCV ArUco pose is in optical frame:
        # x right, y down, z forward.
        # Approximate conversion into a forward-left-up robot frame.
        return np.array([z_cam, -x_cam, -y_cam], dtype=float)

    def get_robot_position_in_map(self):
        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            'base_link',
            rclpy.time.Time())
        return np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=float)

    def rotation_matrix_to_yaw(self, rotation_matrix):
        return math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert quaternion to rotation matrix.
        """
        x, y, z, w = quaternion
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    def calculate_target_pose(self, marker_position):
        """
        Calculate a standoff pose on the line from the robot to the marker.
        """
        robot_position = self.get_robot_position_in_map()
        robot_xy = robot_position[:2]
        marker_xy = np.array(marker_position[:2], dtype=float)

        direction = marker_xy - robot_xy
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            target_xy = marker_xy
            yaw = 0.0
        else:
            direction_unit = direction / distance
            target_xy = marker_xy - direction_unit * min(self.standoff_distance, max(distance - 0.05, 0.0))
            yaw = math.atan2(direction[1], direction[0])

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

    def marker_heading_error(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        marker_normal = rotation_matrix[:, 2]
        return math.atan2(float(marker_normal[0]), abs(float(marker_normal[2])) + 1e-6)

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
            self.stop_robot()
            self.get_logger().info(
                f'Aligned with ArUco ID {self.locked_id}: x={x_error:.3f}, z={tvec[2]:.3f}, yaw_err={yaw_error:.3f}')

            #TODO: activate gpio node
            trigger_msg = Bool()
            trigger_msg.data = True
            self.objA_trigger_pub.publish(trigger_msg)
            #self.objA = True   
            return

        cmd = Twist()
        cmd.linear.x = self.linear_gain * distance_error
        cmd.angular.z = -(self.angular_gain_x * x_error + self.angular_gain_yaw * yaw_error)

        if abs(x_error) > 0.03 or abs(yaw_error) > 0.12:
            cmd.linear.x = min(cmd.linear.x, 0.04)

        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed), -self.max_linear_speed)
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(
            f'Visual align ID {self.locked_id}: x_err={x_error:.3f}, z={tvec[2]:.3f}, dist_err={distance_error:.3f}, yaw_err={yaw_error:.3f}')

    def send_navigation_goal(self, pose):
        """
        Send a navigation goal to Nav2.
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available.')
            return
        
        if self.goal_active and self.last_goal_pose is not None:
            last_x, last_y, last_yaw = self.last_goal_pose
            next_x = pose.pose.position.x
            next_y = pose.pose.position.y
            next_yaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
            move_delta = math.hypot(next_x - last_x, next_y - last_y)
            yaw_delta = abs(self.normalize_angle(next_yaw - last_yaw))
            if move_delta < self.goal_update_distance and yaw_delta < self.goal_update_yaw:
                return
            self.cancel_navigation_goal()
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        self.get_logger().info(f'Sending navigation goal to x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}')
        
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
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
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation goal reached successfully!')
        else:
            self.get_logger().warn(f'Navigation goal finished with status={result.status}.')
        self.goal_active = False
        self.goal_handle = None

    def cancel_navigation_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.goal_active = False

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())


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
