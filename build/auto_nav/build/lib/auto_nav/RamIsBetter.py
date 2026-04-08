#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import cv2
import numpy as np
import tf2_ros
import math
from tf2_ros import TransformException


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
        
        # Navigation client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Subscribe to compressed camera images
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10)
        
        # State variables
        self.locked_id = None
        self.goal_active = False
        
        self.get_logger().info("RamIsBetter node initialized. Ready to detect ArUco markers and navigate towards them!")

    def image_callback(self, msg):
        try:
            # Convert compressed image to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                # Estimate pose of markers
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                # Find the closest marker
                distances = [tvecs[i][0][2] for i in range(len(ids))]
                target_index = np.argmin(distances)
                
                marker_id = ids[target_index][0]
                tvec = tvecs[target_index][0]  # Translation vector in camera frame
                
                # Lock on to the first detected marker
                if self.locked_id is None:
                    self.locked_id = marker_id
                    self.get_logger().info(f"Locked on to ArUco marker ID: {marker_id}")
                
                if marker_id == self.locked_id:
                    # Transform marker position from camera frame to map frame
                    marker_pose_map = self.transform_marker_to_map(tvec)
                    
                    if marker_pose_map is not None:
                        # Navigate to a position in front of the marker
                        target_pose = self.calculate_target_pose(marker_pose_map)
                        self.send_navigation_goal(target_pose)
                        
                        self.get_logger().info(f"Navigating towards ArUco marker ID {marker_id} at position x={target_pose.pose.position.x:.2f}, y={target_pose.pose.position.y:.2f}")
                    else:
                        self.get_logger().warn("Could not transform marker position to map frame")
                
                # Draw detected markers for visualization
                cv2.aruco.drawDetectedMarkers(frame, corners)
                
            else:
                # No markers detected, reset lock
                if self.locked_id is not None:
                    self.get_logger().info("ArUco marker lost. Resetting lock.")
                    self.locked_id = None
                    self.active_source_frame = None
            
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

    def send_navigation_goal(self, pose):
        """
        Send a navigation goal to Nav2.
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available.')
            return
        
        if self.goal_active:
            self.get_logger().info("Navigation goal already active, skipping new goal.")
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        self.get_logger().info(f'Sending navigation goal to x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}')
        
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
        self.goal_active = True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Navigation goal rejected.')
            self.goal_active = False
            return
        
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


def main(args=None):
    rclpy.init(args=args)
    node = RamIsBetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
