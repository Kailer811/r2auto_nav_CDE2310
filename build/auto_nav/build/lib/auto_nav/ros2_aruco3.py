import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class ArucoFollowerCompressed(Node):
    def __init__(self):
        super().__init__('aruco_follower_compressed')
        
        self.bridge = CvBridge()
        
        # Modern OpenCV ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera Matrix - IMPORTANT: Ensure these match your actual camera resolution
        # These values assume a roughly 640x480 resolution. 
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.05 # 5cm

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed', 
            self.image_callback,
            10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # CONFIGURATION
        self.target_distance = 0.30  # Stop 30cm away
        self.target_x = 0.0         # Aim for center of image
        
        # GAINS (Tweak these!)
        self.k_lin = 0.4   # Forward speed gain
        self.k_ang = 1.2   # Rotational speed gain (higher = more aggressive turning)
        
        self.locked_id = None
        self.get_logger().info("Aruco Follower Started - Looking for markers...")

    def image_callback(self, msg):
        try:
            # 1. Decode Image
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Detect Markers
            corners, ids, _ = self.detector.detectMarkers(gray)
            cmd = Twist()

            if ids is not None:
                # Calculate pose for the first detected marker
                # Note: estimatePoseSingleMarkers is deprecated in newer CV2, 
                # but works for simple use cases.
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                # Use the first marker found or implement your "locked_id" logic
                idx = 0 
                marker_id = ids[idx][0]
                
                # tvecs contains [x, y, z] where:
                # x: left/right relative to camera center
                # z: distance forward from camera
                x_err = self.target_x - tvecs[idx][0][0]
                z_err = tvecs[idx][0][2] - self.target_distance

                # 3. SMOOTH CONTROL LOGIC
                # We move and turn AT THE SAME TIME for stability
                
                # Angular Control (Turning)
                # If x is positive, marker is to the right, we need negative angular.z to turn right
                cmd.angular.z = -self.k_ang * x_err
                
                # Linear Control (Forward/Backward)
                # Only move forward if the error is significant
                if abs(z_err) > 0.02:
                    cmd.linear.x = self.k_lin * z_err
                else:
                    cmd.linear.x = 0.0

                # 4. SAFETY LIMITS
                cmd.linear.x = np.clip(cmd.linear.x, -0.2, 0.2)   # Max 0.2 m/s
                cmd.angular.z = np.clip(cmd.angular.z, -1.0, 1.0) # Max 1.0 rad/s

                # If very close, stop moving forward to avoid crashing
                if tvecs[idx][0][2] < 0.10:
                    cmd.linear.x = 0.0

                self.cmd_pub.publish(cmd)
                
                # Draw on frame for debugging
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[idx], tvecs[idx], 0.03)

            else:
                # No marker found: STOP
                self.cmd_pub.publish(Twist())

            cv2.imshow("Follower Debug", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollowerCompressed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist()) # Emergency stop
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()