import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage # Changed from Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoFollowerCompressed(Node):
    def __init__(self):
        super().__init__('aruco_follower_compressed')
        
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera Matrix
        self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.05

        # Subscribe to COMPRESSED topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed', 
            self.image_callback,
            10)
        
        self.get_logger().info("🚀 Subscribed to COMPRESSED stream. Smoothing out the lag!")

    def image_callback(self, msg):
        try:
            # Decode the compressed JPEG/PNG data back into an OpenCV image
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                # Logic remains the same
                for i, marker_id in enumerate(ids.flatten()):
                    x = tvecs[i][0][0]
                    z = tvecs[i][0][2]
                    distance_xz = np.sqrt(x**2 + z**2)
                    self.get_logger().info(
                        f"ID {marker_id}: "
                        f"X: {x:.2f}m, Z: {z:.2f}m"
                    )
                
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.05)

            cv2.imshow("Compressed RPi Stream", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollowerCompressed()
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
