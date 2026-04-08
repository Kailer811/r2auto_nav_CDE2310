import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage # Changed from Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PointStamped
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformException

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
        
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        self.target_distance = 0.25  # meters from marker
        self.target_x = -0.05
        self.marker_actions = {
            0: "fire",
            1: "stop",
            2: "scan"
        }
        self.locked_id = None

        self.get_logger().info("🚀 Subscribed to COMPRESSED stream. Smoothing out the lag!")

    def image_callback(self, msg):
        try:
            # Decode the compressed JPEG/PNG data back into an OpenCV image
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            cmd = Twist()

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

                distances = [tvecs[i][0][2] for i in range(len(ids))]
                target_index = np.argmin(distances)
                marker_id = ids[target_index][0]
                if marker_id != self.locked_id:
                    return  # ignore others
                try:
                    p_cam = PointStamped()
                    p_cam.header.frame_id = "camera_optical_frame"   # ⚠️ adjust if needed
                    p_cam.header.stamp = rclpy.time.Time().to_msg()

                    # OpenCV tvec (camera frame)
                    p_cam.point.x = float(tvecs[target_index][0][0])
                    p_cam.point.y = float(tvecs[target_index][0][1])
                    p_cam.point.z = float(tvecs[target_index][0][2])

                    # Transform to map frame
                    p_map = self.tf_buffer.transform(
                        p_cam,
                        "map",
                        timeout=rclpy.duration.Duration(seconds=0.2)
                    )

                    goal_x = p_map.point.x
                    goal_y = p_map.point.y

                    self.get_logger().info(
                        f"MAP GOAL → x={goal_x:.2f}, y={goal_y:.2f}"
                    )

                except TransformException as e:
                    self.get_logger().warn(f"TF transform failed: {e}")
                    return
               


                self.cmd_pub.publish(cmd)

                self.get_logger().info(
                    f"[ID {marker_id}] y_map={goal_x:.2f}, y_map={goal_y:.2f}"
                )
            else:
                # LOST MARKER → STOP + RESET LOCK
                self.cmd_pub.publish(Twist())
                self.locked_id = None

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
