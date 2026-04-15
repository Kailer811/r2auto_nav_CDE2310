import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class ArucoFollowerCompressed(Node):
    def __init__(self):
        super().__init__('aruco_follower_compressed')

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Replace these with your REAL calibrated values
        self.camera_matrix = np.array([
            [475.0, 0.0, 320.0],
            [0.0, 475.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.marker_size = 0.04  # meters

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.get_logger().info("Subscribed to compressed stream")

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            if ids is not None and len(ids) > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_size,
                    self.camera_matrix,
                    self.dist_coeffs
                )

                cv2.aruco.drawDetectedMarkers(frame, corners)

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i][0]  # marker in camera frame

                    # Convert marker->camera into camera->marker
                    R, _ = cv2.Rodrigues(rvec)
                    R_inv = R.T
                    cam_in_marker = -R_inv @ tvec

                    x = float(cam_in_marker[0])
                    z = float(cam_in_marker[2])
                    yaw = math.degrees(math.atan2(x, z))

                    self.get_logger().info(
                        f"ID {marker_id}: "
                        f"X: {x:.2f} m, Z: {z:.2f} m, Yaw: {yaw:.2f} deg" 
                    )

                    cv2.drawFrameAxes(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec.reshape(3, 1),
                        0.05
                    )

            cv2.imshow("Compressed RPi Stream", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")

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