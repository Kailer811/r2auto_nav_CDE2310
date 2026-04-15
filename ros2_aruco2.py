import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from enum import Enum, auto


class State(Enum):
    SCANNING        = auto()  # Waiting for ArUco detection
    TURN_PARALLEL   = auto()  # Rotate to be parallel to the marker (yaw = marker yaw)
    STRAFE_X        = auto()  # Drive forward/back to align X offset
    TURN_FACE       = auto()  # Rotate 90° to face the marker
    APPROACH_Z      = auto()  # Drive forward until Z ≈ TARGET_Z
    DONE            = auto()


# ── Tunable constants ──────────────────────────────────────────────────────────
TARGET_Z        = 0.16   # metres — stop when this close to marker
X_THRESH        = 0.05  # metres — acceptable X alignment error
Z_THRESH        = 0.05  # metres — acceptable Z distance error
ANGLE_THRESH    = 0.03  # radians — acceptable heading error (~1.7°)

LINEAR_SPEED    = 0.15  # m/s
ANGULAR_SPEED   = 0.4   # rad/s

MARKER_SIZE     = 0.04  # metres — physical ArUco marker side length
# ──────────────────────────────────────────────────────────────────────────────


def yaw_from_quaternion(q):
    """Extract yaw (rotation about Z) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, current):
    """Smallest signed difference between two angles (radians)."""
    d = target - current
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


class ArucoStateMachine(Node):

    def __init__(self):
        super().__init__('aruco_state_machine')

        # ── ArUco setup ──────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Replace with your real calibrated values
        self.camera_matrix = np.array([
            [475.0,    0.0, 320.0],
            [   0.0, 475.0, 240.0],
            [   0.0,    0.0,   1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # ── State ────────────────────────────────────────────────────────────
        self.state            = State.SCANNING
        self.current_yaw      = 0.0   # live odom yaw
        self.target_yaw       = None  # goal heading for turn states
        self.target_x_dist    = None  # signed X distance to drive
        self.target_z_dist    = None  # Z distance remaining to cover
        self.strafe_turn_sign = 1.0

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(
            CompressedImage, '/image_raw/compressed',
            self.image_callback, 10)

        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        self.create_timer(0.05, self.control_loop)  # 20 Hz control loop

        self.get_logger().info("ArUco state machine ready — scanning...")

    # ── Odometry ─────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    # ── Vision (only used in SCANNING) ───────────────────────────────────────

    def image_callback(self, msg):
        if self.state != State.SCANNING:
            return  # YOLO approach: ignore camera once we have a fix

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is None or len(ids) == 0:
                return

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, self.camera_matrix, self.dist_coeffs)

            # Use first detected marker
            rvec = rvecs[0]
            tvec = tvecs[0][0]

            # --- Marker pose in camera frame ---
            # tvec = [x, y, z] of marker relative to camera
            marker_x = float(tvec[0])   # left/right
            marker_z = float(tvec[2])   # depth (forward)

            # --- Marker yaw relative to camera ---
            R, _ = cv2.Rodrigues(rvec)
            #R_inv = R.T
            #cam_in_marker = -R_inv @ tvec
            #x = float(cam_in_marker[0])
            #z = float(cam_in_marker[2])
            # The marker's forward axis (its Z) projected onto the ground plane
            marker_forward = R[:, 2]  # third column
            marker_yaw_cam = math.atan2(marker_forward[0], marker_forward[2])

            self.get_logger().info(
                f"[SCAN] Marker detected — "
                f"X: {marker_x:.3f} m  Z: {marker_z:.3f} m  "
                f"yaw_rel: {math.degrees(marker_yaw_cam):.1f}°"
            )

            # Lock in the plan
            self._plan_sequence(marker_x, marker_z, marker_yaw_cam)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    # ── Plan the full sequence from one snapshot ──────────────────────────────

    def _plan_sequence(self, marker_x, marker_z, marker_yaw_cam):
        """
        From a single detection we derive:
          1. How much to rotate to be parallel to the marker face
          2. How far to drive sideways (X) to centre on the marker
          3. A 90° turn to face the marker
          4. How far to drive forward to reach TARGET_Z
        """
        # Step 1: angle to be parallel = current yaw + marker_yaw_cam
        # (marker_yaw_cam == 0 means marker faces directly toward us → we are already parallel)
        folded = False
        if marker_yaw_cam > math.pi / 2:
            marker_yaw_cam -= math.pi
            folded = True
        elif marker_yaw_cam < -math.pi / 2:
            marker_yaw_cam += math.pi
            folded = True

        self.target_yaw   = self.current_yaw + marker_yaw_cam
        self.target_x_dist = -marker_x          # positive = move right
        self.target_z_dist = marker_z - TARGET_Z  # how much z remains after approach

        self.strafe_turn_sign = -1.0 if folded else 1.0

        self.state = State.TURN_PARALLEL
        self.get_logger().info(
            f"current_yaw: {math.degrees(self.current_yaw):.1f}°  "
            f"marker_yaw_cam: {math.degrees(marker_yaw_cam):.1f}°  "
            f"target_yaw: {math.degrees(self.target_yaw):.1f}°  "
            f"marker_x: {marker_x:.3f}  marker_z: {marker_z:.3f}"
        )
        self.get_logger().info(
            f"[PLAN] turn_parallel → yaw {math.degrees(self.target_yaw):.1f}°  |  "
            f"strafe_x {self.target_x_dist:.3f} m  |  "
            f"approach_z remaining {self.target_z_dist:.3f} m"
        )

    # ── Main control loop ─────────────────────────────────────────────────────

    def control_loop(self):
        cmd = Twist()

        if self.state == State.SCANNING:
            # Nothing to command — wait for camera callback
            pass

        elif self.state == State.TURN_PARALLEL:
            err = angle_diff(self.target_yaw, self.current_yaw)
            if abs(err) < ANGLE_THRESH:
                self.get_logger().info("[STATE] Parallel achieved → STRAFE_X")
                self.state = State.STRAFE_X
            else:
                cmd.angular.z = ANGULAR_SPEED * np.sign(err)

        elif self.state == State.STRAFE_X:
            # Drive along the robot's X axis (forward = local X when parallel)
            # We use linear.x as the drive axis; the robot is now parallel to the marker
            if abs(self.target_x_dist) < X_THRESH:
                self.get_logger().info("[STATE] X aligned → TURN_FACE")
                self.state = State.TURN_FACE
                # Goal: turn 90° toward the marker
                self.target_yaw = self.current_yaw - math.copysign(
                    math.pi / 2,
                    self.target_x_dist * self.strafe_turn_sign  # ← apply fold correction
                )
                # If marker was to the right (positive x) we need to turn right (-90°)
                # If marker was to the left  (negative x) we turn left  (+90°)
                # Adjust sign convention to match your robot's frame if needed
            else:
                direction = math.copysign(1.0, self.target_x_dist)
                cmd.linear.x = LINEAR_SPEED * direction
                # Consume distance (open-loop at 20 Hz → dt = 0.05 s)
                self.target_x_dist -= LINEAR_SPEED * direction * 0.05

        elif self.state == State.TURN_FACE:
            err = angle_diff(self.target_yaw, self.current_yaw)
            if abs(err) < ANGLE_THRESH:
                self.get_logger().info("[STATE] Facing marker → APPROACH_Z")
                self.state = State.APPROACH_Z
            else:
                cmd.angular.z = ANGULAR_SPEED * np.sign(err)

        elif self.state == State.APPROACH_Z:
            if self.target_z_dist <= Z_THRESH:
                self.get_logger().info("[STATE] Reached target Z → DONE")
                self.state = State.DONE
            else:
                cmd.linear.x = LINEAR_SPEED
                self.target_z_dist -= LINEAR_SPEED * 0.05

        elif self.state == State.DONE:
            # All stop
            self.get_logger().info("[DONE] Mission complete.", once=True)

        self.cmd_pub.publish(cmd)

    # ─────────────────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = ArucoStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety stop
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()