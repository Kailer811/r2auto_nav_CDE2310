#!/usr/bin/env python3
"""
auto_nav2.py  —  Frontier + A* autonomous maze mapping
=======================================================
Camera and servo code is commented out.
Robot will autonomously explore and map the maze.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import cmath
import time
import heapq
# import threading                  # ← camera thread (disabled)
# import cv2                        # ← ArUco detection (disabled)
# import RPi.GPIO as GPIO           # ← servo control (disabled)
from scipy.ndimage import label

# ═══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# SERVO_PIN       = 18              # disabled
# CAMERA_INDEX    = 0               # disabled
# ARUCO_DICT      = cv2.aruco.DICT_4X4_50   # disabled

# ANGLE_NO_MARKER = 0               # disabled
# ANGLE_DETECTED  = 90              # disabled

# CENTER_TOLERANCE  = 40            # disabled
# ALIGN_TURN_SPEED  = 0.08          # disabled
# ALIGN_DRIVE_SPEED = 0.04          # disabled
# SERVO_HOLD_TIME   = 3.0           # disabled

# ── navigation constants ──────────────────────────────────────────────────
rotatechange  = 0.1
speedchange   = 0.05
stop_distance = 0.25
front_angle   = 30
front_angles  = range(-front_angle, front_angle + 1, 1)

# ── frontier / A* constants ───────────────────────────────────────────────
MIN_FRONTIER_SIZE = 3
OCCUPANCY_THRESH  = 50
UNKNOWN_VAL       = 0
FREE_VAL          = 1


# ═══════════════════════════════════════════════════════════════════════════
#  SERVO HELPERS  (disabled)
# ═══════════════════════════════════════════════════════════════════════════

# def angle_to_duty(angle):
#     return 2.5 + (angle / 180.0) * 10.0

# def setup_servo():
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(SERVO_PIN, GPIO.OUT)
#     pwm = GPIO.PWM(SERVO_PIN, 50)
#     pwm.start(angle_to_duty(ANGLE_NO_MARKER))
#     time.sleep(0.5)
#     return pwm

# def move_servo(pwm, angle):
#     pwm.ChangeDutyCycle(angle_to_duty(angle))
#     time.sleep(0.3)
#     pwm.ChangeDutyCycle(0)


# ═══════════════════════════════════════════════════════════════════════════
#  QUATERNION → EULER
# ═══════════════════════════════════════════════════════════════════════════

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


# ═══════════════════════════════════════════════════════════════════════════
#  A* PATHFINDER
# ═══════════════════════════════════════════════════════════════════════════

def astar(grid, start, goal):
    rows, cols = grid.shape

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def neighbours(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from = {}
    g_score   = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        r, c = current
        for nr, nc in neighbours(r, c):
            if grid[nr, nc] > OCCUPANCY_THRESH:
                continue
            move_cost   = math.hypot(nr - r, nc - c)
            tentative_g = g + move_cost
            if tentative_g < g_score.get((nr, nc), float('inf')):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tentative_g
                f = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_heap, (f, tentative_g, (nr, nc)))
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  FRONTIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_frontiers(occdata):
    free    = (occdata == FREE_VAL)
    unknown = (occdata == UNKNOWN_VAL)
    has_unknown_nb = (
        np.roll(unknown,  1, axis=0) | np.roll(unknown, -1, axis=0) |
        np.roll(unknown,  1, axis=1) | np.roll(unknown, -1, axis=1)
    )
    frontier_mask      = free & has_unknown_nb
    structure          = np.ones((3, 3), dtype=int)
    labelled, n_labels = label(frontier_mask, structure=structure)
    clusters = []
    for region_id in range(1, n_labels + 1):
        rows, cols = np.where(labelled == region_id)
        if len(rows) >= MIN_FRONTIER_SIZE:
            clusters.append((rows, cols))
    return clusters

def best_frontier(clusters, robot_row, robot_col):
    if not clusters:
        return None
    best, best_dist = None, float('inf')
    for rows, cols in clusters:
        c_row = int(np.mean(rows))
        c_col = int(np.mean(cols))
        dist  = math.hypot(c_row - robot_row, c_col - robot_col)
        if dist < best_dist:
            best_dist = dist
            best      = (c_row, c_col)
    return best


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN NODE
# ═══════════════════════════════════════════════════════════════════════════

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')

        # ── ROS2 publishers / subscribers ────────────────────────────────
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.odom_subscription

        self.roll    = 0
        self.pitch   = 0
        self.yaw     = 0
        self.robot_x = 0.0
        self.robot_y = 0.0

        self.map_info = None

        # ── FIX: map QoS must match SLAM toolbox (TRANSIENT_LOCAL) ───────
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.occ_subscription = self.create_subscription(
            OccupancyGrid, 'map', self.occ_callback, map_qos)
        self.occ_subscription
        self.occdata = np.array([])

        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.scan_subscription
        self.laser_range = np.array([])

        # ── servo + camera setup (disabled) ──────────────────────────────
        # self.pwm            = setup_servo()
        # self.camera         = cv2.VideoCapture(CAMERA_INDEX)
        # self.aruco_detector = self._setup_aruco()
        # self.frame_width    = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height   = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.marker_center  = None
        # self.marker_lock    = threading.Lock()
        # self.camera_thread  = threading.Thread(target=self._camera_loop, daemon=True)
        # self.camera_thread.start()

        self.get_logger().info('AutoNav started — maze mapping mode.')

    # ── ArUco detector setup (disabled) ──────────────────────────────────

    # def _setup_aruco(self):
    #     dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    #     parameters = cv2.aruco.DetectorParameters()
    #     return cv2.aruco.ArucoDetector(dictionary, parameters)

    # ── camera loop (disabled) ────────────────────────────────────────────

    # def _camera_loop(self):
    #     while True:
    #         success, frame = self.camera.read()
    #         if not success:
    #             time.sleep(0.05)
    #             continue
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         corners, ids, _ = self.aruco_detector.detectMarkers(gray)
    #         with self.marker_lock:
    #             if ids is not None and len(ids) > 0:
    #                 pts = corners[0][0]
    #                 cx  = int(np.mean(pts[:, 0]))
    #                 cy  = int(np.mean(pts[:, 1]))
    #                 self.marker_center = (cx, cy)
    #             else:
    #                 self.marker_center = None
    #         time.sleep(0.03)

    # def _get_marker_center(self):
    #     with self.marker_lock:
    #         return self.marker_center

    # ── ArUco alignment sequence (disabled) ──────────────────────────────

    # def handle_aruco(self):
    #     self.get_logger().info('--- ArUco spotted! Starting alignment ---')
    #     self.stopbot()
    #     frame_center_x = self.frame_width // 2
    #     # Step 1: rotate to center marker
    #     while rclpy.ok():
    #         mc = self._get_marker_center()
    #         if mc is None:
    #             self.stopbot()
    #             return
    #         error = mc[0] - frame_center_x
    #         if abs(error) <= CENTER_TOLERANCE:
    #             self.stopbot()
    #             break
    #         twist = Twist()
    #         twist.linear.x  = 0.0
    #         twist.angular.z = -ALIGN_TURN_SPEED if error > 0 else ALIGN_TURN_SPEED
    #         self.publisher_.publish(twist)
    #         rclpy.spin_once(self)
    #         time.sleep(0.05)
    #     # Step 2: drive over marker
    #     while rclpy.ok():
    #         mc = self._get_marker_center()
    #         if mc is None:
    #             self.stopbot()
    #             break
    #         error      = mc[0] - frame_center_x
    #         correction = -(error / self.frame_width) * 0.15
    #         twist = Twist()
    #         twist.linear.x  = ALIGN_DRIVE_SPEED
    #         twist.angular.z = correction
    #         self.publisher_.publish(twist)
    #         rclpy.spin_once(self)
    #         time.sleep(0.05)
    #     # Step 3: fire servo
    #     move_servo(self.pwm, ANGLE_DETECTED)
    #     # Step 4: hold and reset
    #     time.sleep(SERVO_HOLD_TIME)
    #     move_servo(self.pwm, ANGLE_NO_MARKER)

    # ── ROS2 callbacks ───────────────────────────────────────────────────

    def odom_callback(self, msg):
        orientation_quat = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(
            orientation_quat.x, orientation_quat.y,
            orientation_quat.z, orientation_quat.w)
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def occ_callback(self, msg):
        msgdata      = np.array(msg.data)
        oc2          = msgdata + 1
        self.occdata = np.uint8(oc2.reshape(msg.info.height, msg.info.width))
        self.map_info = msg.info

    def scan_callback(self, msg):
        self.laser_range = np.array(msg.ranges)
        self.laser_range[self.laser_range == 0] = np.nan

    # ── coordinate helpers ───────────────────────────────────────────────

    def world_to_cell(self, wx, wy):
        res      = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        col = int((wx - origin_x) / res)
        row = int((wy - origin_y) / res)
        return row, col

    def cell_to_world(self, row, col):
        res      = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        wx = origin_x + (col + 0.5) * res
        wy = origin_y + (row + 0.5) * res
        return wx, wy

    # ── rotation ─────────────────────────────────────────────────────────

    def rotatebot(self, rot_angle):
        twist       = Twist()
        current_yaw = self.yaw
        self.get_logger().info('Current: %f' % math.degrees(current_yaw))
        c_yaw        = complex(math.cos(current_yaw), math.sin(current_yaw))
        target_yaw   = current_yaw + math.radians(rot_angle)
        c_target_yaw = complex(math.cos(target_yaw), math.sin(target_yaw))
        self.get_logger().info('Desired: %f' % math.degrees(cmath.phase(c_target_yaw)))
        c_change     = c_target_yaw / c_yaw
        c_change_dir = np.sign(c_change.imag)
        twist.linear.x  = 0.0
        twist.angular.z = c_change_dir * rotatechange
        self.publisher_.publish(twist)
        c_dir_diff = c_change_dir
        while c_change_dir * c_dir_diff > 0:
            rclpy.spin_once(self)
            current_yaw  = self.yaw
            c_yaw        = complex(math.cos(current_yaw), math.sin(current_yaw))
            c_change     = c_target_yaw / c_yaw
            c_dir_diff   = np.sign(c_change.imag)
        self.get_logger().info('End Yaw: %f' % math.degrees(current_yaw))
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    def stopbot(self):
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    # ── follow A* path ───────────────────────────────────────────────────

    def follow_path(self, path):
        if not path:
            return

        for (row, col) in path[1:]:

            # ArUco check disabled
            # if self._get_marker_center() is not None:
            #     self.handle_aruco()

            target_x, target_y = self.cell_to_world(row, col)
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            angle_to_target = math.degrees(math.atan2(dy, dx))
            angle_to_rotate = angle_to_target - math.degrees(self.yaw)
            angle_to_rotate = (angle_to_rotate + 180) % 360 - 180
            self.rotatebot(angle_to_rotate)

            twist = Twist()
            twist.linear.x  = speedchange
            twist.angular.z = 0.0
            self.publisher_.publish(twist)

            while rclpy.ok():
                rclpy.spin_once(self)

                # ArUco check while driving disabled
                # if self._get_marker_center() is not None:
                #     self.stopbot()
                #     self.handle_aruco()
                #     self.publisher_.publish(twist)

                # obstacle check
                if self.laser_range.size != 0:
                    lri = (self.laser_range[front_angles] < stop_distance).nonzero()
                    if len(lri[0]) > 0:
                        self.stopbot()
                        self.get_logger().info('Obstacle — replanning.')
                        return

                # arrival check
                if math.hypot(target_x - self.robot_x, target_y - self.robot_y) < 0.15:
                    break

            self.stopbot()
            time.sleep(0.1)

    # ── pick direction ───────────────────────────────────────────────────

    def pick_direction(self):
        if self.occdata.size == 0 or self.map_info is None:
            self.get_logger().info('Map not ready — laser fallback')
            self._laser_fallback()
            return

        clusters = find_frontiers(self.occdata)
        if not clusters:
            self.get_logger().info('No frontiers — maze fully mapped!')
            return

        robot_row, robot_col = self.world_to_cell(self.robot_x, self.robot_y)
        goal_cell = best_frontier(clusters, robot_row, robot_col)
        if goal_cell is None:
            self._laser_fallback()
            return

        self.get_logger().info('Frontier target: row=%d col=%d' % goal_cell)
        path = astar(self.occdata, start=(robot_row, robot_col), goal=goal_cell)
        if not path:
            self.get_logger().info('A* no path — laser fallback')
            self._laser_fallback()
            return

        self.get_logger().info('A* path length: %d cells' % len(path))
        self.follow_path(path)

    def _laser_fallback(self):
        if self.laser_range.size != 0:
            lr2i = np.nanargmax(self.laser_range)
        else:
            lr2i = 0
        self.rotatebot(float(lr2i))
        twist = Twist()
        twist.linear.x  = speedchange
        twist.angular.z = 0.0
        time.sleep(1)
        self.publisher_.publish(twist)

    # ── wait for sensors ─────────────────────────────────────────────────

    def wait_for_data(self):
        self.get_logger().info('Waiting for map and laser data...')
        count = 0
        while rclpy.ok():
            rclpy.spin_once(self)
            map_ready   = self.occdata.size != 0 and self.map_info is not None
            laser_ready = self.laser_range.size != 0
            count += 1
            if count % 20 == 0:
                self.get_logger().info(
                    'Map ready: %s | Laser ready: %s' % (map_ready, laser_ready))
            if map_ready and laser_ready:
                self.get_logger().info('All data ready! Starting maze mapping.')
                return
            time.sleep(0.1)

    # ── main loop ─────────────────────────────────────────────────────────

    def mover(self):
        try:
            self.wait_for_data()
            self.pick_direction()

            while rclpy.ok():

                # ArUco check disabled
                # if self._get_marker_center() is not None:
                #     self.handle_aruco()

                if self.laser_range.size != 0:
                    lri = (self.laser_range[front_angles] < float(stop_distance)).nonzero()
                    if len(lri[0]) > 0:
                        self.stopbot()
                        self.pick_direction()

                rclpy.spin_once(self)

        except Exception as e:
            print(e)
        finally:
            self.stopbot()
            # servo cleanup disabled
            # move_servo(self.pwm, ANGLE_NO_MARKER)
            # self.pwm.stop()
            # GPIO.cleanup()
            # self.camera.release()


def main(args=None):
    rclpy.init(args=args)
    auto_nav = AutoNav()
    auto_nav.mover()
    auto_nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()