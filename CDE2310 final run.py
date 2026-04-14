#!/usr/bin/env python3

import heapq
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Bool
from tf2_ros import TransformException


UNKNOWN_VALUE = -1
WALL_THRESHOLD = 60
POOL_SIZE = 2
WALL_INFLATION_RADIUS = 1
MIN_FRONTIER_SIZE = 3
MIN_FREE_CELLS_TO_PLAN = 20
MAX_FRONTIER_GOALS_PER_CLUSTER = 4
MAX_FRONTIER_CLUSTERS = 8
BLOCKED_GOAL_COOLDOWN = 25.0
BLOCKED_GOAL_RADIUS = 2
PROGRESS_TIMEOUT = 15.0
MIN_PROGRESS_DISTANCE = 0.08
PLANNER_PERIOD = 1.0
STARTUP_TIMEOUT = 20.0
GOAL_SEARCH_RADIUS = 4
FRONT_HALF_ANGLE_DEG = 18.0
FRONT_STOP_DISTANCE = 0.24
BLOCKED_RECOVERY_WAIT = 2.5
BACKUP_DURATION = 1.2
BACKUP_SPEED = -0.08
VISITED_MARK_RADIUS = 2
MIN_UNVISITED_TO_CONTINUE = 10
COVERAGE_WAYPOINT_SPACING = 4

MODE_SEARCH = 'search'
MODE_DOCKING = 'docking'
MODE_WAIT_AFTER_TRIGGER = 'wait_after_trigger'
MODE_FINISHED = 'finished'

CARDINAL_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_CONNECTED_NEIGHBORS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def euler_from_quaternion(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def heuristic(cell_a, cell_b):
    return math.hypot(cell_b[0] - cell_a[0], cell_b[1] - cell_a[1])


def astar(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    if grid[start] != 0 or grid[goal] != 0:
        return []

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, g_cost, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        row, col = current
        for d_row, d_col in EIGHT_CONNECTED_NEIGHBORS:
            next_row = row + d_row
            next_col = col + d_col
            next_cell = (next_row, next_col)
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if grid[next_cell] != 0:
                continue
            step_cost = math.sqrt(2.0) if d_row != 0 and d_col != 0 else 1.0
            next_g = g_cost + step_cost
            if next_g >= g_score.get(next_cell, float('inf')):
                continue
            g_score[next_cell] = next_g
            came_from[next_cell] = current
            heapq.heappush(
                open_heap,
                (next_g + heuristic(next_cell, goal), next_g, next_cell),
            )

    return []


class AutoNavNode(Node):
    def __init__(self):
        super().__init__('cde2310_auto_nav')

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_data = np.empty((0, 0), dtype=np.int16)
        self.map_resolution = None
        self.map_origin = None
        self.scan_ranges = np.array([], dtype=float)
        self.scan_angles = np.array([], dtype=float)

        self.current_goal_cell = None
        self.goal_handle = None
        self.goal_in_flight = False
        self.blocked_goals = {}
        self.last_progress_time = time.time()
        self.last_progress_pose = None
        self.last_status_log_time = 0.0
        self.system_ready = False
        self.front_blocked_since = None
        self.last_backup_time = 0.0
        self.cancel_goal_on_accept = False
        self.navigation_paused = False
        self.navigation_stopped_forever = False

        self.visited_cells = set()
        self.coverage_mode = False
        self.get_logger().info('AutoNav node started.')

        self.create_timer(PLANNER_PERIOD, self.exploration_step)

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.map_data = np.array(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isinf(ranges)] = np.nan
        ranges[ranges == 0.0] = np.nan
        if msg.range_min > 0.0:
            ranges[ranges < msg.range_min] = np.nan
        if msg.range_max > 0.0:
            ranges[ranges > msg.range_max] = np.nan
        angles = msg.angle_min + np.arange(len(ranges), dtype=float) * msg.angle_increment
        self.scan_ranges = ranges
        self.scan_angles = np.mod(angles, 2.0 * math.pi)

    def wait_for_system_ready(self, timeout_sec=STARTUP_TIMEOUT):
        self.get_logger().info('Waiting for /map, TF map->base_link, /scan, and Nav2...')
        start_time = time.time()
        while rclpy.ok() and time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            map_ready = (
                self.map_data.size != 0
                and self.map_resolution is not None
                and self.map_origin is not None
            )
            scan_ready = (self.scan_ranges.size != 0 and self.scan_angles.size != 0)
            tf_ready = self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time())
            nav_ready = self.nav_client.wait_for_server(timeout_sec=0.0)
            if map_ready and scan_ready and tf_ready and nav_ready:
                self.system_ready = True
                self.get_logger().info('AutoNav system ready.')
                return True
        self.get_logger().warn('AutoNav system not ready in time.')
        return False

    def pause_navigation(self, reason=''):
        if self.navigation_stopped_forever:
            return
        if self.navigation_paused:
            return
        self.navigation_paused = True
        self.cancel_goal_on_accept = True
        log_reason = f' Reason: {reason}' if reason else ''
        self.get_logger().info(f'Navigation paused.{log_reason}')
        self.cancel_active_goal('paused for ArUco docking', block_current_goal=False)
        self.stop_robot()

    def resume_navigation(self):
        if self.navigation_stopped_forever:
            self.get_logger().info('Resume ignored because navigation is finished.')
            return
        if not self.navigation_paused:
            return
        self.navigation_paused = False
        self.cancel_goal_on_accept = False
        self.front_blocked_since = None
        self.last_progress_pose = None
        self.last_progress_time = time.time()
        self.get_logger().info('Navigation resumed after docking wait.')

    def stop_forever(self):
        if self.navigation_stopped_forever:
            return
        self.navigation_stopped_forever = True
        self.navigation_paused = True
        self.cancel_goal_on_accept = True
        self.cancel_active_goal(
            'both trigger_objA and trigger_objB completed',
            block_current_goal=False,
        )
        self.stop_robot()
        self.get_logger().info('Navigation stopped permanently.')

    def get_pose(self):
        transform = self.tf_buffer.lookup_transform(
            'map', 'base_link', rclpy.time.Time())
        _, _, yaw = euler_from_quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w)
        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            yaw,
        )

    def sector_distance(self, center_deg, half_width_deg):
        if self.scan_ranges.size == 0 or self.scan_angles.size == 0:
            return None
        center_rad = math.radians(center_deg) % (2.0 * math.pi)
        half_width_rad = math.radians(half_width_deg)
        angle_error = np.angle(np.exp(1j * (self.scan_angles - center_rad)))
        mask = np.abs(angle_error) <= half_width_rad
        if not np.any(mask):
            return None
        sector_ranges = self.scan_ranges[mask]
        sector_ranges = sector_ranges[~np.isnan(sector_ranges)]
        if sector_ranges.size == 0:
            return None
        return float(np.percentile(sector_ranges, 20))

    def front_clearance(self):
        return self.sector_distance(0.0, FRONT_HALF_ANGLE_DEG)

    def pool_map(self, occ_grid):
        height, width = occ_grid.shape
        pad_h = (POOL_SIZE - height % POOL_SIZE) % POOL_SIZE
        pad_w = (POOL_SIZE - width % POOL_SIZE) % POOL_SIZE
        if pad_h > 0 or pad_w > 0:
            occ_grid = np.pad(
                occ_grid,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=UNKNOWN_VALUE,
            )
        pooled_h = occ_grid.shape[0] // POOL_SIZE
        pooled_w = occ_grid.shape[1] // POOL_SIZE
        pooled = np.full((pooled_h, pooled_w), UNKNOWN_VALUE, dtype=np.int16)
        for row in range(pooled_h):
            for col in range(pooled_w):
                block = occ_grid[row * POOL_SIZE:(row + 1) * POOL_SIZE,
                                 col * POOL_SIZE:(col + 1) * POOL_SIZE]
                pooled[row, col] = int(np.max(block))
        return pooled

    def classify_grid(self, pooled_occ):
        grid = np.full_like(pooled_occ, 2, dtype=np.uint8)
        grid[(pooled_occ >= 0) & (pooled_occ < WALL_THRESHOLD)] = 0
        grid[pooled_occ >= WALL_THRESHOLD] = 1
        return grid

    def inflate_walls(self, grid):
        inflated = grid.copy()
        rows, cols = grid.shape
        for row, col in np.argwhere(grid == 1):
            for d_row in range(-WALL_INFLATION_RADIUS, WALL_INFLATION_RADIUS + 1):
                for d_col in range(-WALL_INFLATION_RADIUS, WALL_INFLATION_RADIUS + 1):
                    nr, nc = row + d_row, col + d_col
                    if 0 <= nr < rows and 0 <= nc < cols and inflated[nr, nc] == 0:
                        inflated[nr, nc] = 1
        return inflated

    def world_to_pooled_cell(self, world_x, world_y, pooled_shape):
        map_x = (world_x - self.map_origin.x) / self.map_resolution
        map_y = (world_y - self.map_origin.y) / self.map_resolution
        pooled_col = int(math.floor(map_x / POOL_SIZE))
        pooled_row = int(math.floor(map_y / POOL_SIZE))
        pooled_col = min(max(pooled_col, 0), pooled_shape[1] - 1)
        pooled_row = min(max(pooled_row, 0), pooled_shape[0] - 1)
        return pooled_row, pooled_col

    def pooled_cell_to_world(self, row, col):
        map_x = (col * POOL_SIZE + POOL_SIZE / 2.0) * self.map_resolution + self.map_origin.x
        map_y = (row * POOL_SIZE + POOL_SIZE / 2.0) * self.map_resolution + self.map_origin.y
        return map_x, map_y

    def nearest_free_to_robot(self, grid, robot_cell, allow_traverse_occupied=False):
        if grid[robot_cell] == 0:
            return robot_cell
        queue = deque([robot_cell])
        visited = {robot_cell}
        rows, cols = grid.shape
        while queue:
            row, col = queue.popleft()
            for d_row, d_col in CARDINAL_NEIGHBORS:
                next_cell = (row + d_row, col + d_col)
                nr, nc = next_cell
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if next_cell in visited:
                    continue
                if grid[next_cell] == 2:
                    continue
                if grid[next_cell] == 1 and not allow_traverse_occupied:
                    continue
                if grid[next_cell] == 0:
                    return next_cell
                visited.add(next_cell)
                queue.append(next_cell)
        return None

    def mark_visited(self, robot_cell, grid_shape):
        row, col = robot_cell
        rows, cols = grid_shape
        for d_row in range(-VISITED_MARK_RADIUS, VISITED_MARK_RADIUS + 1):
            for d_col in range(-VISITED_MARK_RADIUS, VISITED_MARK_RADIUS + 1):
                nr, nc = row + d_row, col + d_col
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.visited_cells.add((nr, nc))

    def get_unvisited_free_cells(self, inflated_grid):
        free_cells = set(zip(*np.where(inflated_grid == 0)))
        return free_cells - self.visited_cells

    def choose_coverage_goal(self, robot_cell, inflated_grid):
        unvisited = self.get_unvisited_free_cells(inflated_grid)
        if len(unvisited) < MIN_UNVISITED_TO_CONTINUE:
            return None

        robot_row, robot_col = robot_cell
        candidates = [
            cell for cell in unvisited
            if math.hypot(cell[0] - robot_row, cell[1] - robot_col) >= COVERAGE_WAYPOINT_SPACING
            and cell not in self.blocked_goals
        ]
        if not candidates:
            return None

        best_candidate = None
        best_goal_world = None
        best_path_length = -1.0
        for candidate in candidates:
            path = astar(inflated_grid, robot_cell, candidate)
            if not path:
                self.mark_goal_region_blocked(candidate)
                continue
            path_length = self.path_length_cells(path)
            if path_length <= best_path_length:
                continue
            best_candidate = candidate
            best_goal_world = self.pooled_cell_to_world(candidate[0], candidate[1])
            best_path_length = path_length

        if best_candidate is None:
            return None
        return best_candidate, best_goal_world, best_path_length

    def find_frontier_clusters(self, grid):
        rows, cols = grid.shape
        frontier_mask = np.zeros((rows, cols), dtype=bool)
        for row in range(rows):
            for col in range(cols):
                if grid[row, col] != 0:
                    continue
                for d_row, d_col in CARDINAL_NEIGHBORS:
                    nr, nc = row + d_row, col + d_col
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 2:
                        frontier_mask[row, col] = True
                        break

        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters = []
        for row in range(rows):
            for col in range(cols):
                if not frontier_mask[row, col] or visited[row, col]:
                    continue
                queue = deque([(row, col)])
                visited[row, col] = True
                cluster = []
                while queue:
                    cr, cc = queue.popleft()
                    cluster.append((cr, cc))
                    for d_row, d_col in EIGHT_CONNECTED_NEIGHBORS:
                        nr, nc = cr + d_row, cc + d_col
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if frontier_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                if len(cluster) >= MIN_FRONTIER_SIZE:
                    clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters[:MAX_FRONTIER_CLUSTERS]

    def prune_blocked_goals(self):
        now = time.time()
        self.blocked_goals = {
            cell: blocked_at
            for cell, blocked_at in self.blocked_goals.items()
            if now - blocked_at < BLOCKED_GOAL_COOLDOWN
        }

    def mark_goal_region_blocked(self, goal_cell, radius=BLOCKED_GOAL_RADIUS):
        row, col = goal_cell
        blocked_at = time.time()
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                self.blocked_goals[(row + d_row, col + d_col)] = blocked_at

    def cluster_centroid(self, cluster):
        cr = sum(cell[0] for cell in cluster) / len(cluster)
        cc = sum(cell[1] for cell in cluster) / len(cluster)
        return min(cluster, key=lambda cell: abs(cell[0] - cr) + abs(cell[1] - cc))

    def path_length_cells(self, path):
        if len(path) < 2:
            return 0.0
        return sum(heuristic(a, b) for a, b in zip(path, path[1:]))

    def clearance_score(self, raw_grid, row, col):
        rows, cols = raw_grid.shape
        score = 0
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                nr, nc = row + d_row, col + d_col
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                value = raw_grid[nr, nc]
                score += 1 if value == 0 else (-3 if value == 1 else -1)
        return score

    def select_standoff_goal(self, frontier_cell, robot_cell, raw_grid, inflated_grid):
        fr, fc = frontier_cell
        rr, rc = robot_cell
        rows, cols = raw_grid.shape
        best_goal = None
        best_score = None
        for d_row in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS + 1):
            for d_col in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS + 1):
                gr, gc = fr + d_row, fc + d_col
                if not (0 <= gr < rows and 0 <= gc < cols):
                    continue
                if inflated_grid[gr, gc] != 0:
                    continue
                score = (
                    abs(gr - fr) + abs(gc - fc),
                    -self.clearance_score(raw_grid, gr, gc),
                    abs(gr - rr) + abs(gc - rc),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_goal = (gr, gc)
        return best_goal

    def candidate_frontiers(self, clusters, robot_cell, inflated_grid):
        candidates = []
        for cluster in clusters:
            valid = [
                cell for cell in cluster
                if cell not in self.blocked_goals and inflated_grid[cell] == 0
            ]
            if not valid:
                continue
            rep = self.cluster_centroid(valid)
            ranked = sorted(
                valid,
                key=lambda cell: (
                    abs(cell[0] - rep[0]) + abs(cell[1] - rep[1]),
                    -heuristic(robot_cell, cell),
                ),
            )
            for cell in ranked[:MAX_FRONTIER_GOALS_PER_CLUSTER]:
                candidates.append(cell)
        return candidates

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def backup_from_obstacle(self):
        now = time.time()
        if now - self.last_backup_time < BLOCKED_RECOVERY_WAIT:
            return

        self.get_logger().warn('Front blocked for too long, backing up to recover.')
        end_time = now + BACKUP_DURATION
        twist = Twist()
        twist.linear.x = BACKUP_SPEED
        while rclpy.ok() and time.time() < end_time:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)
        self.stop_robot()
        self.last_backup_time = time.time()
        self.front_blocked_since = None

    def choose_goal(self):
        if self.map_data.size == 0:
            return None

        self.prune_blocked_goals()
        pooled_occ = self.pool_map(self.map_data)
        raw_grid = self.classify_grid(pooled_occ)
        inflated_grid = self.inflate_walls(raw_grid)

        if int(np.count_nonzero(raw_grid == 0)) < MIN_FREE_CELLS_TO_PLAN:
            return None

        try:
            robot_x, robot_y, _ = self.get_pose()
        except TransformException:
            return None

        robot_cell = self.world_to_pooled_cell(robot_x, robot_y, raw_grid.shape)
        tracking_cell = self.nearest_free_to_robot(raw_grid, robot_cell)
        if tracking_cell is None:
            self.log_status('No goal: robot is not near any mapped free cell yet.')
            return None
        robot_cell = self.nearest_free_to_robot(
            inflated_grid, tracking_cell, allow_traverse_occupied=True)
        if robot_cell is None:
            self.log_status('No goal: unable to snap robot pose to a safe planning cell.')
            return None

        self.mark_visited(tracking_cell, raw_grid.shape)
        clusters = self.find_frontier_clusters(raw_grid)

        if clusters:
            if self.coverage_mode:
                self.get_logger().info('New frontiers found, returning to frontier mode.')
                self.coverage_mode = False

            candidates = self.candidate_frontiers(clusters, robot_cell, inflated_grid)
            best_goal_cell = None
            best_goal_world = None
            best_path_length = float('inf')

            for frontier_cell in candidates:
                goal_cell = self.select_standoff_goal(
                    frontier_cell, robot_cell, raw_grid, inflated_grid)
                if goal_cell is None:
                    self.mark_goal_region_blocked(frontier_cell)
                    continue
                path_cells = astar(inflated_grid, robot_cell, goal_cell)
                if not path_cells:
                    self.mark_goal_region_blocked(goal_cell)
                    continue
                path_length = self.path_length_cells(path_cells)
                if path_length >= best_path_length:
                    continue
                best_goal_cell = goal_cell
                best_goal_world = self.pooled_cell_to_world(goal_cell[0], goal_cell[1])
                best_path_length = path_length

            if best_goal_cell is not None:
                return best_goal_cell, best_goal_world, best_path_length

        if not self.coverage_mode:
            unvisited_count = len(self.get_unvisited_free_cells(inflated_grid))
            self.get_logger().info(
                f'No frontiers left. Switching to coverage mode. Unvisited cells: {unvisited_count}')
            self.coverage_mode = True

        result = self.choose_coverage_goal(robot_cell, inflated_grid)
        if result is None:
            unvisited_count = len(self.get_unvisited_free_cells(inflated_grid))
            if unvisited_count < MIN_UNVISITED_TO_CONTINUE:
                self.get_logger().info(
                    f'Coverage complete. Unvisited cells remaining: {unvisited_count}')
            else:
                self.log_status(
                    f'Coverage: no reachable unvisited cells ({unvisited_count} remain)')
        return result

    def send_nav_goal(self, goal_cell, goal_world, path_length):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server not available.')
            return

        goal_x, goal_y = goal_world
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.orientation.w = 1.0

        self.current_goal_cell = goal_cell
        self.goal_in_flight = True

        send_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(self.goal_response_callback)

        mode = 'coverage' if self.coverage_mode else 'frontier'
        self.get_logger().info(
            f'[{mode}] Goal row={goal_cell[0]} col={goal_cell[1]} '
            f'x={goal_x:.2f} y={goal_y:.2f} path={path_length:.1f} cells')

    def goal_response_callback(self, future):
        self.goal_in_flight = False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected goal.')
            if self.current_goal_cell is not None:
                self.mark_goal_region_blocked(self.current_goal_cell)
            self.current_goal_cell = None
            self.goal_handle = None
            return
        if self.navigation_paused or self.cancel_goal_on_accept:
            self.get_logger().info('Cancelling accepted Nav2 goal because docking is active.')
            goal_handle.cancel_goal_async()
            self.goal_handle = None
            self.current_goal_cell = None
            self.front_blocked_since = None
            return
        self.goal_handle = goal_handle
        self.front_blocked_since = None
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        status = result.status
        if status == 4:
            self.get_logger().info('Nav2 goal reached.')
        elif self.current_goal_cell is not None:
            self.mark_goal_region_blocked(self.current_goal_cell)
            self.get_logger().warn(f'Goal ended status={status}, blocking region.')
        self.goal_handle = None
        self.current_goal_cell = None
        self.goal_in_flight = False
        self.last_progress_pose = None
        self.last_progress_time = time.time()
        self.front_blocked_since = None

    def feedback_callback(self, _feedback_msg):
        try:
            robot_x, robot_y, _ = self.get_pose()
        except TransformException:
            return
        if self.last_progress_pose is None:
            self.last_progress_pose = (robot_x, robot_y)
            self.last_progress_time = time.time()
            return
        progress = math.hypot(
            robot_x - self.last_progress_pose[0],
            robot_y - self.last_progress_pose[1],
        )
        if progress >= MIN_PROGRESS_DISTANCE:
            self.last_progress_pose = (robot_x, robot_y)
            self.last_progress_time = time.time()
            self.front_blocked_since = None

    def cancel_active_goal(self, reason, block_current_goal=True):
        if self.goal_handle is None:
            self.goal_in_flight = False
            return
        self.get_logger().warn(f'Cancelling goal: {reason}')
        if block_current_goal and self.current_goal_cell is not None:
            self.mark_goal_region_blocked(self.current_goal_cell)
        self.goal_handle.cancel_goal_async()
        self.goal_handle = None
        self.current_goal_cell = None
        self.goal_in_flight = False
        self.front_blocked_since = time.time()

    def log_status(self, message):
        now = time.time()
        if now - self.last_status_log_time >= 1.0:
            self.get_logger().info(message)
            self.last_status_log_time = now

    def exploration_step(self):
        if not self.system_ready:
            return
        if self.navigation_stopped_forever or self.navigation_paused:
            self.stop_robot()
            return

        front = self.front_clearance()
        if front is not None and front < FRONT_STOP_DISTANCE:
            if self.front_blocked_since is None:
                self.front_blocked_since = time.time()
            if self.goal_handle is not None:
                self.cancel_active_goal(f'obstacle too close ({front:.2f}m)')
            else:
                blocked_time = time.time() - self.front_blocked_since
                if blocked_time >= BLOCKED_RECOVERY_WAIT:
                    self.backup_from_obstacle()
                else:
                    self.log_status(
                        f'Waiting for clearance: {front:.2f}m ({blocked_time:.1f}s)')
            return
        self.front_blocked_since = None

        if self.goal_in_flight:
            return
        if self.goal_handle is not None:
            if time.time() - self.last_progress_time > PROGRESS_TIMEOUT:
                self.cancel_active_goal('no progress')
            return

        chosen_goal = self.choose_goal()
        if chosen_goal is None:
            self.log_status('No goal available.')
            return

        goal_cell, goal_world, path_length = chosen_goal
        self.send_nav_goal(goal_cell, goal_world, path_length)


class ArucoDockNode(Node):
    def __init__(self):
        super().__init__('cde2310_aruco_dock')

        self.bridge = CvBridge()

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
        self.final_distance = float(
            self.declare_parameter('final_distance', 0.2).value)
        self.linear_gain = float(self.declare_parameter('linear_gain', 4.0).value)
        self.angular_gain_x = float(self.declare_parameter('angular_gain_x', 7.0).value)
        self.angular_gain_yaw = float(self.declare_parameter('angular_gain_yaw', 0.5).value)
        self.max_linear_speed = float(
            self.declare_parameter('max_linear_speed', 0.20).value)
        self.max_angular_speed = float(
            self.declare_parameter('max_angular_speed', 0.9).value)
        self.search_angular_speed = float(
            self.declare_parameter('search_angular_speed', 0.35).value)
        self.loss_timeout = float(
            self.declare_parameter('loss_timeout', 0.75).value)
        self.post_trigger_wait = float(
            self.declare_parameter('post_trigger_wait', 25.0).value)
        self.align_x_threshold = float(
            self.declare_parameter('align_x_threshold', 0.05).value)
        self.align_yaw_threshold = float(
            self.declare_parameter('align_yaw_threshold', 0.08).value)
        self.distance_threshold = float(
            self.declare_parameter('distance_threshold', 0.05).value)
        self.show_debug_window = bool(
            self.declare_parameter('show_debug_window', True).value)

        camera_matrix_default = [
            1000.0, 0.0, 640.0,
            0.0, 1000.0, 360.0,
            0.0, 0.0, 1.0,
        ]
        dist_coeffs_default = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_matrix = np.array(
            self.declare_parameter('camera_matrix', camera_matrix_default).value,
            dtype=float).reshape((3, 3))
        self.dist_coeffs = np.array(
            self.declare_parameter('dist_coeffs', dist_coeffs_default).value,
            dtype=float).reshape((-1, 1))

        dictionary_name = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.detected_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.objA_trigger_pub = self.create_publisher(Bool, '/trigger_objA', 10)
        self.objB_trigger_pub = self.create_publisher(Bool, '/trigger_objB', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10)

        self.nav_node = None
        self.mode = MODE_SEARCH
        self.locked_id = None
        self.last_seen_time = 0.0
        self.search_direction = 1.0
        self.wait_started_at = 0.0
        self.completed_markers = set()

        self.create_timer(0.1, self.watchdog_callback)
        self.get_logger().info('Aruco dock node started.')

    def set_nav_node(self, nav_node):
        self.nav_node = nav_node

    def image_callback(self, msg):
        if self.mode == MODE_FINISHED:
            self.publish_detected(False)
            self.stop_robot()
            return
        if self.mode == MODE_WAIT_AFTER_TRIGGER:
            self.publish_detected(False)
            if self.show_debug_window:
                try:
                    frame = self.bridge.compressed_imgmsg_to_cv2(
                        msg, desired_encoding='bgr8')
                    cv2.imshow('CDE2310_final_run', frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to decode image: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            self.handle_marker_loss(frame)
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        ids_flat = ids.flatten().astype(int)
        target_index = self.select_target_marker(ids_flat, tvecs)
        if target_index is None:
            self.handle_marker_loss(frame)
            return

        marker_id = int(ids_flat[target_index])
        tvec = tvecs[target_index][0]
        rvec = rvecs[target_index][0]

        if self.mode == MODE_SEARCH:
            self.start_docking(marker_id)

        if self.locked_id != marker_id:
            self.locked_id = marker_id
            self.get_logger().info(f'Locked on ArUco marker ID {marker_id}')

        self.last_seen_time = time.time()
        self.search_direction = -1.0 if float(tvec[0]) < 0.0 else 1.0
        self.publish_detected(True)
        self.visual_servo_to_marker(tvec, rvec)

        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow('CDE2310_final_run', frame)
            cv2.waitKey(1)

    def start_docking(self, marker_id):
        if marker_id in self.completed_markers:
            return
        self.mode = MODE_DOCKING
        self.locked_id = marker_id
        self.last_seen_time = 0.0
        if self.nav_node is not None:
            self.nav_node.pause_navigation(
                reason=f'ArUco marker {marker_id} detected for docking')
        self.get_logger().info(f'Starting docking on marker ID {marker_id}')

    def marker_heading_error(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        marker_normal = rotation_matrix[:, 2]
        return math.atan2(
            float(marker_normal[0]),
            abs(float(marker_normal[2])) + 1e-6,
        )

    def visual_servo_to_marker(self, tvec, rvec):
        if self.mode != MODE_DOCKING:
            return

        x_error = float(tvec[0])
        distance_error = float(tvec[2] - self.final_distance)
        yaw_error = self.marker_heading_error(rvec)

        aligned = (
            abs(x_error) <= self.align_x_threshold
            and abs(yaw_error) <= self.align_yaw_threshold
            and abs(distance_error) <= self.distance_threshold
        )

        if aligned:
            self.handle_successful_dock()
            return

        cmd = Twist()
        cmd.linear.x = self.linear_gain * distance_error
        cmd.angular.z = -(
            self.angular_gain_x * x_error
            + self.angular_gain_yaw * yaw_error
        )
        if abs(x_error) > 0.03 or abs(yaw_error) > 0.12:
            cmd.linear.x = min(cmd.linear.x, 0.04)

        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed), -self.max_linear_speed)
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        self.cmd_vel_pub.publish(cmd)

    def handle_successful_dock(self):
        marker_id = self.locked_id
        if marker_id is None or marker_id in self.completed_markers:
            return

        trigger_msg = Bool()
        trigger_msg.data = True

        if marker_id == self.primary_marker_id:
            self.objA_trigger_pub.publish(trigger_msg)
            self.get_logger().info('Docked on marker A. Published /trigger_objA.')
        elif marker_id == self.secondary_marker_id:
            self.objB_trigger_pub.publish(trigger_msg)
            self.get_logger().info('Docked on marker B. Published /trigger_objB.')
        else:
            self.get_logger().warn(f'Docked on unexpected marker ID {marker_id}, no trigger sent.')

        self.completed_markers.add(marker_id)
        self.mode = MODE_WAIT_AFTER_TRIGGER
        self.wait_started_at = time.time()
        self.stop_robot()

    def handle_marker_loss(self, frame):
        self.publish_detected(False)

        if self.mode != MODE_DOCKING:
            if self.show_debug_window:
                cv2.imshow('CDE2310_final_run', frame)
                cv2.waitKey(1)
            return

        if self.last_seen_time <= 0.0:
            self.publish_search_rotation()
        else:
            elapsed = time.time() - self.last_seen_time
            if elapsed > self.loss_timeout:
                self.abort_docking('marker lost during docking')
            else:
                self.publish_search_rotation()

        if self.show_debug_window:
            cv2.imshow('CDE2310_final_run', frame)
            cv2.waitKey(1)

    def publish_detected(self, state):
        msg = Bool()
        msg.data = bool(state)
        self.detected_pub.publish(msg)

    def publish_search_rotation(self):
        cmd = Twist()
        cmd.angular.z = self.search_direction * self.search_angular_speed
        self.cmd_vel_pub.publish(cmd)

    def abort_docking(self, reason):
        if self.mode != MODE_DOCKING:
            return
        self.get_logger().warn(f'Aborting docking: {reason}')
        self.stop_robot()
        self.mode = MODE_SEARCH
        self.locked_id = None
        self.last_seen_time = 0.0
        if self.nav_node is not None:
            self.nav_node.resume_navigation()

    def resume_search_or_finish(self):
        self.stop_robot()
        if self.primary_marker_id in self.completed_markers and self.secondary_marker_id in self.completed_markers:
            self.mode = MODE_FINISHED
            self.locked_id = None
            self.get_logger().info('Both markers completed. Finishing run.')
            if self.nav_node is not None:
                self.nav_node.stop_forever()
            return

        self.mode = MODE_SEARCH
        self.locked_id = None
        self.last_seen_time = 0.0
        self.wait_started_at = 0.0
        self.get_logger().info('Post-trigger wait complete. Resuming navigation.')
        if self.nav_node is not None:
            self.nav_node.resume_navigation()

    def watchdog_callback(self):
        if self.mode == MODE_DOCKING and self.last_seen_time > 0.0:
            if time.time() - self.last_seen_time > self.loss_timeout:
                self.abort_docking('watchdog timeout while docking')
                return

        if self.mode == MODE_WAIT_AFTER_TRIGGER and self.wait_started_at > 0.0:
            if time.time() - self.wait_started_at >= self.post_trigger_wait:
                self.resume_search_or_finish()

        if self.mode == MODE_FINISHED:
            self.stop_robot()

    def select_target_marker(self, ids_flat, tvecs):
        available = [
            (idx, int(marker_id))
            for idx, marker_id in enumerate(ids_flat)
            if int(marker_id) in (self.primary_marker_id, self.secondary_marker_id)
            and int(marker_id) not in self.completed_markers
        ]
        if not available:
            return None

        if self.mode == MODE_DOCKING and self.locked_id is not None:
            locked_matches = [
                idx for idx, marker_id in available
                if marker_id == self.locked_id
            ]
            if locked_matches:
                return min(locked_matches, key=lambda i: float(tvecs[i][0][2]))
            return None

        primary_matches = [
            idx for idx, marker_id in available
            if marker_id == self.primary_marker_id
        ]
        secondary_matches = [
            idx for idx, marker_id in available
            if marker_id == self.secondary_marker_id
        ]
        if primary_matches:
            return min(primary_matches, key=lambda i: float(tvecs[i][0][2]))
        if secondary_matches:
            return min(secondary_matches, key=lambda i: float(tvecs[i][0][2]))
        return None

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    nav_node = AutoNavNode()
    dock_node = ArucoDockNode()
    dock_node.set_nav_node(nav_node)

    executor = MultiThreadedExecutor()
    executor.add_node(nav_node)
    executor.add_node(dock_node)

    try:
        if nav_node.wait_for_system_ready():
            executor.spin()
    except KeyboardInterrupt:
        nav_node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        dock_node.stop_robot()
        nav_node.stop_robot()
        executor.shutdown()
        dock_node.destroy_node()
        nav_node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
