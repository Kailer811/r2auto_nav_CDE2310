#!/usr/bin/env python3

import heapq
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import tf2_ros
from tf2_ros import TransformException


UNKNOWN_VALUE = -1
WALL_THRESHOLD = 60
POOL_SIZE = 2
WALL_INFLATION_RADIUS = 2
MIN_FRONTIER_SIZE = 5
MIN_FREE_CELLS_TO_PLAN = 20
MAX_FRONTIER_GOALS_PER_CLUSTER = 4
MAX_FRONTIER_CLUSTERS = 8
BLOCKED_GOAL_COOLDOWN = 25.0
BLOCKED_GOAL_RADIUS = 2
PROGRESS_TIMEOUT = 15.0
MIN_PROGRESS_DISTANCE = 0.08
PLANNER_PERIOD = 1.0
STARTUP_TIMEOUT = 20.0
GOAL_SEARCH_RADIUS = 3
FRONT_HALF_ANGLE_DEG = 18.0
FRONT_STOP_DISTANCE = 0.24

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
            f_cost = next_g + heuristic(next_cell, goal)
            heapq.heappush(open_heap, (f_cost, next_g, next_cell))

    return []


class AutoNav(Node):
    def __init__(self):
        super().__init__('auto_nav')

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            map_qos,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
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

        self.create_timer(PLANNER_PERIOD, self.exploration_step)
        self.get_logger().info('AutoNav started.')

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.map_data = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

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
        self.get_logger().info('Waiting for /map, /scan, TF map->base_link, and Nav2...')
        start_time = time.time()

        while rclpy.ok() and time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            map_ready = self.map_data.size != 0 and self.map_resolution is not None and self.map_origin is not None
            scan_ready = self.scan_ranges.size != 0 and self.scan_angles.size != 0
            tf_ready = self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time())
            nav_ready = self.nav_client.wait_for_server(timeout_sec=0.0)
            if map_ready and scan_ready and tf_ready and nav_ready:
                self.system_ready = True
                self.get_logger().info('System ready.')
                return True

        self.get_logger().warn('System not ready in time.')
        return False

    def get_pose(self):
        transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        _, _, yaw = euler_from_quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        return transform.transform.translation.x, transform.transform.translation.y, yaw

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
                block = occ_grid[
                    row * POOL_SIZE:(row + 1) * POOL_SIZE,
                    col * POOL_SIZE:(col + 1) * POOL_SIZE,
                ]
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
                    next_row = row + d_row
                    next_col = col + d_col
                    if 0 <= next_row < rows and 0 <= next_col < cols and inflated[next_row, next_col] == 0:
                        inflated[next_row, next_col] = 1

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

    def nearest_free_to_robot(self, grid, robot_cell):
        if grid[robot_cell] == 0:
            return robot_cell

        queue = deque([robot_cell])
        visited = {robot_cell}
        rows, cols = grid.shape

        while queue:
            row, col = queue.popleft()
            for d_row, d_col in CARDINAL_NEIGHBORS:
                next_cell = (row + d_row, col + d_col)
                next_row, next_col = next_cell
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if next_cell in visited or grid[next_cell] == 1:
                    continue
                if grid[next_cell] == 0:
                    return next_cell
                visited.add(next_cell)
                queue.append(next_cell)

        return None

    def find_frontier_clusters(self, grid):
        rows, cols = grid.shape
        frontier_mask = np.zeros((rows, cols), dtype=bool)

        for row in range(rows):
            for col in range(cols):
                if grid[row, col] != 0:
                    continue
                for d_row, d_col in CARDINAL_NEIGHBORS:
                    next_row = row + d_row
                    next_col = col + d_col
                    if 0 <= next_row < rows and 0 <= next_col < cols and grid[next_row, next_col] == 2:
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
                    cur_row, cur_col = queue.popleft()
                    cluster.append((cur_row, cur_col))
                    for d_row, d_col in EIGHT_CONNECTED_NEIGHBORS:
                        next_row = cur_row + d_row
                        next_col = cur_col + d_col
                        if 0 <= next_row < rows and 0 <= next_col < cols:
                            if frontier_mask[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                queue.append((next_row, next_col))

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
        centroid_row = sum(cell[0] for cell in cluster) / len(cluster)
        centroid_col = sum(cell[1] for cell in cluster) / len(cluster)
        return min(
            cluster,
            key=lambda cell: abs(cell[0] - centroid_row) + abs(cell[1] - centroid_col),
        )

    def path_length_cells(self, path):
        if len(path) < 2:
            return 0.0

        total = 0.0
        for current, next_cell in zip(path, path[1:]):
            total += heuristic(current, next_cell)
        return total

    def clearance_score(self, raw_grid, row, col):
        rows, cols = raw_grid.shape
        score = 0

        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                next_row = row + d_row
                next_col = col + d_col
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                value = raw_grid[next_row, next_col]
                if value == 1:
                    score -= 3
                elif value == 2:
                    score -= 1
                else:
                    score += 1

        return score

    def select_standoff_goal(self, frontier_cell, robot_cell, raw_grid, inflated_grid):
        frontier_row, frontier_col = frontier_cell
        robot_row, robot_col = robot_cell
        rows, cols = raw_grid.shape
        best_goal = None
        best_score = None

        for d_row in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS + 1):
            for d_col in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS + 1):
                goal_row = frontier_row + d_row
                goal_col = frontier_col + d_col
                if not (0 <= goal_row < rows and 0 <= goal_col < cols):
                    continue
                if inflated_grid[goal_row, goal_col] != 0:
                    continue

                frontier_distance = abs(goal_row - frontier_row) + abs(goal_col - frontier_col)
                robot_distance = abs(goal_row - robot_row) + abs(goal_col - robot_col)
                clearance = self.clearance_score(raw_grid, goal_row, goal_col)

                score = (
                    frontier_distance,
                    -clearance,
                    robot_distance,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_goal = (goal_row, goal_col)

        return best_goal

    def candidate_frontiers(self, clusters, robot_cell, inflated_grid):
        candidates = []

        for cluster in clusters:
            valid_cells = [
                cell for cell in cluster
                if cell not in self.blocked_goals and inflated_grid[cell] == 0
            ]
            if not valid_cells:
                continue

            representative_cell = self.cluster_centroid(valid_cells)
            ranked_cluster = sorted(
                valid_cells,
                key=lambda cell: (
                    abs(cell[0] - representative_cell[0]) + abs(cell[1] - representative_cell[1]),
                    -heuristic(robot_cell, cell),
                ),
            )

            for cell in ranked_cluster[:MAX_FRONTIER_GOALS_PER_CLUSTER]:
                candidates.append(cell)

        return candidates

    def choose_goal(self):
        if self.map_data.size == 0:
            return None

        self.prune_blocked_goals()
        pooled_occ = self.pool_map(self.map_data)
        raw_grid = self.classify_grid(pooled_occ)
        inflated_grid = self.inflate_walls(raw_grid)

        free_cells = int(np.count_nonzero(raw_grid == 0))
        if free_cells < MIN_FREE_CELLS_TO_PLAN:
            return None

        try:
            robot_x, robot_y, _ = self.get_pose()
        except TransformException:
            return None

        robot_cell = self.world_to_pooled_cell(robot_x, robot_y, raw_grid.shape)
        robot_cell = self.nearest_free_to_robot(inflated_grid, robot_cell)
        if robot_cell is None:
            return None

        clusters = self.find_frontier_clusters(raw_grid)
        if not clusters:
            return None

        candidates = self.candidate_frontiers(clusters, robot_cell, inflated_grid)
        if not candidates:
            return None

        best_goal_cell = None
        best_goal_world = None
        best_path_length = -1.0

        for frontier_cell in candidates:
            goal_cell = self.select_standoff_goal(frontier_cell, robot_cell, raw_grid, inflated_grid)
            if goal_cell is None:
                self.mark_goal_region_blocked(frontier_cell)
                continue

            path_cells = astar(inflated_grid, robot_cell, goal_cell)
            if not path_cells:
                self.mark_goal_region_blocked(goal_cell)
                continue

            path_length = self.path_length_cells(path_cells)
            if path_length <= best_path_length:
                continue

            best_goal_cell = goal_cell
            best_goal_world = self.pooled_cell_to_world(goal_cell[0], goal_cell[1])
            best_path_length = path_length

        if best_goal_cell is None:
            return None

        return best_goal_cell, best_goal_world, best_path_length

    def send_nav_goal(self, goal_cell, goal_world, path_length):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server is not available.')
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

        send_future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(
            f'Sending farthest frontier goal row={goal_cell[0]} col={goal_cell[1]} '
            f'x={goal_x:.2f} y={goal_y:.2f} path_cells={path_length:.1f}'
        )

    def goal_response_callback(self, future):
        self.goal_in_flight = False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal.')
            if self.current_goal_cell is not None:
                self.mark_goal_region_blocked(self.current_goal_cell)
            self.current_goal_cell = None
            self.goal_handle = None
            return

        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        status = result.status
        success_status = 4

        if status == success_status:
            if self.current_goal_cell is not None:
                self.mark_goal_region_blocked(self.current_goal_cell)
            self.get_logger().info('Nav2 goal reached.')
        elif self.current_goal_cell is not None:
            self.mark_goal_region_blocked(self.current_goal_cell)
            self.get_logger().warn(f'Nav2 goal ended with status={status}; blocking nearby frontier region.')

        self.goal_handle = None
        self.current_goal_cell = None
        self.goal_in_flight = False
        self.last_progress_pose = None
        self.last_progress_time = time.time()

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

    def cancel_active_goal(self, reason):
        if self.goal_handle is None:
            return

        self.get_logger().warn(f'Cancelling current goal: {reason}')
        if self.current_goal_cell is not None:
            self.mark_goal_region_blocked(self.current_goal_cell)
        self.goal_handle.cancel_goal_async()
        self.goal_handle = None
        self.current_goal_cell = None
        self.goal_in_flight = False

    def log_status(self, message):
        now = time.time()
        if now - self.last_status_log_time >= 1.0:
            self.get_logger().info(message)
            self.last_status_log_time = now

    def exploration_step(self):
        if not self.system_ready:
            return

        front = self.front_clearance()
        if front is not None and front < FRONT_STOP_DISTANCE:
            if self.goal_handle is not None:
                self.cancel_active_goal(f'front obstacle too close ({front:.2f} m)')
            else:
                self.log_status(f'Waiting for front clearance: {front:.2f} m')
            return

        if self.goal_in_flight:
            return

        if self.goal_handle is not None:
            if time.time() - self.last_progress_time > PROGRESS_TIMEOUT:
                self.cancel_active_goal('robot is not making progress')
            return

        chosen_goal = self.choose_goal()
        if chosen_goal is None:
            self.log_status('No usable frontier goal available yet.')
            return

        goal_cell, goal_world, path_length = chosen_goal
        self.send_nav_goal(goal_cell, goal_world, path_length)


def main(args=None):
    rclpy.init(args=args)
    node = AutoNav()

    try:
        if node.wait_for_system_ready():
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
