#!/usr/bin/env python3

import heapq
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import tf2_ros
from tf2_ros import TransformException


# ── existing constants ────────────────────────────────────────────────────
UNKNOWN_VALUE = -1
WALL_THRESHOLD = 60
POOL_SIZE = 2
WALL_INFLATION_RADIUS = 1
FRONTIER_GOAL_INFLATION_RADIUS = 0
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
SEARCH_ROTATION_SPEED = 0.5
SEARCH_ROTATION_TARGET = 2.0 * math.pi

# ── coverage constants ────────────────────────────────────────────────────
# how many pooled cells around the robot count as "visited" each step
VISITED_MARK_RADIUS = 2
# stop coverage when fewer than this many unvisited free cells remain
MIN_UNVISITED_TO_CONTINUE = 10
# coverage waypoints must be at least this many cells apart
COVERAGE_WAYPOINT_SPACING = 4

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
            OccupancyGrid, 'map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.stop_nav_sub = self.create_subscription(
            Bool, '/stop_nav', self.stop_nav_callback, 10)
        self.aruco_detected_sub = self.create_subscription(
            Bool, '/aruco_detected', self.aruco_detected_callback, 10)

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
        self.stop_nav_active = False
        self.cancel_goal_on_accept = False
        self.aruco_detected = False
        self.search_rotation_active = False
        self.search_rotation_accumulated = 0.0
        self.search_rotation_last_yaw = None

        # ── coverage tracking ─────────────────────────────────────────────
        # set of pooled (row, col) cells the robot has physically been near
        self.visited_cells = set()
        # True once all frontiers are gone and we switch to coverage mode
        self.coverage_mode = False
        self.get_logger().info('AutoNav started.')

        self.create_timer(PLANNER_PERIOD, self.exploration_step)

    # ── callbacks ─────────────────────────────────────────────────────────

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

    def stop_nav_callback(self, msg):
        stop_active = bool(msg.data)
        if stop_active == self.stop_nav_active:
            return

        self.stop_nav_active = stop_active
        if self.stop_nav_active:
            self.finish_search_rotation()
            self.get_logger().info(
                'Received stop_nav=True. Pausing exploration and cancelling Nav2 goals.')
            self.cancel_goal_on_accept = True
            self.cancel_active_goal(
                'stop_nav asserted by aruco_pid_dock', block_current_goal=False)
            self.stop_robot()
        else:
            self.get_logger().info(
                'Received stop_nav=False. Resuming exploration.')
            self.cancel_goal_on_accept = False

    def aruco_detected_callback(self, msg):
        self.aruco_detected = bool(msg.data)
        if self.aruco_detected and self.search_rotation_active:
            self.get_logger().info('ArUco detected during coverage search spin.')
            self.finish_search_rotation()

    # ── startup ───────────────────────────────────────────────────────────

    def wait_for_system_ready(self, timeout_sec=STARTUP_TIMEOUT):
        self.get_logger().info(
            'Waiting for /map, TF map->base_link, and Nav2...')
        start_time = time.time()
        while rclpy.ok() and time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            map_ready = (self.map_data.size != 0
                         and self.map_resolution is not None
                         and self.map_origin is not None)
            scan_ready = (self.scan_ranges.size != 0
                          and self.scan_angles.size != 0)
            tf_ready = self.tf_buffer.can_transform(
                'map', 'base_link', rclpy.time.Time())
            nav_ready = self.nav_client.wait_for_server(timeout_sec=0.0)
            if map_ready and scan_ready and tf_ready and nav_ready:
                self.system_ready = True
                self.get_logger().info('System ready.')
                return True
        self.get_logger().warn('System not ready in time.')
        return False

    # ── pose ──────────────────────────────────────────────────────────────

    def get_pose(self):
        transform = self.tf_buffer.lookup_transform(
            'map', 'base_link', rclpy.time.Time())
        _, _, yaw = euler_from_quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w)
        return (transform.transform.translation.x,
                transform.transform.translation.y,
                yaw)

    # ── laser ─────────────────────────────────────────────────────────────

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

    # ── map helpers ───────────────────────────────────────────────────────

    def pool_map(self, occ_grid):
        height, width = occ_grid.shape
        pad_h = (POOL_SIZE - height % POOL_SIZE) % POOL_SIZE
        pad_w = (POOL_SIZE - width % POOL_SIZE) % POOL_SIZE
        if pad_h > 0 or pad_w > 0:
            occ_grid = np.pad(occ_grid, ((0, pad_h), (0, pad_w)),
                              mode='constant', constant_values=UNKNOWN_VALUE)
        pooled_h = occ_grid.shape[0] // POOL_SIZE
        pooled_w = occ_grid.shape[1] // POOL_SIZE
        pooled = np.full((pooled_h, pooled_w), UNKNOWN_VALUE, dtype=np.int16)
        for row in range(pooled_h):
            for col in range(pooled_w):
                block = occ_grid[row*POOL_SIZE:(row+1)*POOL_SIZE,
                                 col*POOL_SIZE:(col+1)*POOL_SIZE]
                pooled[row, col] = int(np.max(block))
        return pooled

    def classify_grid(self, pooled_occ):
        grid = np.full_like(pooled_occ, 2, dtype=np.uint8)
        grid[(pooled_occ >= 0) & (pooled_occ < WALL_THRESHOLD)] = 0
        grid[pooled_occ >= WALL_THRESHOLD] = 1
        return grid

    def inflate_walls_with_radius(self, grid, radius):
        inflated = grid.copy()
        rows, cols = grid.shape
        for row, col in np.argwhere(grid == 1):
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    nr, nc = row+d_row, col+d_col
                    if 0 <= nr < rows and 0 <= nc < cols and inflated[nr, nc] == 0:
                        inflated[nr, nc] = 1
        return inflated

    def inflate_walls(self, grid):
        return self.inflate_walls_with_radius(grid, WALL_INFLATION_RADIUS)

    def world_to_pooled_cell(self, world_x, world_y, pooled_shape):
        map_x = (world_x - self.map_origin.x) / self.map_resolution
        map_y = (world_y - self.map_origin.y) / self.map_resolution
        pooled_col = int(math.floor(map_x / POOL_SIZE))
        pooled_row = int(math.floor(map_y / POOL_SIZE))
        pooled_col = min(max(pooled_col, 0), pooled_shape[1]-1)
        pooled_row = min(max(pooled_row, 0), pooled_shape[0]-1)
        return pooled_row, pooled_col

    def pooled_cell_to_world(self, row, col):
        map_x = (col*POOL_SIZE + POOL_SIZE/2.0)*self.map_resolution + self.map_origin.x
        map_y = (row*POOL_SIZE + POOL_SIZE/2.0)*self.map_resolution + self.map_origin.y
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
                next_cell = (row+d_row, col+d_col)
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

    # ── visited cells tracking ────────────────────────────────────────────

    def mark_visited(self, robot_cell, grid_shape):
        """
        Mark all free cells within VISITED_MARK_RADIUS of the robot as visited.
        Think of it like the robot leaving footprints — any cell it gets close
        to counts as visited.
        """
        row, col = robot_cell
        rows, cols = grid_shape
        for d_row in range(-VISITED_MARK_RADIUS, VISITED_MARK_RADIUS+1):
            for d_col in range(-VISITED_MARK_RADIUS, VISITED_MARK_RADIUS+1):
                nr, nc = row+d_row, col+d_col
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.visited_cells.add((nr, nc))

    def get_unvisited_free_cells(self, inflated_grid):
        """
        Returns all free cells that the robot hasn't been near yet.
        These become coverage targets once frontiers are exhausted.
        """
        free_cells = set(zip(*np.where(inflated_grid == 0)))
        return free_cells - self.visited_cells

    def choose_coverage_goal(self, robot_cell, inflated_grid):
        """
        Pick the farthest reachable unvisited free cell that is at least
        COVERAGE_WAYPOINT_SPACING away, so coverage pushes outward instead
        of only mopping up nearby leftovers first.
        """
        unvisited = self.get_unvisited_free_cells(inflated_grid)
        if len(unvisited) < MIN_UNVISITED_TO_CONTINUE:
            return None

        robot_row, robot_col = robot_cell
        # filter: must be far enough away to be worth going to
        candidates = [
            cell for cell in unvisited
            if math.hypot(cell[0]-robot_row, cell[1]-robot_col) >= COVERAGE_WAYPOINT_SPACING
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

    # ── frontier helpers ──────────────────────────────────────────────────

    def find_frontier_clusters(self, grid):
        rows, cols = grid.shape
        frontier_mask = np.zeros((rows, cols), dtype=bool)
        for row in range(rows):
            for col in range(cols):
                if grid[row, col] != 0:
                    continue
                for d_row, d_col in CARDINAL_NEIGHBORS:
                    nr, nc = row+d_row, col+d_col
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
                        nr, nc = cr+d_row, cc+d_col
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
            cell: t for cell, t in self.blocked_goals.items()
            if now - t < BLOCKED_GOAL_COOLDOWN
        }

    def mark_goal_region_blocked(self, goal_cell, radius=BLOCKED_GOAL_RADIUS):
        row, col = goal_cell
        blocked_at = time.time()
        for d_row in range(-radius, radius+1):
            for d_col in range(-radius, radius+1):
                self.blocked_goals[(row+d_row, col+d_col)] = blocked_at

    def cluster_centroid(self, cluster):
        cr = sum(c[0] for c in cluster) / len(cluster)
        cc = sum(c[1] for c in cluster) / len(cluster)
        return min(cluster, key=lambda c: abs(c[0]-cr) + abs(c[1]-cc))

    def path_length_cells(self, path):
        if len(path) < 2:
            return 0.0
        return sum(heuristic(a, b) for a, b in zip(path, path[1:]))

    def clearance_score(self, raw_grid, row, col):
        rows, cols = raw_grid.shape
        score = 0
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                nr, nc = row+d_row, col+d_col
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                v = raw_grid[nr, nc]
                score += 1 if v == 0 else (-3 if v == 1 else -1)
        return score

    def select_standoff_goal(self, frontier_cell, robot_cell, raw_grid, inflated_grid):
        fr, fc = frontier_cell
        rr, rc = robot_cell
        rows, cols = raw_grid.shape
        best_goal = None
        best_score = None
        for d_row in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS+1):
            for d_col in range(-GOAL_SEARCH_RADIUS, GOAL_SEARCH_RADIUS+1):
                gr, gc = fr+d_row, fc+d_col
                if not (0 <= gr < rows and 0 <= gc < cols):
                    continue
                if inflated_grid[gr, gc] != 0:
                    continue
                score = (
                    abs(gr-fr)+abs(gc-fc),
                    -self.clearance_score(raw_grid, gr, gc),
                    abs(gr-rr)+abs(gc-rc),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_goal = (gr, gc)
        return best_goal

    def candidate_frontiers(self, clusters, robot_cell, inflated_grid):
        candidates = []
        for cluster in clusters:
            valid = [c for c in cluster
                     if c not in self.blocked_goals and inflated_grid[c] == 0]
            if not valid:
                continue
            rep = self.cluster_centroid(valid)
            ranked = sorted(valid, key=lambda c: (
                abs(c[0]-rep[0])+abs(c[1]-rep[1]),
                -heuristic(robot_cell, c),
            ))
            for cell in ranked[:MAX_FRONTIER_GOALS_PER_CLUSTER]:
                candidates.append(cell)
        return candidates

    def choose_frontier_goal(self, frontier_cell, robot_cell, raw_grid, frontier_inflated_grid):
        goal_cell = self.select_standoff_goal(
            frontier_cell, robot_cell, raw_grid, frontier_inflated_grid)
        if goal_cell is not None:
            path_cells = astar(frontier_inflated_grid, robot_cell, goal_cell)
            if path_cells:
                return goal_cell, path_cells

        # Fallback: allow narrow passages by using the raw free-space grid for
        # the frontier approach when the stricter frontier map finds no route.
        goal_cell = self.select_standoff_goal(
            frontier_cell, robot_cell, raw_grid, raw_grid)
        if goal_cell is not None:
            path_cells = astar(raw_grid, robot_cell, goal_cell)
            if path_cells:
                self.get_logger().info(
                    f'Using narrow-gap frontier fallback for cell {frontier_cell}.')
                return goal_cell, path_cells

        # Final fallback: if a safe standoff cell cannot be found, try the
        # frontier cell itself on the raw grid.
        if raw_grid[frontier_cell] == 0:
            path_cells = astar(raw_grid, robot_cell, frontier_cell)
            if path_cells:
                self.get_logger().info(
                    f'Using direct frontier fallback for cell {frontier_cell}.')
                return frontier_cell, path_cells

        return None, None

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def start_search_rotation(self):
        if self.search_rotation_active or self.stop_nav_active:
            return

        try:
            _, _, yaw = self.get_pose()
        except TransformException:
            self.get_logger().warn(
                'Skipping coverage search spin because robot pose is unavailable.')
            return

        self.search_rotation_active = True
        self.search_rotation_accumulated = 0.0
        self.search_rotation_last_yaw = yaw
        self.get_logger().info(
            'Coverage goal reached. Rotating 360 degrees to search for ArUco.')

    def finish_search_rotation(self):
        if not self.search_rotation_active:
            return
        self.search_rotation_active = False
        self.search_rotation_accumulated = 0.0
        self.search_rotation_last_yaw = None
        self.stop_robot()

    def update_search_rotation(self):
        if not self.search_rotation_active:
            return False

        if self.aruco_detected:
            self.finish_search_rotation()
            return False

        try:
            _, _, yaw = self.get_pose()
        except TransformException:
            self.get_logger().warn(
                'Stopping coverage search spin because robot pose is unavailable.')
            self.finish_search_rotation()
            return False

        if self.search_rotation_last_yaw is not None:
            yaw_delta = self.normalize_angle(yaw - self.search_rotation_last_yaw)
            self.search_rotation_accumulated += abs(yaw_delta)
        self.search_rotation_last_yaw = yaw

        if self.search_rotation_accumulated >= SEARCH_ROTATION_TARGET:
            self.get_logger().info('Completed 360-degree coverage search spin.')
            self.finish_search_rotation()
            return False

        twist = Twist()
        twist.angular.z = SEARCH_ROTATION_SPEED
        self.cmd_vel_pub.publish(twist)
        return True

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

    # ── goal selection ────────────────────────────────────────────────────

    def choose_goal(self):
        """
        First try frontier exploration.
        If no frontiers remain, switch to coverage mode and
        pick the nearest unvisited free cell instead.
        """
        if self.map_data.size == 0:
            return None

        self.prune_blocked_goals()
        pooled_occ  = self.pool_map(self.map_data)
        raw_grid    = self.classify_grid(pooled_occ)
        inflated_grid = self.inflate_walls(raw_grid)
        frontier_inflated_grid = self.inflate_walls_with_radius(
            raw_grid, FRONTIER_GOAL_INFLATION_RADIUS)

        free_cells = int(np.count_nonzero(raw_grid == 0))
        if free_cells < MIN_FREE_CELLS_TO_PLAN:
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
        frontier_robot_cell = self.nearest_free_to_robot(
            frontier_inflated_grid, tracking_cell, allow_traverse_occupied=True)
        if frontier_robot_cell is None:
            frontier_robot_cell = tracking_cell

        # mark where the robot currently is as visited
        self.mark_visited(tracking_cell, raw_grid.shape)

        # ── Phase 1: frontier exploration ─────────────────────────────────
        clusters = self.find_frontier_clusters(raw_grid)

        if clusters:
            if self.coverage_mode:
                self.get_logger().info('New frontiers found — back to frontier mode.')
                self.coverage_mode = False

            candidates = self.candidate_frontiers(
                clusters, frontier_robot_cell, frontier_inflated_grid)
            if not candidates:
                candidates = self.candidate_frontiers(
                    clusters, frontier_robot_cell, raw_grid)
            best_goal_cell = None
            best_goal_world = None
            best_path_length = float('inf')

            for frontier_cell in candidates:
                goal_cell, path_cells = self.choose_frontier_goal(
                    frontier_cell, frontier_robot_cell, raw_grid, frontier_inflated_grid)
                if goal_cell is None or path_cells is None:
                    self.mark_goal_region_blocked(frontier_cell)
                    continue
                path_length = self.path_length_cells(path_cells)
                if path_length >= best_path_length:
                    continue
                best_goal_cell  = goal_cell
                best_goal_world = self.pooled_cell_to_world(goal_cell[0], goal_cell[1])
                best_path_length = path_length

            if best_goal_cell is not None:
                return best_goal_cell, best_goal_world, best_path_length

        # ── Phase 2: coverage mode ────────────────────────────────────────
        # No frontiers left — drive through unvisited areas
        if not self.coverage_mode:
            unvisited_count = len(self.get_unvisited_free_cells(inflated_grid))
            self.get_logger().info(
                f'No frontiers left. Switching to coverage mode. '
                f'Unvisited cells: {unvisited_count}')
            self.coverage_mode = True

        result = self.choose_coverage_goal(robot_cell, inflated_grid)
        if result is None:
            unvisited_count = len(self.get_unvisited_free_cells(inflated_grid))
            if unvisited_count < MIN_UNVISITED_TO_CONTINUE:
                self.get_logger().info(
                    f'Coverage complete! Unvisited cells remaining: {unvisited_count}')
            else:
                self.log_status(f'Coverage: no reachable unvisited cells ({unvisited_count} remain)')
        return result

    # ── nav2 goal sending ─────────────────────────────────────────────────

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
        if self.stop_nav_active or self.cancel_goal_on_accept:
            self.get_logger().info(
                'Cancelling accepted Nav2 goal because stop_nav is active.')
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
            if self.coverage_mode:
                self.start_search_rotation()
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
            robot_y - self.last_progress_pose[1])
        if progress >= MIN_PROGRESS_DISTANCE:
            self.last_progress_pose = (robot_x, robot_y)
            self.last_progress_time = time.time()
            self.front_blocked_since = None

    def cancel_active_goal(self, reason, block_current_goal=True):
        if self.goal_handle is None:
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

    # ── main timer ────────────────────────────────────────────────────────

    def exploration_step(self):
        if not self.system_ready:
            return

        if self.stop_nav_active:
            self.stop_robot()
            return

        if self.update_search_rotation():
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
                        f'Waiting for clearance: {front:.2f}m '
                        f'({blocked_time:.1f}s)')
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
