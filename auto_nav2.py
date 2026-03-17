#!/usr/bin/env python3

import heapq
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import tf2_ros
from tf2_ros import TransformException


LINEAR_SPEED = 0.10
ANGULAR_SPEED = 0.75
MAX_ANGULAR_CORRECTION = 0.9

STOP_DISTANCE = 0.34
CAUTION_DISTANCE = 0.46
SIDE_CLEARANCE = 0.24
GOAL_REACHED_DISTANCE = 0.18
WAYPOINT_REACHED_DISTANCE = 0.14

WALL_THRESHOLD = 60
UNKNOWN_VALUE = -1
POOL_SIZE = 2
WALL_INFLATION_RADIUS = 1
MIN_FRONTIER_SIZE = 3
MIN_FREE_CELLS_TO_PLAN = 12

FRONT_HALF_ANGLE_DEG = 22.0
SIDE_CENTER_TOLERANCE_DEG = 28.0

STUCK_TIMEOUT = 3.0
MIN_PROGRESS_DISTANCE = 0.08
REPLAN_COOLDOWN = 1.0

STARTUP_TIMEOUT = 20.0
INITIAL_PLAN_TIMEOUT = 30.0
CONTROL_DT = 0.1


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


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def astar(grid, start, goal):
    rows, cols = grid.shape

    def heuristic(cell_a, cell_b):
        return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

    def neighbors(row, col):
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row = row + d_row
            next_col = col + d_col
            if 0 <= next_row < rows and 0 <= next_col < cols:
                yield next_row, next_col

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    if grid[start] != 0 or grid[goal] != 0:
        return []

    open_heap = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
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
        for next_cell in neighbors(row, col):
            if grid[next_cell] != 0:
                continue

            next_g = g_cost + 1
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

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            map_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_data = np.empty((0, 0), dtype=np.int16)
        self.map_resolution = None
        self.map_origin = None

        self.scan_ranges = np.array([], dtype=float)
        self.scan_angles = np.array([], dtype=float)
        self.scan_frame = None

        self.path_world = []
        self.current_goal_cell = None
        self.blocked_goals = set()
        self.last_replan_time = 0.0
        self.last_plan_log_time = 0.0

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
        angles = np.mod(angles, 2.0 * math.pi)

        self.scan_ranges = ranges
        self.scan_angles = angles
        self.scan_frame = msg.header.frame_id

    def wait_for_system_ready(self, timeout_sec=STARTUP_TIMEOUT):
        self.get_logger().info('Waiting for /map, /scan, and TF map->base_link ...')
        start_time = time.time()

        while rclpy.ok() and time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            map_ready = self.map_data.size != 0 and self.map_resolution is not None and self.map_origin is not None
            scan_ready = self.scan_ranges.size != 0 and self.scan_angles.size != 0
            tf_ready = self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time())

            if map_ready and scan_ready and tf_ready:
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
            transform.transform.rotation.w
        )
        return transform.transform.translation.x, transform.transform.translation.y, yaw

    def stop(self):
        self.cmd_pub.publish(Twist())

    def pool_map(self, occ_grid, pool_size=POOL_SIZE):
        height, width = occ_grid.shape
        pad_h = (pool_size - height % pool_size) % pool_size
        pad_w = (pool_size - width % pool_size) % pool_size

        if pad_h > 0 or pad_w > 0:
            occ_grid = np.pad(
                occ_grid,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=UNKNOWN_VALUE
            )

        pooled_h = occ_grid.shape[0] // pool_size
        pooled_w = occ_grid.shape[1] // pool_size
        pooled = np.full((pooled_h, pooled_w), UNKNOWN_VALUE, dtype=np.int16)

        for row in range(pooled_h):
            for col in range(pooled_w):
                block = occ_grid[
                    row * pool_size:(row + 1) * pool_size,
                    col * pool_size:(col + 1) * pool_size
                ]
                pooled[row, col] = int(np.max(block))

        return pooled

    def classify_grid(self, pooled_occ):
        grid = np.full_like(pooled_occ, 2, dtype=np.uint8)
        grid[(pooled_occ >= 0) & (pooled_occ < WALL_THRESHOLD)] = 0
        grid[pooled_occ >= WALL_THRESHOLD] = 1
        return grid

    def inflate_walls(self, grid, radius=WALL_INFLATION_RADIUS):
        inflated = grid.copy()
        rows, cols = grid.shape

        for row, col in np.argwhere(grid == 1):
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    next_row = row + d_row
                    next_col = col + d_col
                    if 0 <= next_row < rows and 0 <= next_col < cols and inflated[next_row, next_col] == 0:
                        inflated[next_row, next_col] = 1

        return inflated

    def world_to_pooled_cell(self, world_x, world_y, pooled_shape):
        map_x = (world_x - self.map_origin.x) / self.map_resolution
        map_y = (world_y - self.map_origin.y) / self.map_resolution

        pooled_col = int(round(map_x / POOL_SIZE))
        pooled_row = int(round(map_y / POOL_SIZE))

        pooled_col = min(max(pooled_col, 0), pooled_shape[1] - 1)
        pooled_row = min(max(pooled_row, 0), pooled_shape[0] - 1)
        return pooled_row, pooled_col

    def pooled_cell_to_world(self, row, col):
        map_x = (col * POOL_SIZE + POOL_SIZE / 2.0) * self.map_resolution + self.map_origin.x
        map_y = (row * POOL_SIZE + POOL_SIZE / 2.0) * self.map_resolution + self.map_origin.y
        return map_x, map_y

    def nearest_free_to_robot(self, grid, robot_cell):
        start_row, start_col = robot_cell
        rows, cols = grid.shape

        if grid[start_row, start_col] == 0:
            return robot_cell

        queue = deque([robot_cell])
        visited = {robot_cell}

        while queue:
            row, col = queue.popleft()
            for next_cell in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                next_row, next_col = next_cell
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if next_cell in visited:
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
                for next_row, next_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
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

                    for next_row, next_col in [
                        (cur_row - 1, cur_col),
                        (cur_row + 1, cur_col),
                        (cur_row, cur_col - 1),
                        (cur_row, cur_col + 1),
                    ]:
                        if 0 <= next_row < rows and 0 <= next_col < cols:
                            if frontier_mask[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                queue.append((next_row, next_col))

                if len(cluster) >= MIN_FRONTIER_SIZE:
                    clusters.append(cluster)

        return clusters

    def rank_frontier_goals(self, clusters, robot_cell, inflated_grid):
        robot_row, robot_col = robot_cell
        candidates = []

        for cluster in clusters:
            best_cell = min(
                cluster,
                key=lambda cell: abs(cell[0] - robot_row) + abs(cell[1] - robot_col)
            )

            if best_cell in self.blocked_goals:
                continue
            if inflated_grid[best_cell] != 0:
                continue

            distance = abs(best_cell[0] - robot_row) + abs(best_cell[1] - robot_col)
            candidates.append((distance, best_cell))

        candidates.sort(key=lambda item: item[0])
        return [cell for _, cell in candidates]

    def simplify_path(self, path_world):
        if len(path_world) <= 2:
            return path_world

        simplified = [path_world[0]]
        for index in range(1, len(path_world) - 1):
            if index % 2 == 0:
                simplified.append(path_world[index])
        simplified.append(path_world[-1])
        return simplified

    def log_plan_status(self, message):
        now = time.time()
        if now - self.last_plan_log_time >= 1.0:
            self.get_logger().info(message)
            self.last_plan_log_time = now

    def plan_route(self):
        if self.map_data.size == 0:
            return []

        pooled_occ = self.pool_map(self.map_data)
        raw_grid = self.classify_grid(pooled_occ)
        inflated_grid = self.inflate_walls(raw_grid)

        free_cells = int(np.count_nonzero(raw_grid == 0))
        blocked_cells = int(np.count_nonzero(raw_grid == 1))
        unknown_cells = int(np.count_nonzero(raw_grid == 2))

        if free_cells < MIN_FREE_CELLS_TO_PLAN:
            self.log_plan_status(
                f'Waiting for more mapped free space: free={free_cells} blocked={blocked_cells} unknown={unknown_cells}'
            )
            return []

        try:
            robot_x, robot_y, _ = self.get_pose()
        except TransformException as exc:
            self.get_logger().warn(f'Pose unavailable during planning: {exc}')
            return []

        robot_cell = self.world_to_pooled_cell(robot_x, robot_y, raw_grid.shape)
        robot_cell = self.nearest_free_to_robot(inflated_grid, robot_cell)
        if robot_cell is None:
            self.log_plan_status(
                f'Robot is not near any free cell after inflation: free={free_cells} blocked={blocked_cells} unknown={unknown_cells}'
            )
            return []

        frontier_clusters = self.find_frontier_clusters(raw_grid)
        if not frontier_clusters:
            self.log_plan_status(
                f'No frontier clusters found: free={free_cells} blocked={blocked_cells} unknown={unknown_cells}'
            )
            return []

        candidate_goals = self.rank_frontier_goals(frontier_clusters, robot_cell, inflated_grid)
        if not candidate_goals:
            inflated_free_cells = int(np.count_nonzero(inflated_grid == 0))
            self.log_plan_status(
                'No usable frontier goals found: '
                f'free={free_cells} inflated_free={inflated_free_cells} '
                f'blocked={blocked_cells} unknown={unknown_cells} frontiers={len(frontier_clusters)} '
                f'blocked_goals={len(self.blocked_goals)}'
            )
            return []

        for goal_cell in candidate_goals:
            path_cells = astar(inflated_grid, robot_cell, goal_cell)
            if not path_cells:
                self.blocked_goals.add(goal_cell)
                continue

            self.current_goal_cell = goal_cell
            path_world = [self.pooled_cell_to_world(row, col) for row, col in path_cells]
            path_world = self.simplify_path(path_world)

            self.get_logger().info(
                f'Planned path with {len(path_world)} waypoints to frontier row={goal_cell[0]} col={goal_cell[1]}'
            )
            return path_world

        self.get_logger().warn('All candidate frontiers were unreachable.')
        return []

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

        return float(np.min(sector_ranges))

    def get_scan_clearance(self):
        return {
            'front': self.sector_distance(0.0, FRONT_HALF_ANGLE_DEG),
            'left': self.sector_distance(90.0, SIDE_CENTER_TOLERANCE_DEG),
            'back': self.sector_distance(180.0, FRONT_HALF_ANGLE_DEG),
            'right': self.sector_distance(270.0, SIDE_CENTER_TOLERANCE_DEG),
        }

    def pick_turn_direction(self, clearance):
        left = clearance['left']
        right = clearance['right']

        if left is None and right is None:
            return 1.0
        if left is None:
            return -1.0
        if right is None:
            return 1.0
        return 1.0 if left >= right else -1.0

    def mark_current_goal_blocked(self):
        if self.current_goal_cell is not None:
            self.blocked_goals.add(self.current_goal_cell)
        self.current_goal_cell = None
        self.path_world = []

    def recover_and_replan(self, reason, turn_sign=None):
        now = time.time()
        if now - self.last_replan_time < REPLAN_COOLDOWN:
            return False

        self.last_replan_time = now
        self.get_logger().warn(f'Replanning: {reason}')

        self.stop()

        clearance = self.get_scan_clearance()
        if turn_sign is None:
            turn_sign = self.pick_turn_direction(clearance)

        reverse_cmd = Twist()
        if clearance['back'] is None or clearance['back'] > 0.22:
            reverse_cmd.linear.x = -0.06
            self.cmd_pub.publish(reverse_cmd)
            time.sleep(0.45)
            self.stop()

        rotate_cmd = Twist()
        rotate_cmd.angular.z = turn_sign * 0.55
        self.cmd_pub.publish(rotate_cmd)
        time.sleep(0.6)
        self.stop()

        self.mark_current_goal_blocked()
        return True

    def drive_to_waypoint(self, waypoint, is_final_waypoint=False):
        goal_x, goal_y = waypoint
        reach_distance = GOAL_REACHED_DISTANCE if is_final_waypoint else WAYPOINT_REACHED_DISTANCE

        try:
            start_x, start_y, _ = self.get_pose()
        except TransformException:
            return False

        last_progress_pose = (start_x, start_y)
        last_progress_time = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            try:
                robot_x, robot_y, robot_yaw = self.get_pose()
            except TransformException:
                continue

            delta_x = goal_x - robot_x
            delta_y = goal_y - robot_y
            distance = math.hypot(delta_x, delta_y)

            if distance < reach_distance:
                self.stop()
                return True

            traveled_since_progress = math.hypot(
                robot_x - last_progress_pose[0],
                robot_y - last_progress_pose[1]
            )
            if traveled_since_progress >= MIN_PROGRESS_DISTANCE:
                last_progress_pose = (robot_x, robot_y)
                last_progress_time = time.time()
            elif time.time() - last_progress_time > STUCK_TIMEOUT:
                self.recover_and_replan('robot is not making progress')
                return False

            clearance = self.get_scan_clearance()
            front = clearance['front']
            left = clearance['left']
            right = clearance['right']

            target_yaw = math.atan2(delta_y, delta_x)
            yaw_error = wrap_angle(target_yaw - robot_yaw)
            yaw_error_deg = abs(math.degrees(yaw_error))

            if front is not None and front < STOP_DISTANCE:
                turn_sign = self.pick_turn_direction(clearance)
                self.recover_and_replan('front path is blocked', turn_sign=turn_sign)
                return False

            cmd = Twist()
            if yaw_error_deg > 25.0:
                cmd.linear.x = 0.0
                cmd.angular.z = ANGULAR_SPEED if yaw_error > 0.0 else -ANGULAR_SPEED
            else:
                speed_scale = max(0.20, min(1.0, distance / 0.8))
                cmd.linear.x = LINEAR_SPEED * speed_scale

                if front is not None and front < CAUTION_DISTANCE:
                    cmd.linear.x *= 0.4

                if left is not None and left < SIDE_CLEARANCE:
                    cmd.angular.z -= 0.45
                    cmd.linear.x = min(cmd.linear.x, 0.05)
                if right is not None and right < SIDE_CLEARANCE:
                    cmd.angular.z += 0.45
                    cmd.linear.x = min(cmd.linear.x, 0.05)

                cmd.angular.z += max(-MAX_ANGULAR_CORRECTION, min(MAX_ANGULAR_CORRECTION, 1.4 * yaw_error))

            cmd.angular.z = max(-MAX_ANGULAR_CORRECTION, min(MAX_ANGULAR_CORRECTION, cmd.angular.z))
            self.cmd_pub.publish(cmd)
            time.sleep(CONTROL_DT)

        self.stop()
        return False

    def follow_path(self):
        while rclpy.ok() and self.path_world:
            waypoint = self.path_world.pop(0)
            is_final_waypoint = len(self.path_world) == 0
            if not self.drive_to_waypoint(waypoint, is_final_waypoint=is_final_waypoint):
                return False

        self.current_goal_cell = None
        return True

    def mover(self):
        if not self.wait_for_system_ready():
            self.get_logger().warn('Exiting because system is not ready.')
            return

        start_time = time.time()
        while rclpy.ok() and not self.path_world:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.path_world = self.plan_route()
            if time.time() - start_time > INITIAL_PLAN_TIMEOUT:
                self.get_logger().warn('No initial path found. Stopping.')
                self.stop()
                return

        while rclpy.ok():
            if not self.path_world:
                self.stop()
                self.path_world = self.plan_route()
                if not self.path_world:
                    self.get_logger().info('No more reachable frontier paths. Stopping.')
                    break

            self.follow_path()

        self.stop()


def main(args=None):
    rclpy.init(args=args)

    node = AutoNav()
    try:
        node.mover()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
