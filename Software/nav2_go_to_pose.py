#!/usr/bin/env python3

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class Nav2GoToPose(Node):
    def __init__(self):
        super().__init__('nav2_go_to_pose')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_done = False

    def go_to_map_position(self, x, y, yaw=0.0):
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 action server not available.')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Sending Nav2 goal to map coordinates x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal.')
            self.goal_done = True
            return

        self.get_logger().info('Nav2 accepted the goal.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        if result.status == 4:
            self.get_logger().info('Goal reached successfully.')
        else:
            self.get_logger().warn(f'Goal finished with status={result.status}.')
        self.goal_done = True

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose.position
        self.get_logger().info(
            f'Current pose x={current_pose.x:.2f}, y={current_pose.y:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoToPose()

    cli_args = sys.argv[1:]
    if len(cli_args) < 2:
        node.get_logger().error('Usage: ros2 run auto_nav nav2_go_to_pose <x> <y> [yaw]')
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        x = float(cli_args[0])
        y = float(cli_args[1])
        yaw = float(cli_args[2]) if len(cli_args) > 2 else 0.0
    except ValueError:
        node.get_logger().error('x, y, and yaw must be numbers.')
        node.destroy_node()
        rclpy.shutdown()
        return

    if not node.go_to_map_position(x, y, yaw):
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        while rclpy.ok() and not node.goal_done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
