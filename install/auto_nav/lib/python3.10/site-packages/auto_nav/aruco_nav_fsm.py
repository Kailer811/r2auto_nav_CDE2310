#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


MODE_NAV = 'nav'
MODE_TO_DOCK = 'to_dock'
MODE_DOCK = 'dock'
MODE_TO_NAV = 'to_nav'


class ArucoNavFsm(Node):
    def __init__(self):
        super().__init__('aruco_nav_fsm')

        self.aruco_loss_timeout = float(
            self.declare_parameter('aruco_loss_timeout', 7.0).value)
        self.post_trigger_timeout = float(
            self.declare_parameter('post_trigger_timeout', 25.0).value)
        self.timer_period = float(
            self.declare_parameter('timer_period', 0.1).value)
        self.nav_settle_delay = float(
            self.declare_parameter('nav_settle_delay', 1.5).value)
        self.dock_settle_delay = float(
            self.declare_parameter('dock_settle_delay', 1.5).value)
        self.retrigger_guard_delay = float(
            self.declare_parameter('retrigger_guard_delay', 1.0).value)

        self.stop_nav_pub = self.create_publisher(Bool, '/stop_nav', 10)
        self.enable_docking_pub = self.create_publisher(Bool, '/enable_docking', 10)
        self.aruco_detected_sub = self.create_subscription(
            Bool, '/aruco_detected', self.aruco_detected_callback, 10)
        self.trigger_obj_a_sub = self.create_subscription(
            Bool, '/trigger_objA', self.trigger_callback, 10)
        self.trigger_obj_b_sub = self.create_subscription(
            Bool, '/trigger_objB', self.trigger_callback, 10)

        self.has_detected_marker_once = False
        self.aruco_visible = False
        self.last_detected_time = None
        self.last_trigger_time = None

        self.mode = MODE_NAV
        self.transition_deadline = None
        self.stop_nav_active = False
        self.docking_enabled = False

        self.create_timer(self.timer_period, self.timer_callback)
        self.publish_enable_docking(False)
        self.publish_stop_nav(False)

        self.get_logger().info(
            'ArucoNavFsm started. '
            f'aruco_loss_timeout={self.aruco_loss_timeout:.1f}s, '
            f'post_trigger_timeout={self.post_trigger_timeout:.1f}s, '
            f'nav_settle_delay={self.nav_settle_delay:.2f}s, '
            f'dock_settle_delay={self.dock_settle_delay:.2f}s')

    def aruco_detected_callback(self, msg):
        now = time.time()
        detected = bool(msg.data)

        if detected:
            self.has_detected_marker_once = True
            self.aruco_visible = True
            self.last_detected_time = now
        else:
            self.aruco_visible = False

        self.update_fsm(now)

    def trigger_callback(self, msg):
        if not bool(msg.data):
            return

        now = time.time()
        self.last_trigger_time = now
        self.get_logger().info('Received docking trigger. Holding docking ownership.')
        self.update_fsm(now)

    def timer_callback(self):
        self.update_fsm(time.time())

    def docking_requested(self, now):
        trigger_active = (
            self.last_trigger_time is not None
            and (now - self.last_trigger_time) <= self.post_trigger_timeout
        )
        detection_hold_active = (
            self.has_detected_marker_once
            and self.last_detected_time is not None
            and (now - self.last_detected_time) <= self.aruco_loss_timeout
        )
        return self.aruco_visible or detection_hold_active or trigger_active

    def update_fsm(self, now):
        docking_requested = self.docking_requested(now)

        if self.mode == MODE_NAV:
            self.publish_enable_docking(False)
            self.publish_stop_nav(False)
            if docking_requested:
                self.publish_stop_nav(True)
                self.publish_enable_docking(False)
                self.mode = MODE_TO_DOCK
                self.transition_deadline = now + self.nav_settle_delay
                self.get_logger().info('FSM: nav -> to_dock')
            return

        if self.mode == MODE_TO_DOCK:
            self.publish_stop_nav(True)
            self.publish_enable_docking(False)
            if not docking_requested:
                self.mode = MODE_NAV
                self.transition_deadline = None
                self.get_logger().info('FSM: to_dock -> nav')
                return
            if now >= self.transition_deadline:
                self.publish_enable_docking(True)
                self.mode = MODE_DOCK
                self.transition_deadline = now + self.retrigger_guard_delay
                self.get_logger().info('FSM: to_dock -> dock')
            return

        if self.mode == MODE_DOCK:
            self.publish_stop_nav(True)
            self.publish_enable_docking(True)
            if self.transition_deadline is not None and now < self.transition_deadline:
                return
            if not docking_requested:
                self.publish_enable_docking(False)
                self.publish_stop_nav(True)
                self.mode = MODE_TO_NAV
                self.transition_deadline = now + self.dock_settle_delay
                self.get_logger().info('FSM: dock -> to_nav')
            return

        if self.mode == MODE_TO_NAV:
            self.publish_enable_docking(False)
            self.publish_stop_nav(True)
            if docking_requested:
                self.mode = MODE_TO_DOCK
                self.transition_deadline = now + self.nav_settle_delay
                self.get_logger().info('FSM: to_nav -> to_dock')
                return
            if now >= self.transition_deadline:
                self.publish_stop_nav(False)
                self.mode = MODE_NAV
                self.transition_deadline = None
                self.get_logger().info('FSM: to_nav -> nav')

    def publish_stop_nav(self, state):
        state = bool(state)
        if state == self.stop_nav_active:
            return

        self.stop_nav_active = state
        msg = Bool()
        msg.data = state
        self.stop_nav_pub.publish(msg)
        self.get_logger().info(f'Publishing stop_nav={state}')

    def publish_enable_docking(self, state):
        state = bool(state)
        if state == self.docking_enabled:
            return

        self.docking_enabled = state
        msg = Bool()
        msg.data = state
        self.enable_docking_pub.publish(msg)
        self.get_logger().info(f'Publishing enable_docking={state}')


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavFsm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_enable_docking(False)
        node.publish_stop_nav(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
