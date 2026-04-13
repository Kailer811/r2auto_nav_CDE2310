#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
import math
import time
import RPi.GPIO as GPIO


class ArucoDockingImproved(Node):
    def __init__(self):
        super().__init__('aruco_docking')

        # ==================== TARGET POSITION ====================
        self.target_distance = self.declare_parameter('target_distance', 0.30).value
        self.target_lateral = self.declare_parameter('target_lateral', -0.1269).value

        # ==================== CONTROL GAINS ====================
        self.kp_distance = self.declare_parameter('kp_distance', 0.5).value
        self.kp_lateral = self.declare_parameter('kp_lateral', 1.2).value

        # ==================== LIMITS ====================
        self.max_linear_speed = float(self.declare_parameter('max_linear_speed', 0.12).value)
        self.max_angular_speed = float(self.declare_parameter('max_angular_speed', 0.6).value)
        self.max_accel_linear = self.declare_parameter('max_accel_linear', 0.3).value
        self.max_accel_angular = self.declare_parameter('max_accel_angular', 1.0).value

        # ==================== TOLERANCES ====================
        self.distance_tolerance = self.declare_parameter('distance_tolerance', 0.05).value
        self.lateral_tolerance = self.declare_parameter('lateral_tolerance', 0.03).value

        # ==================== SAFETY ====================
        self.min_safe_distance = self.declare_parameter('min_safe_distance', 0.20).value
        self.marker_timeout = self.declare_parameter('marker_timeout', 0.8).value

        # ==================== STATE MACHINE ====================
        self.state_idle = 0
        self.state_docking = 1
        self.state_launching = 2
        self.state_complete = 3
        
        self.state = self.state_idle
        
        self.last_pose = None
        self.last_seen_time = 0.0
        self.docking_active = False

        self.prev_linear = 0.0
        self.prev_angular = 0.0
        
        # Launch state machine
        self.launch_start_time = 0.0
        self.launch_step = 0

        # ==================== ROS INTERFACES ====================
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/aruco/pose',
            self.pose_callback,
            10
        )

        self.docking_trigger_sub = self.create_subscription(
            Bool,
            '/trigger_objA',
            self.docking_trigger_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/docking/status', 10)

        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_loop)

        # ==================== GPIO SETUP ====================
        # ✅ Motor A (flywheel) pins
        self.ena = 18   # PWM speed control (Motor A)
        self.in1 = 23   # Direction 1 (Motor A)
        self.in2 = 24   # Direction 2 (Motor A)
        
        # ✅ Servo on GPIO 13
        self.servo = 13

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.ena, GPIO.OUT)
        GPIO.setup(self.servo, GPIO.OUT)

        self.motor_pwm = GPIO.PWM(self.ena, 1000)
        self.motor_pwm.start(0)

        self.servo_pwm = GPIO.PWM(self.servo, 50)
        self.servo_pwm.start(0)

        # Servo angles (calibrate these!)
        self.servo_open = 2.5
        self.servo_close = 7.5

        self.feeder_close()

        self.get_logger().info("Aruco Docking node initialized. Ready for docking sequence!")

    # ==================== CALLBACKS ====================
    def pose_callback(self, msg):
        """Store latest marker pose"""
        self.last_pose = msg
        self.last_seen_time = time.time()

    def docking_trigger_callback(self, msg):
        """Handle docking trigger from RamIsBetter when marker is aligned"""
        self.get_logger().info(f"Docking trigger received: {msg.data}")
        
        if msg.data and self.state == self.state_idle:
            # Start docking
            self.get_logger().info("Starting docking sequence")
            self.state = self.state_docking
            self.docking_active = True
            self.prev_linear = 0.0
            self.prev_angular = 0.0
            
        elif not msg.data:
            # Stop/abort docking
            self.get_logger().info("Docking aborted")
            self.state = self.state_idle
            self.docking_active = False
            self.stop_robot()
            self.motor_stop()

    # ==================== MAIN CONTROL LOOP ====================
    def control_loop(self):
        """Main state machine - runs at 20 Hz"""
        
        # ========== STATE: IDLE ==========
        if self.state == self.state_idle:
            self.stop_robot()
            return
        
        # ========== STATE: DOCKING ==========
        elif self.state == self.state_docking:
            self.docking_control()
        
        # ========== STATE: LAUNCHING ==========
        elif self.state == self.state_launching:
            self.launch_control()
        
        # ========== STATE: COMPLETE ==========
        elif self.state == self.state_complete:
            self.stop_robot()
            # Wait here until FSM sends new command

    # ==================== DOCKING CONTROL ====================
    def docking_control(self):
        """Docking state - align with marker"""
        
        # Safety: No marker visible
        if self.last_pose is None:
            self.get_logger().warn("No marker visible - waiting...", throttle_duration_sec=2.0)
            self.stop_robot()
            return

        # Safety: Marker timeout
        time_since_seen = time.time() - self.last_seen_time
        if time_since_seen > self.marker_timeout:
            self.get_logger().warn(
                f"Marker lost for {time_since_seen:.1f}s - STOPPING",
                throttle_duration_sec=1.0
            )
            self.stop_robot()
            return

        # Extract marker position
        marker_lateral = self.last_pose.pose.position.x
        marker_distance = self.last_pose.pose.position.z

        # Calculate errors
        distance_error = marker_distance - self.target_distance
        lateral_error = marker_lateral - self.target_lateral

        # Check if docked
        if (abs(distance_error) < self.distance_tolerance and
            abs(lateral_error) < self.lateral_tolerance):
            
            self.get_logger().info(
                f"DOCKED! Distance: {marker_distance:.3f}m, Lateral: {marker_lateral:.3f}m"
            )
            
            # Transition to launching state
            self.state = self.state_launching
            self.launch_step = 0
            self.launch_start_time = time.time()
            self.stop_robot()
            
            return

        # ========== CONTROL CALCULATIONS ==========
        
        # Adaptive gain
        if abs(distance_error) < 0.3:
            distance_gain = self.kp_distance * 0.6
        else:
            distance_gain = self.kp_distance

        linear_cmd = distance_gain * distance_error
        angular_cmd = -self.kp_lateral * lateral_error

        # Safety: Too close
        if marker_distance < self.min_safe_distance:
            if linear_cmd > 0:
                linear_cmd = 0.0
                self.get_logger().warn("Too close - stopping forward motion", throttle_duration_sec=1.0)

        # Clamp velocities
        linear_cmd = max(min(linear_cmd, self.max_linear_speed), -self.max_linear_speed)
        angular_cmd = max(min(angular_cmd, self.max_angular_speed), -self.max_angular_speed)

        # Smooth acceleration
        linear_cmd = self.rate_limit(linear_cmd, self.prev_linear, self.max_accel_linear * self.dt)
        angular_cmd = self.rate_limit(angular_cmd, self.prev_angular, self.max_accel_angular * self.dt)

        self.prev_linear = linear_cmd
        self.prev_angular = angular_cmd

        # Publish command
        cmd = Twist()
        cmd.linear.x = float(linear_cmd)
        cmd.angular.z = float(angular_cmd)
        self.cmd_pub.publish(cmd)

        # Log status
        self.get_logger().info(
            f"Docking: dist={marker_distance:.2f}m (err={distance_error:+.2f}), "
            f"lat={marker_lateral:+.2f}m (err={lateral_error:+.2f}) | "
            f"cmd: v={linear_cmd:.2f}, ω={angular_cmd:.2f}",
            throttle_duration_sec=0.5
        )

    # ==================== LAUNCH CONTROL (NON-BLOCKING) ====================
    def launch_control(self):
        """Non-blocking launch sequence using state machine"""
        
        # For testing: just print docked and return to idle
        self.get_logger().info("Docked")
        print("Docked")
        
        # Reset state to IDLE
        self.state = self.state_idle
        self.docking_active = False
        self.get_logger().info("Returning to IDLE state")

    # ==================== MOTOR FUNCTIONS ====================
    def motor_forward(self, speed=80):
        """Spin flywheel forward at given speed (0-100%)"""
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.motor_pwm.ChangeDutyCycle(speed)

    def motor_stop(self):
        """Stop flywheel"""
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.motor_pwm.ChangeDutyCycle(0)

    # ==================== SERVO FUNCTIONS ====================
    def feeder_open(self):
        """Open feeder gate (non-blocking version)"""
        self.servo_pwm.ChangeDutyCycle(self.servo_open)

    def feeder_close(self):
        """Close feeder gate (non-blocking version)"""
        self.servo_pwm.ChangeDutyCycle(self.servo_close)

    # ==================== UTILITIES ====================
    def rate_limit(self, desired, previous, max_change):
        """Limit rate of change for smooth acceleration"""
        delta = desired - previous
        if delta > max_change:
            return previous + max_change
        elif delta < -max_change:
            return previous - max_change
        else:
            return desired

    def stop_robot(self):
        """Stop robot motion"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.prev_linear = 0.0
        self.prev_angular = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDockingImproved()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⚠️  Shutting down...")
    finally:
        node.stop_robot()
        node.motor_stop()
        node.motor_pwm.stop()
        node.servo_pwm.stop()
        GPIO.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
