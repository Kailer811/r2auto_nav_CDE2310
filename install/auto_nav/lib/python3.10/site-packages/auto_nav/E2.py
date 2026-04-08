import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import RPi.GPIO as GPIO
import time


SolenoidPin = 20
ServoPin = 29
pwm = GPIO.PWM(ServoPin, 50)
pwm.start(0)

class LidarSubscriber(Node):
    
    

    def __init__(self):
        super().__init__('lidar_subscriber_node')

        # 1. Create a variable to store the lidar data
        self.latest_scan = None 

        # 2. Subscribe to the '/scan' topic
        # Note: We use qos_profile_sensor_data because Lidar is a high-frequency sensor
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.servoSolenoid,
            qos_profile_sensor_data)
        
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SolenoidPin, GPIO.OUT)
        GPIO.output(SolenoidPin, GPIO.LOW)  
        GPIO.setup(ServoPin, GPIO.OUT)
        GPIO.output(ServoPin, GPIO.LOW)  

        self.get_logger().info('Lidar Subscriber Node has been started.')

    def servoSolenoid(self, msg):
        # 3. Store the entire lidar message into your variable
        self.latest_scan = msg

        # Optional: Example of how to use the variable
        if self.latest_scan is not None:
            # Index 0 is usually directly in front of the TurtleBot
            front_distance = self.latest_scan.ranges[0]
            self.get_logger().info(f'Distance to front: {front_distance:.2f} m')
            if front_distance < 1:
                print("servo should have turned")
                GPIO.output(SolenoidPin, GPIO.HIGH)
                GPIO.output(ServoPin, GPIO.HIGH)  
                duty = 2 + (90/ 18)   # maps 0–180° → ~2–12% duty
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.5)
                pwm.ChangeDutyCycle(0)    # stop jitter

            else:
                GPIO.output(SolenoidPin, GPIO.LOW)
                GPIO.output(ServoPin, GPIO.LOW)


        

def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()
    
    try:
        rclpy.spin(lidar_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        lidar_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
