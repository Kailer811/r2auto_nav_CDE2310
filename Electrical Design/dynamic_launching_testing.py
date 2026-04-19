"""
AMR Payload Deployment Test: Dynamic Targets
This script controls the flywheel and servo to deploy 3 ping pong balls.
Deployment is strictly triggered when an obstacle is detected within the 
target window (17 cm to 23 cm) by the ultrasonic sensor.
"""

import RPi.GPIO as GPIO
import time

# --- Hardware Pin Configuration (BCM Mode) ---
FLYWHEEL_PIN = 18    # Controls the MOSFET gate for the DC motor
SERVO_PIN = 19       # Outputs PWM signal to the feeder servo
TRIG_PIN = 23        # Ultrasonic Trigger Pin
ECHO_PIN = 24        # Ultrasonic Echo Pin

# --- Operation Parameters ---
PWM_FREQUENCY = 50       # 50Hz for standard servo
DUTY_CYCLE_REST = 2.5    # Servo home position (Block)
DUTY_CYCLE_FIRE = 7.5    # Servo fire position (Drop 1 ball)
TARGET_MIN_DIST = 17.0   # Minimum trigger distance (cm)
TARGET_MAX_DIST = 23.0   # Maximum trigger distance (cm)
MAX_PAYLOAD = 3          # Total number of balls to fire

def setup_gpio():
    """Initializes all GPIO pins for motors and sensors."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor Setup
    GPIO.setup(FLYWHEEL_PIN, GPIO.OUT)
    GPIO.output(FLYWHEEL_PIN, GPIO.LOW) 
    
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    servo_pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
    servo_pwm.start(DUTY_CYCLE_REST)
    
    # Ultrasonic Sensor Setup
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    
    # Allow sensor to settle
    time.sleep(0.5) 
    return servo_pwm

def get_distance():
    """
    Sends a 10us pulse to the ultrasonic sensor and measures the echo duration.
    Includes a timeout mechanism to prevent hanging on non-RTOS systems.
    Returns distance in cm, or -1 if timeout occurs.
    """

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    start_time = time.time()
    stop_time = time.time()
    timeout_start = start_time
    
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
        if start_time - timeout_start > 0.1:
            return -1.0 
            
    timeout_start = start_time
    # Wait for the Echo pin to go LOW (Pulse end)
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
        # 0.1s timeout
        if stop_time - timeout_start > 0.1:
            return -1.0
            
    # Calculate pulse length
    time_elapsed = stop_time - start_time
    
    # Sonic speed is approx 34300 cm/s. 
    # Distance = (Time * Speed) / 2 (round trip)
    distance = (time_elapsed * 34300) / 2
    return round(distance, 2)

def deploy_payload(servo_pwm, ball_number):
    """Actuates the servo to drop exactly one ball and resets."""
    print(f"[ACTION] Firing ball #{ball_number}!")
    servo_pwm.ChangeDutyCycle(DUTY_CYCLE_FIRE)
    time.sleep(0.5) # Wait for drop clearance
    
    servo_pwm.ChangeDutyCycle(DUTY_CYCLE_REST)
    time.sleep(0.5) # Wait for return movement

def main():
    servo_pwm = setup_gpio()
    balls_fired = 0
    
    try:
        print("=== Starting Dynamic Target (Ultrasonic) Deployment Test ===")
        
        # Step 1: Pre-heat the flywheel
        print("Powering on flywheel MOSFET (GPIO 18: HIGH)...")
        GPIO.output(FLYWHEEL_PIN, GPIO.HIGH)
        print("Warming up flywheel for 3 seconds...")
        time.sleep(3.0)
        
        print(f"Tracking target... Waiting for object in {TARGET_MIN_DIST}cm - {TARGET_MAX_DIST}cm range.")
        
        # Step 2: Continuous sensing loop
        while balls_fired < MAX_PAYLOAD:
            dist = get_distance()
            
            # Print distance for debugging/monitoring
            if dist > 0:
                print(f"Current Distance: {dist} cm", end="\r")
            
            # Step 3: Trigger check
            if TARGET_MIN_DIST <= dist <= TARGET_MAX_DIST:
                print(f"\n\n[DETECTED] Target locked at {dist} cm!")
                balls_fired += 1
                deploy_payload(servo_pwm, balls_fired)
                
                # Small cooldown to allow flywheel RPM to recover 
                # and prevent rapid multi-firing on the same pass
                if balls_fired < MAX_PAYLOAD:
                    print("Recharging flywheel RPM...\n")
                    time.sleep(1.0) 
                    
            time.sleep(0.05) 
            
        print(f"\nMission Accomplished: All {MAX_PAYLOAD} balls deployed.")

    except KeyboardInterrupt:
        print("\nTest interrupted by user (Ctrl+C).")
        
    finally:
        # Step 4: Safe shutdown sequence
        print("Initiating safe shutdown...")
        GPIO.output(FLYWHEEL_PIN, GPIO.LOW) # Turn off the flywheel immediately
        servo_pwm.ChangeDutyCycle(DUTY_CYCLE_REST)
        time.sleep(0.5)
        servo_pwm.stop()
        GPIO.cleanup()
        print("Hardware safed. Exiting.")

if __name__ == '__main__':
    main()