"""
AMR Payload Deployment Test: Static Target
This script tests the flywheel and servo motor actuation for a static target scenario.
It spins up the flywheel, deploys 3 ping pong balls with specific timing delays 
(3s and 7s).
"""

import RPi.GPIO as GPIO
import time

# --- Hardware Pin Configuration (BCM Mode) ---
FLYWHEEL_PIN = 18  # Controls the MOSFET gate for the flywheel
SERVO_PIN = 19     # Outputs PWM signal to the feeder servo motor

# --- Servo PWM Configuration ---
PWM_FREQUENCY = 50   
# Note: Duty cycle values may need slight calibration depending on your specific servo brand
DUTY_CYCLE_REST = 2.5    # Approximate duty cycle for 0 degrees (Home position)
DUTY_CYCLE_FIRE = 7.5    # Approximate duty cycle for 90 degrees (Drop one ball)

def setup_gpio():
    """Initializes GPIO pins and sets up the PWM instance."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup Flywheel pin
    GPIO.setup(FLYWHEEL_PIN, GPIO.OUT)
    GPIO.output(FLYWHEEL_PIN, GPIO.LOW) # Ensure it's OFF initially
    
    # Setup Servo pin
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    servo_pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
    servo_pwm.start(DUTY_CYCLE_REST) # Move to initial blocking position
    time.sleep(0.5) # Give servo time to reach home position
    
    return servo_pwm

def deploy_payload(servo_pwm, ball_number):
    """Actuates the servo to drop exactly one ball and immediately returns to block."""
    print(f"Deploying ball #{ball_number}...")
    
    # Rotate 90 degrees to release one ball
    servo_pwm.ChangeDutyCycle(DUTY_CYCLE_FIRE)
    time.sleep(0.5) # Wait for mechanical movement and ball drop clearance
    
    # Immediately rotate back to -90 degrees (Home) to block the next ball
    servo_pwm.ChangeDutyCycle(DUTY_CYCLE_REST)
    time.sleep(0.5) # Wait for mechanical movement to complete

def main():
    servo_pwm = setup_gpio()
    
    try:
        print("=== Starting Static Target Deployment Test ===")
        
        # Step 1: Turn on the flywheel and pre-heat
        print("Powering on flywheel MOSFET (GPIO 18: HIGH)...")
        GPIO.output(FLYWHEEL_PIN, GPIO.HIGH)
        print("Warming up flywheel for 3 seconds...")
        time.sleep(3.0) 
        
        # Step 2: Deploy 1st ball
        deploy_payload(servo_pwm, 1)
        
        # Step 3: Wait 3 seconds
        print("Waiting 3 seconds...")
        time.sleep(3.0)
        
        # Step 4: Deploy 2nd ball
        deploy_payload(servo_pwm, 2)
        
        # Step 5: Wait 7 seconds
        print("Waiting 7 seconds...")
        time.sleep(7.0)
        
        # Step 6: Deploy 3rd ball
        deploy_payload(servo_pwm, 3)
        
        print("Deployment sequence completed successfully.")

    except KeyboardInterrupt:
        print("\nTest interrupted by user (Ctrl+C).")
        
    finally:
        # Step 7: Safe shutdown
        print("Initiating safe shutdown...")
        GPIO.output(FLYWHEEL_PIN, GPIO.LOW) # Turn off the flywheel
        servo_pwm.ChangeDutyCycle(DUTY_CYCLE_REST) # Ensure servo is in block position
        time.sleep(0.5)
        servo_pwm.stop()
        GPIO.cleanup()
        print("Flywheel stopped (GPIO 18: LOW). GPIO cleaned up. Exiting.")

if __name__ == '__main__':
    main()