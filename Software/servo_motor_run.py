#!/usr/bin/env python3

import time

import RPi.GPIO as GPIO


MOTOR_PIN = 13  # BCM numbering


def run_motor():
    """Run a standard DC motor at constant HIGH until interrupted."""
    GPIO.output(MOTOR_PIN, GPIO.HIGH)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def aruco_detected():
    """
    Placeholder for future ArUco-triggered motor control.
    Replace this with your camera / ArUco detection logic later.
    """
    return False


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)

    try:
        # Future behavior:
        # if aruco_detected():
        #     run_motor()

        # For now, always run the motor when the script starts.
        run_motor()
    finally:
        GPIO.output(MOTOR_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
