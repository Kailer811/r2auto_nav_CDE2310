# Electrical Design

This readme has highlighted the connections between the electrical components in the design including the specific pinout and destination. 

Besides the electrical components that Turtlebot3 Burger provides (LiDAR, Raspberry Pi, and OpenCR), the external electrical components are used to ensure the mission completion. The external eletrical components includes servo motor (feeding the ball), flywheel (launching the ball), and ultrasonic sensor (moving target detection). 

| Component | Connection on Raspberry Pi pinout |
| :---: | :---: |
| Servo motor PWM pin | GPIO 19 |
| Ultrasonic sensor (trigger) | GPIO 23 |
| Ultrasonic sensor (echo) | GPIO 24 |
| Flywheel control (MOSFET gate) | GPIO 18 |

All the power supply of the components listed in the table need to be connected to 5V, and the ground should be connected to the common GND on Raspberry Pi to ensure the system has the same reference grounding point. 

There is addtional "Flywheel control circuit" under Schematic Diagram for Electrical Design.png. This circuit is the key circuit to control the on and off of the flywheel. 
