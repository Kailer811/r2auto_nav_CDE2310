# Electrical Design

This readme has highlighted the connections between the electrical components in the design including the specific pinout and destination. 

Besides the electrical components that Turtlebot3 Burger provides (LiDAR, Raspberry Pi, and OpenCR), the external electrical components are used to ensure the mission completion. The external eletrical components includes servo motor (feeding the ball), flywheel (launching the ball), and ultrasonic sensor (moving target detection). 
<div align="center">

| Component | Connection on Raspberry Pi pinout |
| :---: | :---: |
| Servo motor PWM pin | GPIO 19 |
| Ultrasonic sensor (trigger) | GPIO 23 |
| Ultrasonic sensor (echo) | GPIO 24 |
| Flywheel control (MOSFET gate) | GPIO 18 |

</div>

All the power supply of the components listed in the table need to be connected to 5V, and the ground should be connected to the common GND on Raspberry Pi to ensure the system has the same grounding reference point. 

There is a "Flywheel control circuit" in the [`Schematic_Diagram_for_Electrical_Design.png`](SchematicDiagram_for_Electrical_Design.png) . This circuit is the key circuit to control the on and off of the flywheel. 

For the testing, there are two testing pyton files. The [`static_launching_testing.py`](./static_launching_testing.py) is meant for the static target. Place the turtlebot directly to the target 4 to 8 centimeters away, then run the python file to check if the flywheel starts successfully, servo motor feeds one ball at a time and three balls in total, and the timing required is correct. 

For dynamic (moving) target, run [`dynamic_launching_testing.py`](./dynamic_launching_testing.py). Place the turtlebot directly to the target 4 to 8 centimeters away, then run the python file to check if the flywheel starts successfully, ultrasonic sensor detects the moving target in time, and servo motor feeds the ball immediately when there is an object between 17 to 23 centimeters. 

These two files are for testing only, for the final mission, files are to modify to fit the navigation and docking files such that the turtlebot can detect the station when it senses the Aruco, move to directly face the target, and then return to navigation mode once all balls are launched. 
