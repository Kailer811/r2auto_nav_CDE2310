# 🤖 Autonomous Payload Delivery AMR (CDE2310)
[![ROS2 - Humble](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Hardware - TurtleBot3](https://img.shields.io/badge/Hardware-TurtleBot3_Burger-orange)](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)


## Table of Contents
- Project Overview
- Repository Structure
- System Architecture
  - Mechanical Subsystem
  - Electrical Subsystem
  - Software Subsystem
- Team Members

---

## Project Overview
The core objective of this system is to perform autonomous exploration, localization, and navigation within an unknown maze environment without a pre-built map. When the robot's vision subsystem detects a specific ArUco marker, the system automatically transitions into **Delivery Mode**. It precisely aligns with the target station and utilizes Finite State Machine (FSM) to trigger the launching mechanism, accurately deploying 3 ping pong balls. Upon completion, it automatically resumes the exploration mode.

**Key Features:**
- **Autonomous Exploration:** Real-time SLAM mapping and navigation using a 2D LiDAR and frontier based algorithms.
- **Vision Targeting:** ArUco marker recognition and relative pose estimation via an RPi Camera stream and OpenCV.
- **Payload Deployment:** A customized high speed flywheel launcher and servo motor feeder mechanism, controlled by independent hardware circuitry for precise timing control.

---

## Repository Structure

This project is organized into three main engineering modules:

```text
r2auto_nav_CDE2310/
│
├── Electrical Design                               # Electrical and Control Subsystems         
│
├── Mechanical Engineering                          # Mechanical Subsystem                         
│
├── Software                                        # Software and Algorithmic Subsystem (ROS2 Workspace)
│                     
├── End_User_Documentation.pdf                      # End-user documentation and test reports
│
└── README.md
```
---

## System Architecture

### 1. Mechanical Subsystem
Built upon the Turtlebot3 chassis, we design folloing 3D printed part to complete the mission and store maximum 9 ping pong balls:

- **Launcher Assembly:** Customize the bracket for DC motor (flywheel), payload reservoir, and servo motor bracket. 
- **Sensor Mount:** Design the bracket for ultrasonic sensor. 

### 2. Electrical Subsystem
The electrical system ensures the communication between sensing and control nodes. It also ensures the stability of launching:

- **Controllers:** Raspberry Pi 4 (High-level computing) + OpenCR 1.0 (Low-level actuation).
- **Motor Control:** Flywheel is controlled by the MOSFET circuit whose gate is connected to Raspbeery Pi GPIO. 

### 3. Software Subsystem
The software architecture is built on ROS2 Humble, utilizing a highly decoupled publish and subscribe modular design:

- **SLAM Node:** Integrate LiDAR and odometry data to construct real time occupancy grid map. 
- **Exploration Node:** Automatically generates waypoints based on map frontiers for continuous exploration.
- **Visio Node:** Rpi camera identifying the Aruco, and calculate the relative trasnlation vector
- **Mission Manager (FSM):** Transition between "Exploration", "Docking", and "Delivery".

---

## Team Member
- **Amber (YUCHEN):** Electrical subsystem design and launching testing.

- **Jon (Jonathan):** Mechanical subsystem design and mechanical assembly.

- **Ethan (Kai Ler):** Software subsystem design and docking mechanism.

- **Ram (Ramanathan):** Software subsystem design and exploration algorithm.
