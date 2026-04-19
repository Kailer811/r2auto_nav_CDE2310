# 🤖 Autonomous Payload Delivery AMR (CDE2310)
[![ROS2 - Humble](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Hardware - TurtleBot3](https://img.shields.io/badge/Hardware-TurtleBot3_Burger-orange)](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)


## Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [System Architecture](#-system-architecture)
  - [Mechanical Subsystem](#1-mechanical-subsystem)
  - [Electrical Subsystem](#2-electrical-subsystem)
  - [Software Subsystem](#3-software-subsystem)
- [Team Members](#-team-members)

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
├── Electrical Design/                              # Electrical and Control Subsystems         
│
├── Mechanical Engineering/                         # Mechanical Subsystem                         
│
├── Software/                                       # Software and Algorithmic Subsystem (ROS2 Workspace)
│                     
├── End_User_documentation.pdf                      # End-user documentation and test reports
│
└── README.md
