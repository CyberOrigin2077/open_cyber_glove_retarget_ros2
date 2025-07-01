# cyber_retarget_ros2_py

## 1. Setup / Installation and Requirements

This package provides a ROS2 node for real-time robot hand retargeting using marker-based hand kinematics and the [dex-realtime-retargeting](https://github.com/CyberOrigin2077/dex_realtime_retargeting) library.

### Requirements
- ROS2 (tested with Foxy/Galactic/Humble)
- Python 3.7+
- [dex-realtime-retargeting](https://github.com/CyberOrigin2077/dex_realtime_retargeting) (and its dependencies)
- [sapien](https://sapien.ucsd.edu/) (for simulation visualization, optional)
- numpy
- rclpy
- std_msgs
- visualization_msgs

#### Install Python dependencies
```bash
pip install numpy sapien
```

#### Install ROS2 dependencies
Make sure you have sourced your ROS2 environment and installed the required ROS2 packages:
```bash
sudo apt install ros-${ROS_DISTRO}-rclpy ros-${ROS_DISTRO}-std-msgs ros-${ROS_DISTRO}-visualization-msgs ros-${ROS_DISTRO}-pinocchio
```

#### Install dex-realtime-retargeting
Clone and install [dex-realtime-retargeting](https://github.com/CyberOrigin2077/dex_realtime_retargeting) and its dependencies. Set the environment variable `DEX_RETARGETING_PATH` to the root of the dex-realtime-retargeting repository:

```bash
export DEX_RETARGETING_PATH=/path/to/dex_realtime_retargeting
```

## 2. How to Run the Node

Build the ROS2 workspace as usual:
```bash
colcon build --packages-select cyber_retarget_ros2_py
source install/setup.bash
```

Run the retargeting node with the required arguments:

```bash
ros2 run cyber_retarget_ros2_py retarget_node \
  --finger-config ${DEX_RETARGETING_PATH}/vector_retargeting/configs/cyberglove_config.yaml \
  --robot-config ${DEX_RETARGETING_PATH}/dex-urdf/robots/hands/inspire_hand/
```

- `--finger-config`: Path to the finger configuration YAML file (from dex-realtime-retargeting)
- `--robot-config`: Path to the robot configuration directory (URDFs, etc.)
- `--robot-name`: (Optional) Robot name, default is `inspire`

**Note:** The node expects the environment variable `DEX_RETARGETING_PATH` to be set so it can import the dex-realtime-retargeting Python modules.

## 3. Subscribed and Published Topics

### Subscribed Topics
- `/hand_kinematics_markers` (`visualization_msgs/MarkerArray`):
  - Input marker array containing the hand kinematics (joint positions and orientations).

### Published Topics
- `/retargeted_qpos` (`std_msgs/Float32MultiArray`):
  - Output array containing the retargeted robot joint positions (qpos) as computed by the retargeting pipeline.

## 4. Additional Notes
- If you want to enable simulation visualization, make sure you have a working SAPIEN installation and set `sim_vis=True` in the code or modify the launch command accordingly.
- For more details on configuration files and supported robots, refer to the [dex-realtime-retargeting documentation](https://github.com/CyberOrigin2077/dex_realtime_retargeting).

---
Maintainer: Jessey (<jessey.li@cyberorigin.ai>)
License: BSD-3