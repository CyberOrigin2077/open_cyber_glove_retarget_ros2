# cyber_retarget_ros2_py

## 1. Setup / Installation and Requirements

This package provides a ROS2 node for real-time robot hand retargeting using marker-based hand kinematics and the [dex-realtime-retargeting](https://github.com/CyberOrigin2077/dex_realtime_retargeting) library.

### Requirements
- ROS2 (tested with Foxy/Galactic/Humble/Jazzy)
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

### Basic Usage

Run the retargeting node with the required arguments:

```bash
ros2 run cyber_retarget_ros2_py retarget_node --robot-name roboterax --hand-type right
```

### Command Line Arguments

The node supports the following command line arguments:

- `--robot-name`: Robot name to use for retargeting (default: "inspire")
  - Available options: "inspire", "allegro", "shadow", "svh", "leap", "ability", "panda", "roboterax"
- `--hand-type`: Hand type (right or left) (default: "right")
- `--retargeting-type`: Retargeting algorithm type (default: "dexpilot")
  - Available options: "dexpilot", "vector", "position"
- `--sim-vis`: Enable simulation visualization window
- `--dual-hands`: Display both left and right hands simultaneously
- `--topic`: Topic to subscribe for hand kinematics markers (default: "/hand_kinematics_markers")
- `--disable-collision`: Disable collision detection to avoid STL warnings

### Examples

#### Single Hand Retargeting
```bash
ros2 run cyber_retarget_ros2_py retarget_node --robot-name roboterax --hand-type right --sim-vis
```

#### Dual Hand Retargeting
```bash
ros2 run cyber_retarget_ros2_py retarget_node --robot-name roboterax --hand-type right --sim-vis --dual-hands
```

#### Custom Topic
```bash
ros2 run cyber_retarget_ros2_py retarget_node --robot-name roboterax --hand-type right --topic /custom/hand_markers
```

## 3. Subscribed and Published Topics

### Subscribed Topics
- `/hand_kinematics_markers` (`visualization_msgs/MarkerArray`):
  - Input marker array containing the hand kinematics (joint positions and orientations).
  - The node filters markers based on the namespace in the marker message to determine which hand they belong to.

### Published Topics
- `/retargeted_qpos` (`std_msgs/Float32MultiArray`):
  - Output array containing the retargeted robot joint positions (qpos) as computed by the retargeting pipeline.
- `/left_retargeted_qpos` (`std_msgs/Float32MultiArray`):
  - Output for left hand when using dual-hands mode.
- `/right_retargeted_qpos` (`std_msgs/Float32MultiArray`):
  - Output for right hand when using dual-hands mode.

## 4. Dual-Hand Mode

The node supports simultaneous retargeting of both left and right hands. When the `--dual-hands` parameter is used, the node creates two separate instances:
- One for the left hand
- One for the right hand

Each instance subscribes to the same topic (`/hand_kinematics_markers` by default) but filters the markers based on their namespace to get the correct hand data.

## 5. Visualization

When the `--sim-vis` parameter is used, the node creates a SAPIEN visualization window showing:
- The robot hand model
- The human hand joint positions
- The retargeted robot joint positions

In dual-hand mode, both hands are visualized in the same window.

## 6. Troubleshooting

### STL File Warnings
If you see warnings about "loading multiple convex collision meshes from STL file", you can use the `--disable-collision` parameter to disable collision detection and suppress these warnings.

### No Marker Data Received
Make sure that the topic you're subscribing to is being published. You can check this with:
```bash
ros2 topic info /hand_kinematics_markers
```

### Integration with Glove Visualizer
This node is designed to work with the `glove_visualizer` package, which publishes hand kinematics data to the `/hand_kinematics_markers` topic. Make sure the `hand_display_node` from that package is running:

## 4. Additional Notes
- If you want to enable simulation visualization, make sure you have a working SAPIEN installation and set `sim_vis=True` in the code or modify the launch command accordingly.
- For more details on configuration files and supported robots, refer to the [dex-realtime-retargeting documentation](https://github.com/CyberOrigin2077/dex_realtime_retargeting).

---
Maintainer: Jessey (<jessey.li@cyberorigin.ai>)
License: BSD-3