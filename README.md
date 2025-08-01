# OpenCyberGlove Retarget Wrapper ROS2

![Allegro](media/allegro.webp)
![Inspire](media/inspire.webp)
![Ability](media/ability.webp)


## 1. Setup / Installation and Requirements

This package provides a ROS2 node for real-time robot hand retargeting using marker-based hand kinematics, leveraging the [dex-retargeting](https://github.com/dexsuite/dex-retargeting) library. Joint information is sourced from [open_cyber_glove_ros2](https://github.com/CyberOrigin2077/open_cyber_glove_ros2), and the retargeted robot joint positions (`qpos`) are published for downstream use. 

To control a real robot, you will need to convert the published `qpos` values into control signals compatible with your specific hardware. Example scripts for this conversion will be provided soon.

### Requirements
- ROS2 (tested with Jazzy)
- Python 3.7+
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) (and its dependencies)
- [sapien](https://sapien.ucsd.edu/) (for simulation visualization, tested with 3.0.0b1)
- numpy
- rclpy
- std_msgs
- visualization_msgs

#### Install Python dependencies

It is highly recommended to use a Python virtual environment for managing dependencies.

```bash
pip install numpy sapien
```

#### Install ROS2 dependencies
Make sure you have sourced your ROS2 environment and installed the required ROS2 packages:
```bash
sudo apt install ros-${ROS_DISTRO}-rclpy ros-${ROS_DISTRO}-std-msgs ros-${ROS_DISTRO}-visualization-msgs ros-${ROS_DISTRO}-pinocchio
```

#### Install dex-retargeting dependencies

Refer to [dex-retargeting](https://github.com/dexsuite/dex-retargeting). Download [dex-urdf](https://github.com/dexsuite/dex-urdf) and set the environment variable `DEX_URDF_PATH` to the root of the dex-urdf repository.

```bash
export DEX_URDF_PATH=/path/to/dex-urdf
```

#### Add your virtual environment to the Python path

If you are using a Python virtual environment, make sure its `site-packages` directory is included in your `PYTHONPATH` so ROS2 can find all dependencies.
```bash
export PYTHONPATH=/path/to/your/venv/site-packages/:$PYTHONPATH
```

## 2. How to Run the Node

Build the ROS2 workspace as usual:
```bash
colcon build --packages-select open_cyber_glove_retarget_ros2
source install/setup.bash
```

### Basic Usage

Run the retargeting node with the required arguments:

```bash
ros2 run open_cyber_glove_retarget_ros2 retarget_node --robot-name inspire --hand-type right
```

### Command Line Arguments

The node supports the following command line arguments:

- `--robot-name`: Robot name to use for retargeting (default: "inspire")
  - Available options: "inspire", "allegro", "shadow", "svh", "leap", "ability", "panda"
- `--hand-type`: Hand type (right or left) (default: "right")
- `--retargeting-type`: Retargeting algorithm type (default: "dexpilot")
  - Available options: "dexpilot", "vector", "position"
- `--sim-vis`: Enable simulation visualization window
- `--dual-hands`: Display both left and right hands simultaneously
- `--topic`: Topic to subscribe for hand kinematics markers (default: "/joints_position")
- `--disable-collision`: Disable collision detection to avoid STL warnings
- `--hide-joints`: Hide joint visualization spheres in the visualization window

### Examples

#### Single Hand Retargeting
```bash
ros2 run open_cyber_glove_retarget_ros2 retarget_node --robot-name inspire --hand-type right --sim-vis
```

#### Dual Hand Retargeting
```bash
ros2 run open_cyber_glove_retarget_ros2 retarget_node --robot-name inspire --dual-hands --sim-vis
```

#### Custom Topic
```bash
ros2 run open_cyber_glove_retarget_ros2 retarget_node --robot-name inspire --hand-type right --topic /custom/hand_markers
```

## 3. Subscribed and Published Topics

### Subscribed Topics
- `/joints_position` (`visualization_msgs/MarkerArray`):
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

The node supports simultaneous retargeting of both left and right hands. When the `--dual-hands` parameter is used, the node creates a unified dual-hand retargeting instance that:

- Loads both left and right hand models in the same SAPIEN scene
- Processes both hands' marker data simultaneously
- Applies hand joint scaling to both hands independently
- Visualizes both hands in a single window with optimized camera view
- Publishes to both `/left_retargeted_qpos` and `/right_retargeted_qpos` topics

## 5. Hand Joint Scaling

Both single-hand and dual-hand modes support hand joint scaling to fine-tune the retargeting effect:

- Global scaling factor: 1.3 (scales the entire hand)
- Individual finger scaling:
  - Thumb: 1.0
  - Index: 1.0
  - Middle: 1.0
  - Ring: 1.0
  - Pinky: 1.0

These scaling factors can be adjusted in the code to achieve different retargeting effects. The scaling is applied to the joint positions relative to the wrist, preserving the wrist position while scaling the finger movements.

## 6. Visualization

When the `--sim-vis` parameter is used, the node creates a SAPIEN visualization window showing:
- The robot hand model(s)
- The human hand joint positions
- The retargeted robot joint positions

In dual-hand mode, both hands are visualized in the same window with an optimized camera view.

### Controlling Joint Visualization

By default, both the human hand and robot hand joints are visualized as colored spheres. If you want to hide these joint spheres and fingertip indicators (for clearer visualization or performance reasons), you can use the `--hide-joints` parameter:

```bash
ros2 run open_cyber_glove_retarget_ros2 retarget_node --robot-name inspire --dual-hands --sim-vis --hide-joints
```

This will still visualize the hand models but without the additional joint and fingertip spheres, resulting in a cleaner visualization.

## 7. Troubleshooting

### STL File Warnings
If you see warnings about "loading multiple convex collision meshes from STL file", you can use the `--disable-collision` parameter to disable collision detection and suppress these warnings.

### No Marker Data Received
Make sure that the topic you're subscribing to is being published. You can check this with:
```bash
ros2 topic info /joints_position
```

### Integration with Glove Visualizer
This node is designed to work with the `glove_visualizer` package, which publishes hand kinematics data to the `/joints_position` topic. Make sure the `hand_display_node` from that package is running.

## 8. Additional Notes
- If you want to enable simulation visualization, make sure you have a working SAPIEN installation and use the `--sim-vis` flag.
- For more details on configuration files and supported robots, refer to the [dex-retargeting documentation](https://github.com/dexsuite/dex-retargeting).
- Supporting hands roadmap:
  - [ ] Roboterax
  - [ ] RO Hand

## 9. Acknowledgements

This project heavily borrows code and concepts from the [dex-retargeting](https://github.com/dexsuite/dex-retargeting) project. We are grateful to the original authors for their excellent work in real-time hand motion retargeting. The core retargeting algorithms, optimization methods, and much of the visualization code are adapted from their implementation. Our modifications focus on ROS2 integration with OpenCyberGlove while maintaining compatibility with their proven retargeting approach.


---
License: BSD-3
