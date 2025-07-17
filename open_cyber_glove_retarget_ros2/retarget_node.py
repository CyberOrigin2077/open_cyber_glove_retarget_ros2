#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
import numpy as np
import argparse
from loguru import logger
import time
from pathlib import Path
import multiprocessing
from queue import Empty

# Add debug information, print Python path
import sys

# Filter SAPIEN STL warning messages
import warnings
import logging
# Configure log filter to filter out specific SAPIEN warnings
logging.getLogger('sapien').setLevel(logging.ERROR)
# Ignore specific warnings
warnings.filterwarnings("ignore", message="loading multiple convex collision meshes from STL file")

# Import constants module directly from local
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    ROBOT_NAME_MAP,
)

# Print all members in RobotName enum
print("All members in RobotName enum:")
print([name for name in RobotName.__members__])

import os
DEX_URDF_PATH = os.getenv("DEX_URDF_PATH", "")

from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

INSPIRE_JOINT_ORDER = [
    'pinky_proximal_joint',
    'ring_proximal_joint', 
    'middle_proximal_joint',
    'index_proximal_joint',
    'thumb_proximal_pitch_joint',
    'thumb_proximal_yaw_joint'
]


ROBOTERAX_RIGHT_JOINT_ORDER = [
    'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2', 
    'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
    'right_hand_mid_joint1', 'right_hand_mid_joint2',
    'right_hand_ring_joint1', 'right_hand_ring_joint2',
    'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
]

ROBOTERAX_LEFT_JOINT_ORDER = [
    'left_hand_thumb_bend_joint', 'left_hand_thumb_rota_joint1', 'left_hand_thumb_rota_joint2', 
    'left_hand_index_bend_joint', 'left_hand_index_joint1', 'left_hand_index_joint2',
    'left_hand_mid_joint1', 'left_hand_mid_joint2',
    'left_hand_ring_joint1', 'left_hand_ring_joint2',
    'left_hand_pinky_joint1', 'left_hand_pinky_joint2'
]


ROBOTERAX_JOINT_ORDER = ROBOTERAX_RIGHT_JOINT_ORDER


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)



def load_finger_config():
    """
    Load finger configuration information
    
    Returns:
        Dictionary containing joint mapping and visualization settings
    """
    # Default configuration
    config = {
        "joint_mapping": {
            "thumb_end": 3,
            "index_end": 7,
            "middle_end": 11,
            "ring_end": 15,
            "pinky_end": 19
        },
        "visualization": {
            "scale_factor": 0.8,
            "joint_radius": 0.01,
            "fingertip_radius": 0.005,
            "position_offset": [0.0, 0.0, 0.0]
        }
    }
    return config


# Function to visualize hand joints in 3D space using SAPIEN
def visualize_hand_joints(scene, joint_positions, fingertip_positions, hand_type="Right", joint_orientations=None, finger_config=None, position_offset=None, name_prefix="", show_joints=True):
    """
    Visualize hand joints and fingertips in 3D space using SAPIEN
    
    Args:
        scene: SAPIEN scene
        joint_positions: Array of joint positions [20, 3]
        fingertip_positions: Dictionary with fingertip positions
        hand_type: "Left" or "Right"
        joint_orientations: Optional array of joint orientations as quaternions [20, 4]
        finger_config: Configuration dictionary containing joint mappings and visualization settings
        position_offset: Optional offset to apply to visualization positions [x, y, z]
        name_prefix: Prefix to add to visualization object names to differentiate left/right hands
        show_joints: Whether to show joint visualization spheres (default: True)
    """
    # Load configuration if not provided
    if finger_config is None:
        finger_config = load_finger_config()
    
    # Get visualization settings from config
    viz_config = finger_config.get("visualization", {})
    
    # Get joint mappings from config
    joint_mapping = finger_config["joint_mapping"]
    
    # Define finger chains (only used for fingertip colors)
    finger_chains = {
        'thumb': [0, 1, 2, joint_mapping["thumb_end"]],
        'index': [0, 4, 5, 6, joint_mapping["index_end"]],
        'middle': [0, 8, 9, 10, joint_mapping["middle_end"]],
        'ring': [0, 12, 13, 14, joint_mapping["ring_end"]],
        'pinky': [0, 16, 17, 18, joint_mapping["pinky_end"]],
    }
    
    # Finger colors for visualization
    finger_colors = {
        'thumb': [1.0, 0.2, 0.2],   # Thumb - red
        'index': [0.2, 1.0, 0.2],   # Index - green
        'middle': [0.2, 0.2, 1.0],  # Middle - blue
        'ring': [1.0, 1.0, 0.2],    # Ring - yellow
        'pinky': [1.0, 0.2, 1.0],   # Pinky - purple
    }
    
    # Get visualization settings from config
    scale_factor = viz_config.get("scale_factor", 1.1)
    joint_radius = viz_config.get("joint_radius", 0.01)
    fingertip_radius = viz_config.get("fingertip_radius", 0.005)
    default_offset = viz_config.get("position_offset", [0.25, 0, 0.25])
    
    # Use provided position_offset if specified, otherwise use default from config
    if position_offset is not None:
        pos_offset = np.array(position_offset)
    else:
        pos_offset = np.array(default_offset)
    
    # Keep track of visualization objects to allow returning them
    vis_objects = []
    
    # Add a prefix to each visualization object, if not provided use default value
    # Ensure the prefix is a valid string
    if name_prefix is None:
        name_prefix = ""
    
    # Only clear visualization objects for current hand type (if there's a prefix)
    if name_prefix:
        for actor in scene.get_all_actors():
            if actor.name.startswith(name_prefix):
                try:
                    scene.remove_actor(actor)
                except Exception as e:
                    logger.debug(f"Failed to remove actor {actor.name}: {e}")
    
    # Visualize joints
    if show_joints:
        for i in range(len(joint_positions)):
            pos = joint_positions[i] * scale_factor + pos_offset
            builder = scene.create_actor_builder()
            # Create render material with color
            material = sapien.render.RenderMaterial()
            
            # Set different colors for different finger joints
            # Determine which finger the joint belongs to based on index
            if i == 0:  # Wrist
                material.base_color = [0.5, 0.5, 0.5, 1.0]  # Gray
            elif i in [1, 2, 3, 4]:  # Thumb joints
                material.base_color = [1.0, 0.2, 0.2, 1.0]  # Red
            elif i in [5, 6, 7, 8]:  # Index joints
                material.base_color = [0.2, 1.0, 0.2, 1.0]  # Green
            elif i in [9, 10, 11, 12]:  # Middle joints
                material.base_color = [0.2, 0.2, 1.0, 1.0]  # Blue
            elif i in [13, 14, 15, 16]:  # Ring joints
                material.base_color = [1.0, 1.0, 0.2, 1.0]  # Yellow
            elif i in [17, 18, 19, 20]:  # Pinky joints
                material.base_color = [1.0, 0.2, 1.0, 1.0]  # Purple
            else:
                material.base_color = [0.7, 0.7, 0.7, 1.0]  # Default gray
            
            # Adjust joint size to make fingertips more visible
            radius = joint_radius
                
            builder.add_sphere_visual(radius=radius, material=material)
            joint = builder.build(name=f"{name_prefix}joint_{i}")
            joint.set_pose(sapien.Pose(pos))
            vis_objects.append(joint)
    
        # Visualize fingertips
        for finger_name, chain in finger_chains.items():
            if finger_name in fingertip_positions:
                tip_pos = fingertip_positions[finger_name] * scale_factor + pos_offset
                builder = scene.create_actor_builder()
                # Create render material with color
                material = sapien.render.RenderMaterial()
                material.base_color = finger_colors[finger_name] + [1.0]
                builder.add_sphere_visual(radius=fingertip_radius, material=material)
                fingertip = builder.build(name=f"{name_prefix}fingertip_{finger_name}")
                fingertip.set_pose(sapien.Pose(tip_pos))
                vis_objects.append(fingertip)
    
    # Return created visualization objects to allow removing them later
    return vis_objects


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def process_hand_marker_array(msg, hand_type="left", glove_name="cyber"):
    if glove_name == "cyber":
        joints_number = 21
    else:
        joints_number = 20

    try:
        # Create mapping from marker ID to marker object
        markers_by_id = {marker.id: marker for marker in msg.markers}
        
        # Check if wrist marker exists
        if 0 not in markers_by_id:
            logger.warning(f"No wrist marker (ID 0) found in {hand_type} hand message")
            return None, None
        
        # Get wrist marker position and orientation
        wrist_marker = markers_by_id[0]
        wrist_pos = np.array([
            wrist_marker.pose.position.x,
            wrist_marker.pose.position.y,
            wrist_marker.pose.position.z
        ])
        
        wrist_ori = np.array([
            wrist_marker.pose.orientation.x,
            wrist_marker.pose.orientation.y,
            wrist_marker.pose.orientation.z,
            wrist_marker.pose.orientation.w
        ])
        
        # Initialize position array - for storing world coordinates of all joints
        keypoint_3d_array = np.zeros((joints_number, 3))
        keypoint_3d_array[0] = wrist_pos  # Set wrist position
        
        # Initialize orientation data array
        orientations_data = np.zeros((joints_number, 4))
        orientations_data[0] = wrist_ori  # Set wrist orientation
        
        # Process all other joint markers
        for joint_id in range(1, joints_number):
            if joint_id not in markers_by_id:
                # Use default values if marker is missing for a joint
                continue
            
            # Get joint marker position and orientation
            marker = markers_by_id[joint_id]
            keypoint_3d_array[joint_id] = [
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ]
            
            orientations_data[joint_id] = [
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w
            ]

            # transformed_pos, transformed_ori = transform_pose_to_wrist_frame(
            #     wrist_pos, wrist_ori, joint_pos, joint_ori
            # )
            
        # 1. Make all joint coordinates relative to wrist
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        
        # 2. Estimate wrist rotation frame
        mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
        
        # 3. Apply coordinate system transformation
        operator2mano = OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else OPERATOR2MANO_LEFT
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano
        
        # 4. Set default values for orientation data
        orientations_data = np.zeros((joints_number, 4))
        orientations_data[:, 3] = 1.0  # Set to unit quaternion

        
        return joint_pos, orientations_data
        
    except Exception as e:
        print(f"Error processing {hand_type} hand data: {e}")
        return None, None


class HandRetargetNode(Node):
    def __init__(
        self,
        robot_name="inspire",
        hand_type="right",
        retargeting_type="dexpilot",
        sim_vis=False,
        disable_collision=False,
        is_second_hand=False,  # Whether this is the second hand
        topic=None,  # Custom topic name
        show_joints=True,  # Whether to show joint visualization
    ):
        # Create different node name for second hand
        node_name = "hand_retarget_node"
        if is_second_hand:
            node_name = f"hand_retarget_node_{hand_type}"
            
        super().__init__(node_name)

        self.hand_type = hand_type
        self.robot_name = robot_name
        self.disable_collision = disable_collision
        self.is_second_hand = is_second_hand
        self.sim_vis = sim_vis
        self.show_joints = show_joints  # Store visualization control parameter
        
        # Convert strings to enum values
        robot_name_enum = RobotName[robot_name]
        hand_type_enum = HandType[hand_type]
        retargeting_type_enum = RetargetingType[retargeting_type]
        
        self.get_logger().info(
            f"Using robot: {robot_name_enum}, hand type: {hand_type_enum}, retargeting type: {retargeting_type_enum}"
        )
        
        # Create message queue
        self.marker_queue = multiprocessing.Queue(maxsize=1000)
        
        robot_config_path = os.path.join(DEX_URDF_PATH, "robots/hands")
        
        # If directory doesn't exist, try other paths
        if not os.path.exists(robot_config_path):
            self.get_logger().error(f"Cannot find robot configuration directory: {robot_config_path}")
            raise ValueError(f"Cannot find robot configuration directory")
        
        self.get_logger().info(f"Using robot configuration path: {robot_config_path}")
        
        # Set default URDF directory
        RetargetingConfig.set_default_urdf_dir(str(robot_config_path))
        
        # Get configuration file path using enum values
        config_path = get_default_config_path(
            robot_name_enum, retargeting_type_enum, hand_type_enum
        )
        self.get_logger().info(f"Using configuration file: {config_path}")
        
        # Start retargeting process
        self.consumer_process = multiprocessing.Process(
            target=self.start_retargeting,
            args=(
                self.marker_queue,
                str(robot_config_path),
                str(config_path),
                self.hand_type,
                disable_collision,
                self.sim_vis,
            ),
        )
        self.consumer_process.daemon = True
        self.consumer_process.start()
        
        # Create subscriber for hand kinematics markers
        marker_topic_name = topic if topic else "/joints_position"
            
        self.marker_sub = self.create_subscription(
            MarkerArray, marker_topic_name, self.marker_callback, 1
        )
        self.get_logger().info(f"Subscribed to marker topic: {marker_topic_name}")

        # Create publisher for retargeted joint positions
        pub_topic = "/retargeted_qpos"
        if hand_type.lower() == "left":
            pub_topic = "/left_retargeted_qpos"
        elif hand_type.lower() == "right":
            pub_topic = "/right_retargeted_qpos"
            
        self.retargeted_pub = self.create_publisher(
            Float32MultiArray, pub_topic, 1
        )
        self.get_logger().info("Hand retargeting node initialized")

    def marker_callback(self, msg):
        # Check if marker array is empty
        if not msg.markers:
            self.get_logger().warn("Received empty marker array")
            return

        # Filter markers based on hand type
        filtered_markers = MarkerArray()
        
        # Get the namespace from the first marker
        if msg.markers and hasattr(msg.markers[0], 'ns'):
            namespace = msg.markers[0].ns
            
            # Check if this is the correct hand type
            is_left_hand = "left" in namespace.lower()
            is_right_hand = "right" in namespace.lower()
            
            # Only process markers for the correct hand
            if (self.hand_type.lower() == "left" and is_left_hand) or (self.hand_type.lower() == "right" and is_right_hand):
                filtered_markers = msg
                self.get_logger().debug(f"Processing {self.hand_type} hand markers, namespace: {namespace}")
            else:
                # Skip markers for the wrong hand
                return
        else:
            # If no namespace, process all markers
            filtered_markers = msg
            
        # Put marker data into queue
        self.marker_queue.put(filtered_markers)
        self.get_logger().debug(
            f"Marker data put into queue, current queue size: {self.marker_queue.qsize()}"
        )

    def start_retargeting(self, queue, robot_dir, config_path, hand_type, disable_collision=False, sim_vis=False):
        """
        Run retargeting process in a separate process
        
        Args:
            queue: Queue for receiving marker data
            robot_dir: Robot configuration directory
            config_path: Retargeting configuration file path
            hand_type: Hand type ("left" or "right")
            disable_collision: Whether to disable collision detection to avoid STL warnings
            sim_vis: Whether to enable simulation visualization
        """
        logger.info(f"Starting retargeting process, config file: {config_path}, hand type: {hand_type}")
        
        # Set URDF directory
        robot_dir = Path(robot_dir)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        try:
            # Load configuration and build retargeter
            config = RetargetingConfig.load_from_file(config_path)
            
            # Ensure type is lowercase
            if hasattr(config, "type") and isinstance(config.type, str):
                config.type = config.type.lower()
                
            # Build retargeter
            retargeting = config.build()
            logger.info("Successfully built retargeter")
            
        except Exception as e:
            logger.error(f"Failed to load config or build retargeter: {e}")
            raise
            
        # Set up SAPIEN renderer
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        except Exception as e:
            logger.warning(f"Error setting SAPIEN renderer, this might not affect functionality: {e}")
            
        # Create scene
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

        # Set up lighting
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

        # Set up camera
        cam = scene.add_camera(
            name="main_camera", width=600, height=600, fovy=1, near=0.1, far=10
        )
        cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

        # Set up viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
        
        # If sim_vis is disabled, hide the viewer window
        if not sim_vis:
            viewer.hide_window()

        # Load robot
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        logger.info(f"Using URDF file: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"URDF file not found: {filepath}")
            raise FileNotFoundError(f"URDF file not found: {filepath}")
            
        # Get robot name
        robot_name = filepath.stem
        logger.info(f"Robot name obtained from file path: {robot_name}")
        loader.load_multiple_collisions_from_file = True

        # Set scale based on robot type
        if "ability" in robot_name:
            loader.scale = 1.5
        elif "dclaw" in robot_name:
            loader.scale = 1.25
        elif "allegro" in robot_name:
            loader.scale = 1.4
        elif "shadow" in robot_name:
            loader.scale = 0.9
        elif "bhand" in robot_name:
            loader.scale = 1.5
        elif "leap" in robot_name:
            loader.scale = 1.4
        elif "svh" in robot_name:
            loader.scale = 1.5
        elif "roboterax" in robot_name:
            loader.scale = 1.0  # Adjust based on actual Roboterax size

        # Special file path handling
        if "glb" not in robot_name and not robot_name.startswith("inspire"):
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
            logger.info(f"Applying _glb suffix: {filepath}")
        else:
            filepath = str(filepath)
            logger.info(f"Using original file path: {filepath}")
            
        # Load robot
        robot = loader.load(filepath)

        # Set pose based on robot type
        if "ability" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "shadow" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "dclaw" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "allegro" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.05]))
        elif "bhand" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "leap" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "svh" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.13]))
        elif "roboterax" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, 0]))  # Adjust based on actual Roboterax size
        else:
            robot.set_pose(sapien.Pose([0, 0, 0]))  # Default

        # Get joint names
        retargeting_joint_names = retargeting.joint_names
            
        # Get SAPIEN joint names
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        
        # Set mapping from retargeting to SAPIEN
        retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)
        
        retargeting_to_robot = np.array(
            [retargeting_joint_names.index(name) for name in retargeting_joint_names]
        ).astype(int)
        print(f"retargeting_to_robot: {retargeting_to_robot}")

        # Create ROS2 node for publishing retargeted joint positions
        ros_context = rclpy.Context()
        ros_context.init()
        ros_node = rclpy.create_node("retargeting_publisher", context=ros_context)
        retargeted_pub = ros_node.create_publisher(Float32MultiArray, "/retargeted_qpos", 1)
        
        # Main loop
        while True:
            try:
                # Get marker data from queue, timeout 5 seconds
                msg = queue.get(timeout=5)
                
                # Process marker data
                joint_pos, joint_ori = process_hand_marker_array(msg, hand_type=hand_type)
                
                # Scale and offset joint positions to fine-tune retargeting effect
                if joint_pos is not None:
                    # Define scale factors and offsets (can be adjusted as needed)
                    scale_factors = {
                        'global': 1.3,  # Global scale factor
                        'thumb': 1.0,   # Thumb scale factor
                        'index': 1.0,   # Index scale factor
                        'middle': 1.0,  # Middle scale factor
                        'ring': 1.0,    # Ring scale factor
                        'pinky': 1.0  # Pinky scale factor
                    }
                    
                    # Define joint groups
                    joint_groups = {
                        'thumb': [1, 2, 3, 4],    # Thumb joint indices
                        'index': [5, 6, 7, 8],    # Index joint indices
                        'middle': [9, 10, 11, 12], # Middle joint indices
                        'ring': [13, 14, 15, 16],  # Ring joint indices
                        'pinky': [17, 18, 19, 20]  # Pinky joint indices
                    }
                    
                    # Global scaling
                    joint_pos = joint_pos * scale_factors['global']
                    
                    # Scale each finger individually
                    for finger, indices in joint_groups.items():
                        # Ensure indices are within valid range
                        valid_indices = [idx for idx in indices if idx < joint_pos.shape[0]]
                        if valid_indices:
                            # Get wrist position as reference point
                            wrist_pos = joint_pos[0] if joint_pos.shape[0] > 0 else np.zeros(3)
                            
                            for idx in valid_indices:
                                # Calculate relative position
                                rel_pos = joint_pos[idx] - wrist_pos
                                # Apply scaling
                                scaled_rel_pos = rel_pos * scale_factors[finger]
                                # Update joint position
                                joint_pos[idx] = wrist_pos + scaled_rel_pos
                
                # Use retargeting logic to process joint positions
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                qpos = retargeting.retarget(ref_value)
                
                # Set robot pose
                robot.set_qpos(qpos[retargeting_to_sapien])
                
                # Get robot finger joint positions (through forward kinematics)
                robot_joints = robot.get_active_joints()
                robot_joint_positions = []
                robot_joint_names = []
                
                # Collect robot joint positions and names
                for joint in robot_joints:
                    # Get position of joint in world coordinates
                    link = joint.get_child_link()
                    if link:
                        pos = link.get_pose().p
                        robot_joint_positions.append(pos)
                        robot_joint_names.append(joint.get_name())
                
                # Visualize finger joints
                # Get palm position and finger joint positions
                if joint_pos is not None and len(joint_pos) > 0:
                    # Create fingertip position dictionary
                    fingertip_positions = {}
                    finger_config = load_finger_config()
                    joint_mapping = finger_config["joint_mapping"]
                    
                    # If enough joint points, add fingertip positions
                    if len(joint_pos) > max(joint_mapping.values()):
                        fingertip_positions = {
                            'thumb': joint_pos[joint_mapping["thumb_end"]],
                            'index': joint_pos[joint_mapping["index_end"]],
                            'middle': joint_pos[joint_mapping["middle_end"]],
                            'ring': joint_pos[joint_mapping["ring_end"]],
                            'pinky': joint_pos[joint_mapping["pinky_end"]]
                        }
                    
                    # Visualize human hand joint points (input)
                    name_prefix = f"{hand_type}_human_hand_"
                    visualize_hand_joints(
                        scene, 
                        joint_pos, 
                        fingertip_positions, 
                        hand_type=hand_type,
                        joint_orientations=joint_ori if 'joint_ori' in locals() else None,
                        finger_config=finger_config,
                        name_prefix=name_prefix,
                        position_offset=[0, 0, 0],  # Shift right a bit to avoid overlap with robot hand
                        show_joints=self.show_joints  # Pass the visualization control parameter
                    )
                    
                    # Visualize robot hand joint points (output)
                    if robot_joint_positions:
                        # Convert to numpy array
                        robot_joints_np = np.array(robot_joint_positions)
                        
                        # Create robot fingertip position dictionary (simplified, for demonstration)
                        robot_fingertips = {}
                        
                        # Try to find robot hand fingertip joints
                        for i, name in enumerate(robot_joint_names):
                            if "thumb" in name.lower() and "tip" in name.lower():
                                robot_fingertips["thumb"] = robot_joint_positions[i]
                            elif "index" in name.lower() and "tip" in name.lower():
                                robot_fingertips["index"] = robot_joint_positions[i]
                            elif "middle" in name.lower() and "tip" in name.lower():
                                robot_fingertips["middle"] = robot_joint_positions[i]
                            elif "ring" in name.lower() and "tip" in name.lower():
                                robot_fingertips["ring"] = robot_joint_positions[i]
                            elif "pinky" in name.lower() and "tip" in name.lower():
                                robot_fingertips["pinky"] = robot_joint_positions[i]
                        
                        # Visualize robot hand joints
                        robot_name_prefix = f"{hand_type}_robot_hand_"
                        visualize_hand_joints(
                            scene, 
                            robot_joints_np, 
                            robot_fingertips, 
                            hand_type=hand_type,
                            finger_config=finger_config,
                            name_prefix=robot_name_prefix,
                            position_offset=[0, 0, 0],  # No offset, use actual robot position
                            show_joints=self.show_joints  # Pass the visualization control parameter
                        )

                # Render scene
                for _ in range(2):
                    viewer.render()
                
                # Publish retargeted joint positions
                retargeted_msg = Float32MultiArray()
                retargeted_msg.data = qpos[retargeting_to_robot].flatten().tolist()
                retargeted_pub.publish(retargeted_msg)

                # Process ROS2 events
                rclpy.spin_once(ros_node, timeout_sec=0.001)
                
            except Empty:
                logger.warning("No marker data received within 5 seconds")
                continue
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, exiting retargeting process")
                break
            except Exception as e:
                # Log error but continue, unless it's a fatal error
                logger.error(f"Error processing marker data: {e}")
                if isinstance(e, (ValueError, AttributeError, IndexError, TypeError)):
                    # These errors might be temporary, continue
                    continue
                else:
                    # Other errors might be more severe, raise exception
                    logger.critical(f"Encountered fatal error, exiting process: {e}")
                    raise
        
        # Clean up resources
        ros_node.destroy_node()
        ros_context.shutdown()

    def destroy_node(self):
        # Terminate retargeting process
        if hasattr(self, "consumer_process") and self.consumer_process.is_alive():
            self.consumer_process.terminate()
            self.consumer_process.join(timeout=1.0)
            self.get_logger().info("Retargeting process terminated")

        super().destroy_node()


class DualHandRetargetNode(Node):
    def __init__(
        self,
        robot_name="inspire",
        retargeting_type="dexpilot",
        sim_vis=False,
        disable_collision=False,
        topic=None,  # Custom topic name
        show_joints=True,  # Whether to show joint visualization
    ):
        super().__init__("dual_hand_retarget_node")
        
        self.robot_name = robot_name
        self.disable_collision = disable_collision
        self.sim_vis = sim_vis
        self.show_joints = show_joints  # Store the visualization control parameter
        
        # Convert strings to enum values
        robot_name_enum = RobotName[robot_name]
        left_hand_enum = HandType["left"]
        right_hand_enum = HandType["right"]
        retargeting_type_enum = RetargetingType[retargeting_type]
        
        self.get_logger().info(
            f"Using robot: {robot_name_enum}, dual hands mode with retargeting type: {retargeting_type_enum}"
        )
        
        # Create message queues for both hands
        self.left_marker_queue = multiprocessing.Queue(maxsize=1000)
        self.right_marker_queue = multiprocessing.Queue(maxsize=1000)
        
        # Find robot configuration directory
        robot_config_path = os.path.join(DEX_URDF_PATH, "robots/hands")
        
        # If directory doesn't exist, try other paths
        if not robot_config_path.exists():
            self.get_logger().error(f"Cannot find robot configuration directory: {robot_config_path}")
            raise ValueError(f"Cannot find robot configuration directory")
        
        self.get_logger().info(f"Using robot configuration path: {robot_config_path}")
        
        # Set default URDF directory
        RetargetingConfig.set_default_urdf_dir(str(robot_config_path))
        
        # Get configuration file paths for left and right hands
        left_config_path = get_default_config_path(
            robot_name_enum, retargeting_type_enum, left_hand_enum
        )
        right_config_path = get_default_config_path(
            robot_name_enum, retargeting_type_enum, right_hand_enum
        )
        self.get_logger().info(f"Using left hand configuration file: {left_config_path}")
        self.get_logger().info(f"Using right hand configuration file: {right_config_path}")
        
        # Start dual retargeting process in a single process
        self.consumer_process = multiprocessing.Process(
            target=self.start_dual_retargeting,
            args=(
                self.left_marker_queue,
                self.right_marker_queue,
                str(robot_config_path),
                str(left_config_path),
                str(right_config_path),
                self.disable_collision,
                self.sim_vis,
            ),
        )
        self.consumer_process.daemon = True
        self.consumer_process.start()
        
        # Create subscriber for hand kinematics markers
        marker_topic_name = topic if topic else "/joints_position"
            
        self.marker_sub = self.create_subscription(
            MarkerArray, marker_topic_name, self.marker_callback, 1
        )
        self.get_logger().info(f"Subscribed to marker topic: {marker_topic_name}")

        # Create publishers for retargeted joint positions
        self.left_retargeted_pub = self.create_publisher(
            Float32MultiArray, "/left_retargeted_qpos", 1
        )
        self.right_retargeted_pub = self.create_publisher(
            Float32MultiArray, "/right_retargeted_qpos", 1
        )
        self.get_logger().info("Dual hand retargeting node initialized")

    def marker_callback(self, msg):
        # Check if marker array is empty
        if not msg.markers:
            self.get_logger().warn("Received empty marker array")
            return

        # Filter markers based on namespace
        left_markers = MarkerArray()
        right_markers = MarkerArray()
        
        # Get the namespace from the first marker
        if msg.markers and hasattr(msg.markers[0], 'ns'):
            namespace = msg.markers[0].ns
            
            # Check which hand this message belongs to
            is_left_hand = "left" in namespace.lower()
            is_right_hand = "right" in namespace.lower()
            
            if is_left_hand:
                left_markers.markers = [marker for marker in msg.markers]
                self.left_marker_queue.put(left_markers)
                self.get_logger().debug(f"Left hand markers put into queue, size: {self.left_marker_queue.qsize()}")
            elif is_right_hand:
                right_markers.markers = [marker for marker in msg.markers]
                self.right_marker_queue.put(right_markers)
                self.get_logger().debug(f"Right hand markers put into queue, size: {self.right_marker_queue.qsize()}")
            else:
                self.get_logger().warn(f"Unknown marker namespace: {namespace}")
        else:
            # If no namespace, try to determine hand type by marker IDs or other means
            self.get_logger().warn("No namespace found in markers, unable to determine hand type")

    def start_dual_retargeting(self, left_queue, right_queue, robot_dir, left_config_path, right_config_path, disable_collision=False, sim_vis=False):
        """
        Run retargeting process for both hands in a single process
        
        Args:
            left_queue: Queue for receiving left hand marker data
            right_queue: Queue for receiving right hand marker data
            robot_dir: Robot configuration directory
            left_config_path: Left hand retargeting configuration file path
            right_config_path: Right hand retargeting configuration file path
            disable_collision: Whether to disable collision detection to avoid STL warnings
            sim_vis: Whether to enable simulation visualization
        """
        logger.info(f"Starting dual hand retargeting process")
        logger.info(f"Left hand config: {left_config_path}")
        logger.info(f"Right hand config: {right_config_path}")
        
        # Set URDF directory
        robot_dir = Path(robot_dir)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        try:
            # Load configurations and build retargeters for both hands
            left_config = RetargetingConfig.load_from_file(left_config_path)
            right_config = RetargetingConfig.load_from_file(right_config_path)
            
            # Ensure type is lowercase
            if hasattr(left_config, "type") and isinstance(left_config.type, str):
                left_config.type = left_config.type.lower()
            if hasattr(right_config, "type") and isinstance(right_config.type, str):
                right_config.type = right_config.type.lower()
                
            # Build retargeters
            left_retargeting = left_config.build()
            right_retargeting = right_config.build()
            logger.info("Successfully built retargeters for both hands")
            
        except Exception as e:
            logger.error(f"Failed to load configs or build retargeters: {e}")
            raise
            
        # Set up SAPIEN renderer
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        except Exception as e:
            logger.warning(f"Error setting SAPIEN renderer, this might not affect functionality: {e}")
            
        # Create a single scene for both hands
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

        # Set up lighting
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

        # Set up camera - wider view to capture both hands
        cam = scene.add_camera(
            name="main_camera", width=800, height=600, fovy=1, near=0.1, far=10
        )
        cam.set_local_pose(sapien.Pose([0.7, 0.0, 0.3], [0, -0.2588, 0, 0.9659]))

        # Set up viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
        
        # If sim_vis is disabled, hide the viewer window
        if not sim_vis:
            viewer.hide_window()

        # Create URDF loader
        loader = scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True

        # Load left hand robot
        left_filepath = Path(left_config.urdf_path)
        logger.info(f"Using left hand URDF file: {left_filepath}")
        
        # Check if file exists
        if not os.path.exists(left_filepath):
            logger.error(f"Left hand URDF file not found: {left_filepath}")
            raise FileNotFoundError(f"Left hand URDF file not found: {left_filepath}")
            
        # Get robot name
        left_robot_name = left_filepath.stem
        logger.info(f"Left robot name: {left_robot_name}")

        # Apply scale based on robot type (same logic as before)
        if "ability" in left_robot_name:
            loader.scale = 1.5
        elif "dclaw" in left_robot_name:
            loader.scale = 1.25
        elif "allegro" in left_robot_name:
            loader.scale = 1.4
        elif "shadow" in left_robot_name:
            loader.scale = 0.9
        elif "bhand" in left_robot_name:
            loader.scale = 1.5
        elif "leap" in left_robot_name:
            loader.scale = 1.4
        elif "svh" in left_robot_name:
            loader.scale = 1.5
        elif "roboterax" in left_robot_name:
            loader.scale = 1.0

        # Special file path handling
        if "glb" not in left_robot_name and not left_robot_name.startswith("inspire"):
            left_filepath = str(left_filepath).replace(".urdf", "_glb.urdf")
            logger.info(f"Applying _glb suffix: {left_filepath}")
        else:
            left_filepath = str(left_filepath)
            logger.info(f"Using original file path: {left_filepath}")
            
        # Load left robot
        left_robot = loader.load(left_filepath)
        
        # Set left hand position - offset to the left side
        left_robot.set_pose(sapien.Pose([0, -0.2, 0]))  # Offset to left side

        # Load right hand robot
        right_filepath = Path(right_config.urdf_path)
        logger.info(f"Using right hand URDF file: {right_filepath}")
        
        # Check if file exists
        if not os.path.exists(right_filepath):
            logger.error(f"Right hand URDF file not found: {right_filepath}")
            raise FileNotFoundError(f"Right hand URDF file not found: {right_filepath}")
            
        # Get robot name
        right_robot_name = right_filepath.stem
        logger.info(f"Right robot name: {right_robot_name}")

        # Apply scale based on robot type (same logic as before)
        if "ability" in right_robot_name:
            loader.scale = 1.5
        elif "dclaw" in right_robot_name:
            loader.scale = 1.25
        elif "allegro" in right_robot_name:
            loader.scale = 1.4
        elif "shadow" in right_robot_name:
            loader.scale = 0.9
        elif "bhand" in right_robot_name:
            loader.scale = 1.5
        elif "leap" in right_robot_name:
            loader.scale = 1.4
        elif "svh" in right_robot_name:
            loader.scale = 1.5
        elif "roboterax" in right_robot_name:
            loader.scale = 1.0

        # Special file path handling
        if "glb" not in right_robot_name and not right_robot_name.startswith("inspire"):
            right_filepath = str(right_filepath).replace(".urdf", "_glb.urdf")
            logger.info(f"Applying _glb suffix: {right_filepath}")
        else:
            right_filepath = str(right_filepath)
            logger.info(f"Using original file path: {right_filepath}")
            
        # Load right robot
        right_robot = loader.load(right_filepath)
        
        # Set right hand position - offset to the right side
        right_robot.set_pose(sapien.Pose([0, 0.2, 0]))  # Offset to right side

        # Get joint names for both hands
        left_retargeting_joint_names = left_retargeting.joint_names if hasattr(left_retargeting, "joint_names") else []
        right_retargeting_joint_names = right_retargeting.joint_names if hasattr(right_retargeting, "joint_names") else []
            
        # Get SAPIEN joint names
        left_sapien_joint_names = [joint.get_name() for joint in left_robot.get_active_joints()]
        right_sapien_joint_names = [joint.get_name() for joint in right_robot.get_active_joints()]
        
        # Set mapping from retargeting to SAPIEN
        left_retargeting_to_sapien = np.array(
            [left_retargeting_joint_names.index(name) for name in left_sapien_joint_names]
        ).astype(int)
        
        right_retargeting_to_sapien = np.array(
            [right_retargeting_joint_names.index(name) for name in right_sapien_joint_names]
        ).astype(int)
        
        right_retargeting_to_robot = np.array(
            [right_retargeting_joint_names.index(name) for name in right_retargeting_joint_names]
        ).astype(int)

        left_retargeting_to_robot = np.array(
            [left_retargeting_joint_names.index(name) for name in left_retargeting_joint_names]
        ).astype(int)

        # Create ROS2 node for publishing retargeted joint positions
        ros_context = rclpy.Context()
        ros_context.init()
        ros_node = rclpy.create_node("retargeting_publisher", context=ros_context)
        left_retargeted_pub = ros_node.create_publisher(Float32MultiArray, "/left_retargeted_qpos", 1)
        right_retargeted_pub = ros_node.create_publisher(Float32MultiArray, "/right_retargeted_qpos", 1)
        
        # Store the last processed data for both hands
        left_joint_pos = None
        left_joint_ori = None
        left_qpos = None
        
        right_joint_pos = None
        right_joint_ori = None
        right_qpos = None

        joint_groups = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        if self.robot_name == "allegro":
            global_scale = 1.3
        elif self.robot_name == "roboterax":
            global_scale = 1.3
        else:
            global_scale = 1.3

        scale_factors = {
            'global': global_scale,
            'thumb': 1.0,
            'index': 1.0,
            'middle': 1.0,
            'ring': 1.0,
            'pinky': 1.0
        }
        # Main loop
        while True:
            try:
                # Check and process left hand queue
                try:
                    left_msg = left_queue.get_nowait()
                    left_joint_pos, left_joint_ori = process_hand_marker_array(left_msg, hand_type="left")
                    
                    if left_joint_pos is not None:
                        # Global scaling
                        left_joint_pos = left_joint_pos * scale_factors['global']
                        
                        # Scale each finger individually
                        for finger, indices in joint_groups.items():
                            # Ensure indices are within valid range
                            valid_indices = [idx for idx in indices if idx < left_joint_pos.shape[0]]
                            if valid_indices:
                                # Get wrist position as reference point
                                wrist_pos = left_joint_pos[0] if left_joint_pos.shape[0] > 0 else np.zeros(3)
                                
                                for idx in valid_indices:
                                    # Calculate relative position
                                    rel_pos = left_joint_pos[idx] - wrist_pos
                                    # Apply scaling
                                    scaled_rel_pos = rel_pos * scale_factors[finger]
                                    # Update joint position
                                    left_joint_pos[idx] = wrist_pos + scaled_rel_pos
                        
                        # Use retargeting logic to process joint positions
                        left_retargeting_type = left_retargeting.optimizer.retargeting_type
                        left_indices = left_retargeting.optimizer.target_link_human_indices
                        
                        if left_retargeting_type == "POSITION":
                            left_ref_value = left_joint_pos[left_indices, :]
                        else:
                            left_origin_indices = left_indices[0, :]
                            left_task_indices = left_indices[1, :]
                            left_ref_value = left_joint_pos[left_task_indices, :] - left_joint_pos[left_origin_indices, :]
                            
                        left_qpos = left_retargeting.retarget(left_ref_value)

                        
                        # Set robot pose
                        left_robot.set_qpos(left_qpos[left_retargeting_to_sapien])
                        
                        # Publish retargeted joint positions
                        left_retargeted_msg = Float32MultiArray()
                        left_retargeted_msg.data = left_qpos[left_retargeting_to_robot].flatten().tolist()
                        left_retargeted_pub.publish(left_retargeted_msg)
                        
                except Empty:
                    # No new left hand data, continue
                    pass
                    
                # Check and process right hand queue
                try:
                    right_msg = right_queue.get_nowait()
                    right_joint_pos, right_joint_ori = process_hand_marker_array(right_msg, hand_type="right")
                    
                    if right_joint_pos is not None:
                        # Global scaling
                        right_joint_pos = right_joint_pos * scale_factors['global']
                        
                        # Scale each finger individually
                        for finger, indices in joint_groups.items():
                            # Ensure indices are within valid range
                            valid_indices = [idx for idx in indices if idx < right_joint_pos.shape[0]]
                            if valid_indices:
                                # Get wrist position as reference point
                                wrist_pos = right_joint_pos[0] if right_joint_pos.shape[0] > 0 else np.zeros(3)
                                
                                for idx in valid_indices:
                                    # Calculate relative position
                                    rel_pos = right_joint_pos[idx] - wrist_pos
                                    # Apply scaling
                                    scaled_rel_pos = rel_pos * scale_factors[finger]
                                    # Update joint position
                                    right_joint_pos[idx] = wrist_pos + scaled_rel_pos
                        
                        # Use retargeting logic to process joint positions
                        right_retargeting_type = right_retargeting.optimizer.retargeting_type
                        right_indices = right_retargeting.optimizer.target_link_human_indices
                        
                        if right_retargeting_type == "POSITION":
                            right_ref_value = right_joint_pos[right_indices, :]
                        else:
                            right_origin_indices = right_indices[0, :]
                            right_task_indices = right_indices[1, :]
                            right_ref_value = right_joint_pos[right_task_indices, :] - right_joint_pos[right_origin_indices, :]
                            
                        right_qpos = right_retargeting.retarget(right_ref_value)
                        
                        # Set robot pose
                        right_robot.set_qpos(right_qpos[right_retargeting_to_sapien])
                        
                        # Publish retargeted joint positions
                        right_retargeted_msg = Float32MultiArray()
                        right_retargeted_msg.data = right_qpos[right_retargeting_to_robot].flatten().tolist()
                        right_retargeted_pub.publish(right_retargeted_msg)
                        
                except Empty:
                    # No new right hand data, continue
                    pass
                
                # Get left hand joint positions (through forward kinematics)
                left_robot_joints = left_robot.get_active_joints()
                left_robot_joint_positions = []
                left_robot_joint_names = []
                
                for joint in left_robot_joints:
                    link = joint.get_child_link()
                    if link:
                        pos = link.get_pose().p
                        left_robot_joint_positions.append(pos)
                        left_robot_joint_names.append(joint.get_name())
                
                # Get right hand joint positions (through forward kinematics)
                right_robot_joints = right_robot.get_active_joints()
                right_robot_joint_positions = []
                right_robot_joint_names = []
                
                for joint in right_robot_joints:
                    link = joint.get_child_link()
                    if link:
                        pos = link.get_pose().p
                        right_robot_joint_positions.append(pos)
                        right_robot_joint_names.append(joint.get_name())
                
                # Visualize human hands if we have valid data
                if left_joint_pos is not None and len(left_joint_pos) > 0:
                    # Create fingertip position dictionary
                    fingertip_positions = {}
                    finger_config = load_finger_config()
                    joint_mapping = finger_config["joint_mapping"]
                    
                    # If enough joint points, add fingertip positions
                    if len(left_joint_pos) > max(joint_mapping.values()):
                        fingertip_positions = {
                            'thumb': left_joint_pos[joint_mapping["thumb_end"]],
                            'index': left_joint_pos[joint_mapping["index_end"]],
                            'middle': left_joint_pos[joint_mapping["middle_end"]],
                            'ring': left_joint_pos[joint_mapping["ring_end"]],
                            'pinky': left_joint_pos[joint_mapping["pinky_end"]]
                        }
                    
                    # Visualize left human hand joint points (input)
                    visualize_hand_joints(
                        scene, 
                        left_joint_pos, 
                        fingertip_positions, 
                        hand_type="left",
                        joint_orientations=left_joint_ori,
                        finger_config=finger_config,
                        name_prefix="left_human_hand_",
                        position_offset=[0, -0.4, 0],  # Offset for left human hand
                        show_joints=self.show_joints
                    )
                
                if right_joint_pos is not None and len(right_joint_pos) > 0:
                    # Create fingertip position dictionary
                    fingertip_positions = {}
                    finger_config = load_finger_config()
                    joint_mapping = finger_config["joint_mapping"]
                    
                    # If enough joint points, add fingertip positions
                    if len(right_joint_pos) > max(joint_mapping.values()):
                        fingertip_positions = {
                            'thumb': right_joint_pos[joint_mapping["thumb_end"]],
                            'index': right_joint_pos[joint_mapping["index_end"]],
                            'middle': right_joint_pos[joint_mapping["middle_end"]],
                            'ring': right_joint_pos[joint_mapping["ring_end"]],
                            'pinky': right_joint_pos[joint_mapping["pinky_end"]]
                        }
                    
                    # Visualize right human hand joint points (input)
                    visualize_hand_joints(
                        scene, 
                        right_joint_pos, 
                        fingertip_positions, 
                        hand_type="right",
                        joint_orientations=right_joint_ori,
                        finger_config=finger_config,
                        name_prefix="right_human_hand_",
                        position_offset=[0, 0.4, 0],  # Offset for right human hand
                        show_joints=self.show_joints
                    )
                
                # Render scene
                for _ in range(2):
                    viewer.render()

                # Process ROS2 events
                rclpy.spin_once(ros_node, timeout_sec=0.001)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, exiting retargeting process")
                break
            except Exception as e:
                # Log error but continue, unless it's a fatal error
                logger.error(f"Error processing marker data: {e}")
                import traceback
                traceback.print_exc()
                if isinstance(e, (ValueError, AttributeError, IndexError, TypeError)):
                    # These errors might be temporary, continue
                    continue
                else:
                    # Other errors might be more severe, raise exception
                    logger.critical(f"Encountered fatal error, exiting process: {e}")
                    raise
        
        # Clean up resources
        ros_node.destroy_node()
        ros_context.shutdown()

    def destroy_node(self):
        # Terminate retargeting process
        if hasattr(self, "consumer_process") and self.consumer_process.is_alive():
            self.consumer_process.terminate()
            self.consumer_process.join(timeout=1.0)
            self.get_logger().info("Retargeting process terminated")

        super().destroy_node()


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hand Retargeting Node")
    parser.add_argument(
        "--robot-name",
        type=str,
        required=False,
        default="inspire",
        help="Robot name",
        choices=["inspire", "allegro", "shadow", "svh", "leap", "ability", "panda", "roboterax"],
    )
    parser.add_argument(
        "--hand-type",
        type=str,
        required=False,
        default="right",
        choices=["right", "left"],
        help="Hand type (right or left)",
    )
    parser.add_argument(
        "--retargeting-type",
        type=str,
        required=False,
        default="dexpilot",
        choices=["dexpilot", "vector", "position"],
        help="Retargeting type",
    )
    parser.add_argument(
        "--sim-vis",
        action="store_true",
        help="Enable simulation visualization",
    )
    parser.add_argument(
        "--dual-hands",
        action="store_true",
        help="Display both left and right hands simultaneously",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/joints_position",
        help="Topic to subscribe for hand kinematics markers",
    )
    parser.add_argument(
        "--disable-collision",
        action="store_true",
        help="Disable collision detection to avoid STL warnings",
    )
    parser.add_argument(
        "--hide-joints",
        action="store_true",
        help="Hide joint visualization spheres",
    )

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Print all available robot names
    print("Available robot names:")
    available_robots = [name for name in RobotName.__members__]
    print(available_robots)
    
    # Check if user-specified robot name is in the available list
    if args.robot_name not in available_robots:
        print(f"Warning: Robot name '{args.robot_name}' is not in the available list")

    # Initialize ROS2
    rclpy.init(args=unknown)

    try:
        if args.dual_hands:
            # Create a unified dual hand node
            node = DualHandRetargetNode(
                robot_name=args.robot_name,
                retargeting_type=args.retargeting_type.lower(),
                sim_vis=args.sim_vis,
                topic=args.topic,
                show_joints=not args.hide_joints,  # Pass the negated hide-joints flag
            )
            rclpy.spin(node)
        else:
            # Create a single hand node as before
            node = HandRetargetNode(
                robot_name=args.robot_name,
                hand_type=args.hand_type.lower(),
                retargeting_type=args.retargeting_type.lower(),
                sim_vis=args.sim_vis,
                topic=args.topic,
                show_joints=not args.hide_joints,  # Pass the negated hide-joints flag
            )
            rclpy.spin(node)
    except KeyboardInterrupt:
        print("Received interrupt signal, exiting node")
    except Exception as e:
        print(f"Error running node: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
