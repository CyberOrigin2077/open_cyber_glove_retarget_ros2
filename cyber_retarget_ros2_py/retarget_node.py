#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
import numpy as np
import argparse

# import from dex-realtime-retargeting
import sys
import os
sys.path.insert(0, os.environ["DEX_RETARGETING_PATH"])

from vector_retargeting.csv_retargeting_fingertip import (
    load_finger_config,
    transform_coordinates,
    compute_fingertip_positions,
)

from vector_retargeting.retargeting.utils import (
    process_hand_marker_array,
    transform_pose_to_wrist_frame,
    visualize_hand_joints,
)
from vector_retargeting.retargeting.hand_retargeting import HandRetargeting

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
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

class HandRetargetNode(Node):
    def __init__(
        self,
        finger_config_path="",
        robot_config_path="/home/zimo/Documents/dex_realtime_retargeting/dex-urdf/robots/hands",
        robot_name="inspire",
        hand_type="right",
        retargeting_type="dexpilot",
        sim_vis=False,
    ):
        super().__init__("hand_retarget_node")

        self.hand_type = hand_type
        self.finger_config = load_finger_config(finger_config_path)

        self.retargeting = HandRetargeting(
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_types=[hand_type],
            finger_config_path=finger_config_path,
            glove_name="cyber",
        )

        retargeting_joint_names = self.retargeting.get_joint_names(hand_type=self.hand_type)

        if robot_name == "inspire":
            self.retargeting_to_robot = np.array(
                [retargeting_joint_names.index(name) for name in INSPIRE_JOINT_ORDER]
            ).astype(int)
        else:
            self.retargeting_to_robot = np.array(
                [retargeting_joint_names.index(name) for name in retargeting_joint_names]
            ).astype(int)

        self.sim_vis = sim_vis
        if self.sim_vis:
            self.scene = sapien.Scene()
            render_mat = sapien.render.RenderMaterial()
            render_mat.base_color = [0.06, 0.08, 0.12, 1]
            render_mat.metallic = 0.0
            render_mat.roughness = 0.9
            render_mat.specular = 0.8
            self.scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

            # Setup lighting
            self.scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
            self.scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
            self.scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
            self.scene.set_environment_map(
                create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
            )
            self.scene.add_area_light_for_ray_tracing(
                sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
            )

            # Setup camera
            cam = self.scene.add_camera(
                name="main_camera",
                width=600,
                height=600,
                fovy=1,
                near=0.1,
                far=10
            )
            cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.control_window.show_origin_frame = False
            self.viewer.control_window.move_speed = 0.01
            self.viewer.control_window.toggle_camera_lines(False)
            self.viewer.set_camera_pose(cam.get_local_pose())

            # Load robot and set to a good pose
            loader = self.scene.create_urdf_loader()
            filepath = Path(self.retargeting.configs[hand_type].urdf_path)
            robot_name = filepath.stem
            loader.load_multiple_collisions_from_file = True

            # Set scale based on robot type - exactly as in csv_retargeting_fingertip.py
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

            # Fix the bug of the urdf file name for inspire hand - exactly as in csv_retargeting_fingertip.py
            if "glb" not in robot_name and not robot_name.startswith("inspire"):
                filepath = str(filepath).replace(".urdf", "_glb.urdf")
            else:
                filepath = str(filepath)
            self.robot = loader.load(filepath)

            # Set robot pose based on robot type - exactly as in csv_retargeting_fingertip.py
            if "ability" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "shadow" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.2]))
            elif "dclaw" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "allegro" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.05]))
            elif "bhand" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.2]))
            elif "leap" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.15]))
            elif "svh" in robot_name:
                self.robot.set_pose(sapien.Pose([0, 0, -0.13]))
            else:
                self.robot.set_pose(sapien.Pose([0, 0, -0.1]))  # Default

            self.sapien_joint_names = [
                joint.get_name() for joint in self.robot.get_active_joints()
            ]
            
            self.retargeting_to_sapien = np.array(
                [retargeting_joint_names.index(name) for name in self.sapien_joint_names]
            ).astype(int)

            self.get_logger().info(f"Robot joint names: {self.sapien_joint_names}")

        # Create subscriber for hand kinematics markers
        self.marker_sub = self.create_subscription(
            MarkerArray,
            "/hand_kinematics_markers",
            self.marker_callback,
            1
        )

        # Create publisher for retargeted joint positions
        self.retargeted_pub = self.create_publisher(
            Float32MultiArray,
            "/retargeted_qpos",
            1
        )
        self.get_logger().info("Hand retarget node initialized")
        

    def marker_callback(self, msg):
        # Check if marker array is empty
        if not msg.markers:
            self.get_logger().warn("Received empty marker array")
            return

        # Retarget the positions
        retargeted_positions = self.retarget_joints(msg)

        # Publish retargeted positions
        retargeted_msg = Float32MultiArray()
        retargeted_msg.data = retargeted_positions.flatten().tolist()
        self.retargeted_pub.publish(retargeted_msg)

    def retarget_joints(self, msg):
        """
        Retarget the joint positions from marker positions to robot joint angles.

        Args:
            msg (MarkerArray): Marker array of hand kinematics

        Returns:
            np.ndarray: Retargeted joint positions
        """
        # refer to https://github.com/CyberOrigin2077/dex_realtime_retargeting/blob/main/vector_retargeting/utils/ros2_realtime_retargeting.py
        joint_pos, joint_ori = process_hand_marker_array(
            msg,
            hand_type=self.hand_type
        )

        if self.hand_type == "right":
            retarget_result, finger_pos = self.retargeting.process_frame(
                right_joint_positions=joint_pos,
                right_joint_orientations=joint_ori,
                fingertips_data=True,
            )
        else:
            retarget_result, finger_pos = self.retargeting.process_frame(
                left_joint_positions=joint_pos,
                left_joint_orientations=joint_ori,
                fingertips_data=True,
            )
        qpos = retarget_result[self.hand_type]
        # self.get_logger().info(f"Retargeted joint positions: {qpos}")
        if self.sim_vis:
            self.robot.set_qpos(qpos[self.retargeting_to_sapien])
            # visualize_hand_joints(self.scene, np.zeros((20,3)), finger_pos, hand_type=self.hand_type, finger_config=self.finger_config)
            for _ in range(2):
                self.viewer.render()

        return qpos[self.retargeting_to_robot]


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hand Retargeting Node")
    parser.add_argument(
        "--finger-config",
        type=str,
        required=True,
        help="Path to the finger configuration file",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        required=True,
        help="Path to the robot configuration directory",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        required=False,
        default="inspire",
        help="Robot name",
    )

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Initialize ROS2
    rclpy.init(args=unknown)

    # Create and run the node with configuration paths
    node = HandRetargetNode(
        finger_config_path=args.finger_config,
        robot_config_path=args.robot_config,
        robot_name=args.robot_name,
        sim_vis=True,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)  # Process one message with minimal timeout
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
