#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
import numpy as np
import argparse
import cv2
from loguru import logger
import time
from pathlib import Path
import multiprocessing
from queue import Empty
from scipy.spatial.transform import Rotation

# 添加调试信息，打印Python路径
import sys
print("Python路径:")
print(sys.path)

# 过滤SAPIEN的STL警告消息
import warnings
import logging
# 配置日志过滤器，过滤掉特定的SAPIEN警告
logging.getLogger('sapien').setLevel(logging.ERROR)
# 忽略特定警告
warnings.filterwarnings("ignore", message="loading multiple convex collision meshes from STL file")

# 添加本地lib路径到Python路径
import os
import os.path as osp

current_dir = osp.dirname(osp.abspath(__file__))
package_dir = osp.dirname(osp.dirname(current_dir))
lib_dir = osp.join(package_dir, "lib")
sys.path.insert(0, lib_dir)
print(f"添加本地lib路径: {lib_dir}")

# 从本地lib导入dex_retargeting模块
import dex_retargeting
print(f"dex_retargeting模块位置: {dex_retargeting.__file__}")
# 注释掉可能导致错误的reload语句
# import importlib
# importlib.reload(dex_retargeting)
# importlib.reload(dex_retargeting.constants)

# 直接从本地导入constants模块
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    ROBOT_NAME_MAP,
)

# 打印RobotName枚举中的所有成员
print("RobotName枚举中的所有成员:")
print([name for name in RobotName.__members__])

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

ROBOTERAX_JOINT_ORDER = [
    'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2', 
                        'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
                        'right_hand_mid_joint1', 'right_hand_mid_joint2',
                        'right_hand_ring_joint1', 'right_hand_ring_joint2',
                        'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
]


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
    加载手指配置信息
    
    Returns:
        字典，包含关节映射和可视化设置
    """
    # 默认配置
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
def visualize_hand_joints(scene, joint_positions, fingertip_positions, hand_type="Right", joint_orientations=None, finger_config=None, position_offset=None, name_prefix=""):
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
    """
    # Load configuration if not provided
    if finger_config is None:
        finger_config = load_finger_config()
    
    # Get visualization settings from config
    viz_config = finger_config.get("visualization", {})
    
    # Get joint mappings from config
    joint_mapping = finger_config["joint_mapping"]
    
    # Determine if left or right hand
    is_left_hand = True
    if hand_type is not None:
        if isinstance(hand_type, str):
            is_left_hand = hand_type.lower() == "left"
        else:
            is_left_hand = hand_type.name.lower() == "left"
    
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
    for i in range(len(joint_positions)):
        pos = joint_positions[i] * scale_factor + pos_offset
        builder = scene.create_actor_builder()
        # Create render material with color
        material = sapien.render.RenderMaterial()
        
        # 为不同的手指关节设置不同的颜色
        # 根据关节索引确定是哪个手指的关节
        if i == 0:  # 手腕
            material.base_color = [0.5, 0.5, 0.5, 1.0]  # 灰色
        elif i in [1, 2, 3, 4]:  # 拇指关节
            material.base_color = [1.0, 0.2, 0.2, 1.0]  # 红色
        elif i in [5, 6, 7, 8]:  # 食指关节
            material.base_color = [0.2, 1.0, 0.2, 1.0]  # 绿色
        elif i in [9, 10, 11, 12]:  # 中指关节
            material.base_color = [0.2, 0.2, 1.0, 1.0]  # 蓝色
        elif i in [13, 14, 15, 16]:  # 无名指关节
            material.base_color = [1.0, 1.0, 0.2, 1.0]  # 黄色
        elif i in [17, 18, 19, 20]:  # 小指关节
            material.base_color = [1.0, 0.2, 1.0, 1.0]  # 紫色
        else:
            material.base_color = [0.7, 0.7, 0.7, 1.0]  # 默认灰色
        
        # 调整关节大小，让指尖更明显
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
    
    # Visualize fingertip joint axes
    if joint_orientations is not None:
        # Define fingertip joints from config
        fingertip_joints = {
            'thumb': joint_mapping["thumb_end"],
            'index': joint_mapping["index_end"],
            'middle': joint_mapping["middle_end"],
            'ring': joint_mapping["ring_end"],
            'pinky': joint_mapping["pinky_end"]
        }
        
        # visualize_joint_axes(scene, joint_positions, joint_orientations, fingertip_joints, finger_config)
    
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
        # 创建标记ID到标记对象的映射
        markers_by_id = {marker.id: marker for marker in msg.markers}
        
        # 检查是否存在腕部标记
        if 0 not in markers_by_id:
            print(f"No wrist marker (ID 0) found in {hand_type} hand message")
            return None, None
        
        # 获取腕部标记的位置和方向
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
        
        # 初始化位置数组 - 用于存储所有关节的世界坐标
        keypoint_3d_array = np.zeros((joints_number, 3))
        keypoint_3d_array[0] = wrist_pos  # 设置腕部位置
        
        # 初始化方向数据数组
        orientations_data = np.zeros((joints_number, 4))
        orientations_data[0] = wrist_ori  # 设置腕部方向
        
        # 处理所有其他关节标记
        for joint_id in range(1, joints_number):
            if joint_id not in markers_by_id:
                # 如果缺少某个关节的标记，使用默认值
                continue
            
            # 获取关节标记的位置和方向
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
            
        # 1. 将所有关节坐标相对于腕部
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        
        # 2. 估计手腕旋转框架
        mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
        
        # 3. 应用坐标系变换
        operator2mano = OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else OPERATOR2MANO_LEFT
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano
        
        # 4. 为方向数据设置默认值
        orientations_data = np.zeros((joints_number, 4))
        orientations_data[:, 3] = 1.0  # 设置为单位四元数

        
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
    ):
        super().__init__("hand_retarget_node")

        self.hand_type = hand_type
        self.robot_name = robot_name
        
        # 将字符串转换为枚举值
        robot_name_enum = RobotName[robot_name]
        hand_type_enum = HandType[hand_type]
        retargeting_type_enum = RetargetingType[retargeting_type]
        
        self.get_logger().info(
            f"使用机器人: {robot_name_enum}, 手型: {hand_type_enum}, 重定向类型: {retargeting_type_enum}"
        )
        
        # 创建消息队列
        self.marker_queue = multiprocessing.Queue(maxsize=1000)
        
        # 查找机器人配置目录
        robot_config_path = Path(__file__).parent.parent.parent / "lib" / "dex-urdf" / "robots" / "hands"
        
        # 如果目录不存在，尝试其他路径
        if not robot_config_path.exists():
            robot_config_path = Path("/home/wind/ros2_ws/src/dex_retarget_wrapper_ros2_py/lib/dex-urdf/robots/hands")
            if not robot_config_path.exists():
                self.get_logger().error(f"找不到机器人配置目录: {robot_config_path}")
                raise ValueError(f"找不到机器人配置目录")
        
        self.get_logger().info(f"使用机器人配置路径: {robot_config_path}")
        
        # 设置默认URDF目录
        RetargetingConfig.set_default_urdf_dir(str(robot_config_path))
        
        # 使用枚举值获取配置文件路径
        config_path = get_default_config_path(
            robot_name_enum, retargeting_type_enum, hand_type_enum
        )
        self.get_logger().info(f"使用配置文件: {config_path}")
        
        # 启动重定向处理进程
        self.consumer_process = multiprocessing.Process(
            target=self.start_retargeting,
            args=(
                self.marker_queue,
                str(robot_config_path),
                str(config_path),
                self.hand_type,
            ),
        )
        self.consumer_process.daemon = True
        self.consumer_process.start()
        
        # 创建手部运动学标记的订阅者
        self.marker_sub = self.create_subscription(
            MarkerArray, "/hand_kinematics_markers", self.marker_callback, 1
        )

        # 创建重定向关节位置的发布者
        self.retargeted_pub = self.create_publisher(
            Float32MultiArray, "/retargeted_qpos", 1
        )
        self.get_logger().info("手部重定向节点已初始化")

    def marker_callback(self, msg):
        # 检查标记数组是否为空
        if not msg.markers:
            self.get_logger().warn("接收到空标记数组")
            return

        # 将标记数据放入队列
        self.marker_queue.put(msg)
        self.get_logger().debug(
            f"已将标记数据放入队列，当前队列大小: {self.marker_queue.qsize()}"
        )

    def start_retargeting(self, queue, robot_dir, config_path, hand_type):
        """
        在单独的进程中运行重定向处理
        
        Args:
            queue: 用于接收标记数据的队列
            robot_dir: 机器人配置目录
            config_path: 重定向配置文件路径
            hand_type: 手型（"left"或"right"）
        """
        logger.info(f"启动重定向处理进程，配置文件: {config_path}, 手型: {hand_type}")
        
        # 设置URDF目录
        robot_dir = Path(robot_dir)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        try:
            # 加载配置并构建重定向器
            config = RetargetingConfig.load_from_file(config_path)
            
            # 确保type是小写的
            if hasattr(config, "type") and isinstance(config.type, str):
                config.type = config.type.lower()
                
            # 构建重定向器
            retargeting = config.build()
            logger.info("成功构建重定向器")
            
        except Exception as e:
            logger.error(f"加载配置或构建重定向器失败: {e}")
            raise
            
        # 设置SAPIEN渲染器
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        except Exception as e:
            logger.warning(f"设置SAPIEN渲染器时出错，这可能不影响功能: {e}")
            
        # 创建场景
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

        # 设置光照
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

        # 设置相机
        cam = scene.add_camera(
            name="main_camera", width=600, height=600, fovy=1, near=0.1, far=10
        )
        cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

        # 设置查看器
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())

        # 加载机器人
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        logger.info(f"使用URDF文件: {filepath}")
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            logger.error(f"找不到URDF文件: {filepath}")
            raise FileNotFoundError(f"找不到URDF文件: {filepath}")
            
        # 获取机器人名称
        robot_name = filepath.stem
        logger.info(f"从文件路径获取的机器人名称: {robot_name}")
        loader.load_multiple_collisions_from_file = True

        # 根据机器人类型设置缩放比例
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
            loader.scale = 1.0  # 根据Roboterax的实际大小调整

        # 特殊处理文件路径
        if "glb" not in robot_name and not robot_name.startswith("inspire"):
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
            logger.info(f"应用_glb后缀: {filepath}")
        else:
            filepath = str(filepath)
            logger.info(f"使用原始文件路径: {filepath}")
            
        # 加载机器人
        robot = loader.load(filepath)

        # 根据机器人类型设置姿势
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
            robot.set_pose(sapien.Pose([0, 0, 0]))  # 根据Roboterax的实际情况调整
        else:
            robot.set_pose(sapien.Pose([0, 0, 0]))  # 默认

        # 获取关节名称
        retargeting_joint_names = retargeting.joint_names if hasattr(retargeting, "joint_names") else []
            
        # 获取SAPIEN关节名称
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        
        # 设置从重定向到SAPIEN的映射
        retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)
        
        # 设置从重定向到机器人的映射
        if self.robot_name == "inspire":
            retargeting_to_robot = np.array(
                [retargeting_joint_names.index(name) for name in INSPIRE_JOINT_ORDER]
            ).astype(int)
        elif self.robot_name == "roboterax":
            retargeting_to_robot = np.array(
                [retargeting_joint_names.index(name) for name in ROBOTERAX_JOINT_ORDER]
            ).astype(int)
        else:
            retargeting_to_robot = np.array(
                [retargeting_joint_names.index(name) for name in retargeting_joint_names]
            ).astype(int)

        # 创建ROS2节点用于发布重定向的关节位置
        ros_context = rclpy.Context()
        ros_context.init()
        ros_node = rclpy.create_node("retargeting_publisher", context=ros_context)
        retargeted_pub = ros_node.create_publisher(Float32MultiArray, "/retargeted_qpos", 1)
        
        # 主循环
        while True:
            try:
                # 从队列获取标记数据，超时5秒
                msg = queue.get(timeout=5)
                
                # 处理标记数据
                joint_pos, joint_ori = process_hand_marker_array(msg, hand_type=hand_type)
                
                # 对关节点坐标进行放缩和偏移来微调重定向效果
                if joint_pos is not None:
                    # 定义缩放因子和偏移量（可以根据需要调整）
                    scale_factors = {
                        'global': 1.3,  # 全局缩放因子
                        'thumb': 1.0,   # 拇指缩放因子
                        'index': 1.0,   # 食指缩放因子
                        'middle': 1.0,  # 中指缩放因子
                        'ring': 1.0,    # 无名指缩放因子
                        'pinky': 1.0    # 小指缩放因子
                    }
                    
                    # 定义关节组
                    joint_groups = {
                        'thumb': [1, 2, 3, 4],    # 拇指关节索引
                        'index': [5, 6, 7, 8],    # 食指关节索引
                        'middle': [9, 10, 11, 12], # 中指关节索引
                        'ring': [13, 14, 15, 16],  # 无名指关节索引
                        'pinky': [17, 18, 19, 20]  # 小指关节索引
                    }
                    
                    # 全局缩放
                    joint_pos = joint_pos * scale_factors['global']
                    
                    # 对每个手指单独缩放
                    for finger, indices in joint_groups.items():
                        # 确保索引在有效范围内
                        valid_indices = [idx for idx in indices if idx < joint_pos.shape[0]]
                        if valid_indices:
                            # 获取手腕位置作为基准点
                            wrist_pos = joint_pos[0] if joint_pos.shape[0] > 0 else np.zeros(3)
                            
                            for idx in valid_indices:
                                # 计算相对于手腕的位置
                                rel_pos = joint_pos[idx] - wrist_pos
                                # 应用缩放
                                scaled_rel_pos = rel_pos * scale_factors[finger]
                                # 更新关节位置
                                joint_pos[idx] = wrist_pos + scaled_rel_pos
                
                # 使用重定向逻辑处理关节位置
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    
                qpos = retargeting.retarget(ref_value)
                
                # 设置机器人姿势
                robot.set_qpos(qpos[retargeting_to_sapien])
                
                # 获取机器人手指关节的位置（通过正向运动学）
                robot_joints = robot.get_active_joints()
                robot_joint_positions = []
                robot_joint_names = []
                
                # 收集机器人关节位置和名称
                for joint in robot_joints:
                    # 获取关节在世界坐标系中的位置
                    link = joint.get_child_link()
                    if link:
                        pos = link.get_pose().p
                        robot_joint_positions.append(pos)
                        robot_joint_names.append(joint.get_name())
                
                # # 打印调试信息
                # logger.info(f"机器人关节数量: {len(robot_joint_positions)}")
                # if robot_joint_positions:
                #     logger.info(f"机器人关节名称: {robot_joint_names}")
                
                # 可视化手指关节
                # 获取手掌位置和手指关节位置
                if joint_pos is not None and len(joint_pos) > 0:
                    # 创建指尖位置字典
                    fingertip_positions = {}
                    finger_config = load_finger_config()
                    joint_mapping = finger_config["joint_mapping"]
                    
                    # 如果有足够的关节点，添加指尖位置
                    if len(joint_pos) > max(joint_mapping.values()):
                        fingertip_positions = {
                            'thumb': joint_pos[joint_mapping["thumb_end"]],
                            'index': joint_pos[joint_mapping["index_end"]],
                            'middle': joint_pos[joint_mapping["middle_end"]],
                            'ring': joint_pos[joint_mapping["ring_end"]],
                            'pinky': joint_pos[joint_mapping["pinky_end"]]
                        }
                    
                    # 可视化人手关节点（输入）
                    name_prefix = f"{hand_type}_human_hand_"
                    visualize_hand_joints(
                        scene, 
                        joint_pos, 
                        fingertip_positions, 
                        hand_type=hand_type,
                        joint_orientations=joint_ori if 'joint_ori' in locals() else None,
                        finger_config=finger_config,
                        name_prefix=name_prefix,
                        position_offset=[0, 0, 0]  # 向右偏移一点，避免与机器人手重叠
                    )
                    
                    # 可视化机器人手关节点（输出）
                    if robot_joint_positions:
                        # 转换为numpy数组
                        robot_joints_np = np.array(robot_joint_positions)
                        
                        # 创建机器人指尖位置字典（简化版，仅用于演示）
                        robot_fingertips = {}
                        
                        # 尝试找到机器人手的指尖关节
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
                        
                        # 可视化机器人手关节
                        robot_name_prefix = f"{hand_type}_robot_hand_"
                        visualize_hand_joints(
                            scene, 
                            robot_joints_np, 
                            robot_fingertips, 
                            hand_type=hand_type,
                            finger_config=finger_config,
                            name_prefix=robot_name_prefix,
                            position_offset=[0, 0, 0]  # 不需要偏移，使用机器人的实际位置
                        )

                # 渲染场景
                for _ in range(2):
                    viewer.render()
                
                # 发布重定向的关节位置
                retargeted_msg = Float32MultiArray()
                retargeted_msg.data = qpos[retargeting_to_robot].flatten().tolist()
                retargeted_pub.publish(retargeted_msg)

                # 处理ROS2事件
                rclpy.spin_once(ros_node, timeout_sec=0.001)
                
            except Empty:
                logger.warning("5秒内未收到标记数据")
                continue
            except KeyboardInterrupt:
                logger.info("接收到中断信号，退出重定向进程")
                break
            except Exception as e:
                # 记录错误但继续运行，除非是致命错误
                logger.error(f"处理标记数据时出错: {e}")
                if isinstance(e, (ValueError, AttributeError, IndexError, TypeError)):
                    # 这些错误可能是暂时的，继续运行
                    continue
                else:
                    # 其他错误可能更严重，抛出异常
                    logger.critical(f"遇到致命错误，退出进程: {e}")
                    raise
        
        # 清理资源
        ros_node.destroy_node()
        ros_context.shutdown()

    def destroy_node(self):
        # 终止重定向处理进程
        if hasattr(self, "consumer_process") and self.consumer_process.is_alive():
            self.consumer_process.terminate()
            self.consumer_process.join(timeout=1.0)
            self.get_logger().info("已终止重定向处理进程")

        super().destroy_node()


def main(args=None):
    # 解析命令行参数
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

    # 解析参数
    args, unknown = parser.parse_known_args()

    # 打印所有可用的机器人名称
    print("可用的机器人名称:")
    available_robots = [name for name in RobotName.__members__]
    print(available_robots)
    
    # 检查用户指定的机器人名称是否在可用列表中
    if args.robot_name not in available_robots:
        print(f"警告: 机器人名称 '{args.robot_name}' 不在可用列表中")

    # 初始化ROS2
    rclpy.init(args=unknown)

    try:
        # 创建并运行节点
        node = HandRetargetNode(
            robot_name=args.robot_name,
            hand_type=args.hand_type.lower(),
            retargeting_type=args.retargeting_type.lower(),
            sim_vis=args.sim_vis,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("接收到中断信号，退出节点")
    except Exception as e:
        print(f"运行节点时出错: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
