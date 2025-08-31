import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # morning rain fog overcast sunset

    depthai_node = Node(
        package='virtual_kitti_publisher',
        executable='virtual_kitti_publisher_cuda_node',
        name='virtual_kitti_publisher_cuda_node',
        output='screen',
        parameters=[
            {'kitti_path': './../../vkitti/rgb/Scene02/rain/frames/rgb/'},
            {'depth_kitti_path': './../../vkitti/depth/Scene02/rain/frames/depth/'},
            {'model_path': '/tmp/StereoModel_576_960_fp16.plan'},
            {'record_video': True},
            {'net_input_width': 960},
            {'net_input_height': 576},
            {'fx': 725.0087},
            {'baseline': 0.532725},
            {'max_disp': 192.0}]

    )

    return LaunchDescription([
        depthai_node,
    ])
