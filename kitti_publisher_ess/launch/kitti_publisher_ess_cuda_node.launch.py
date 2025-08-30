import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # morning rain fog overcast sunset

    depthai_node = Node(
        package='kitti_publisher_ess',
        executable='kitti_publisher_ess_cuda_node',
        name='kitti_publisher_ess_cuda_node',
        output='screen',
        parameters=[
            {'kitti_path': './../../vkitti/rgb/Scene02/rain/frames/rgb/'},
            {'plugin_path': './../dnn_stereo_disparity_v4.1.0_onnx/plugins/x86_64/ess_plugins.so'},
            {'depth_kitti_path': './../../vkitti/depth/Scene02/rain/frames/depth/'},
            {'model_path': '/tmp/ess.plan'},
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
