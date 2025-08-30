from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    kitti_node = Node(
        package='kitti_publisher',
        executable='kitti_publisher_cuda_node',
        name='kitti_publisher_cuda_node',
        output='screen',
        parameters=[
            {'kitti_path': '/path/to/sequence/04'},
            {'model_path': '/tmp/StereoModel.plan'},
            {'record_video': True},
            {'net_input_width': 1248},
            {'net_input_height': 384},
            {'fx': 707.0912},
            {'baseline': 0.536},
            {'max_disp': 192.0}
        ]
    )

    return LaunchDescription([
        kitti_node,
    ])
