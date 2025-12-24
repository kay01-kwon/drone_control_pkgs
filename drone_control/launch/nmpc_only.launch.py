import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_prj_drone_control = get_package_share_directory('drone_control')

    config_file = os.path.join(pkg_prj_drone_control, 'config', 'nmpc_only.yaml')

    nmpc_only_node = Node(
        package='drone_control',
        executable='nmpc_only_node',
        name = 'nmpc_only',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': True},
        ]
    )

    return LaunchDescription([nmpc_only_node])