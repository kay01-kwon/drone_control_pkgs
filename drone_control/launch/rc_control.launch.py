import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_prj_drone_control = get_package_share_directory('drone_control')

    config_file = os.path.join(pkg_prj_drone_control, 'config', 'rc_control.yaml')

    rc_control_node = Node(
        package='drone_control',
        executable='rc_control_node',
        name = 'rc_control',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': False},
        ]
    )

    return LaunchDescription([rc_control_node])