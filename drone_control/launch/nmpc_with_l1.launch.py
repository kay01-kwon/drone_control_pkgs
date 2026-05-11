import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_prj_drone_control = get_package_share_directory('drone_control')

    config_file = os.path.join(pkg_prj_drone_control, 'config', 'nmpc_with_l1.yaml')

    nmpc_with_l1_node = Node(
        package='drone_control',
        executable='nmpc_with_dob',
        name='nmpc_with_l1',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': False},
        ]
    )

    return LaunchDescription([nmpc_with_l1_node])
