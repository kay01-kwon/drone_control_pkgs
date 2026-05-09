import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_prj_drone_control = get_package_share_directory('drone_control')

    config_file = os.path.join(pkg_prj_drone_control, 'config', 'pd_nmpc_att_with_l1.yaml')

    pd_nmpc_att_node = Node(
        package='drone_control',
        executable='pd_nmpc_att_with_dob',
        name='pd_nmpc_att_with_l1',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': False},
        ]
    )

    return LaunchDescription([pd_nmpc_att_node])
