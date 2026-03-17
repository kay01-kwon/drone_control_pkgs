import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_bag_record(context):
    pkg_share = get_package_share_directory('drone_control')
    config_path = os.path.join(pkg_share, 'config', 'bag_record.yaml')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dob_type = cfg['bag_record']['dob_type']

    if dob_type == 'hgdo':
        dob_wrench_topic = '/hgdo/wrench'
        filtered_odom_topic = '/hgdo/filtered_odom'
    else:
        dob_wrench_topic = '/l1_adaptive/wrench'
        filtered_odom_topic = '/l1_adaptive/filtered_odom'

    topics = [
        '/uav/cmd',
        '/uav/actual_vel',
        '/uav/rotor_state',
        '/mavros/local_position/odom_sim',
        filtered_odom_topic,
        dob_wrench_topic,
    ]

    bag_name = LaunchConfiguration('bag_name').perform(context)

    cmd = ['ros2', 'bag', 'record']
    if bag_name:
        cmd += ['-o', bag_name]
    cmd += topics

    bag_record = ExecuteProcess(
        cmd=cmd,
        output='screen',
    )

    return [bag_record]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_name',
            default_value='',
            description='Output bag file name',
        ),
        OpaqueFunction(function=_launch_bag_record),
    ])
