import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction


def _launch_bag_record(context):
    pkg_share = get_package_share_directory('drone_control')
    config_path = os.path.join(pkg_share, 'config', 'bag_record.yaml')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dob_type = cfg['bag_record']['dob_type']

    if dob_type == 'hgdo':
        dob_wrench_topic = '/hgdo/wrench'
    else:
        dob_wrench_topic = '/l1_adaptive/wrench'

    topics = [
        '/uav/cmd',
        '/uav/actual_vel',
        '/uav/rotor_state',
        '/mavros/local_position/odom',
        dob_wrench_topic,
    ]

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record'] + topics,
        output='screen',
    )

    return [bag_record]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_launch_bag_record),
    ])
