import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the directory of the package
    drone_dob_pkg_share = get_package_share_directory('drone_dob')

    # Define the path to the configuration file
    config_file_path = os.path.join(drone_dob_pkg_share, 'config', 'hgdo_config.yaml')
    print(f"Using configuration file: {config_file_path}")

    # Create the Node action for the hgdo_node
    hgdo_node = Node(
        package='drone_dob',
        executable='hgdo_node',
        name='hgdo_node',
        output='screen',
        parameters=[config_file_path,
                    {'use_sim_time': True}]
    )

    # Create and return the LaunchDescription
    return LaunchDescription([
        hgdo_node
    ])