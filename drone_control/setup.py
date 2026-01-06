from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kay',
    maintainer_email='kay1216@yonsei.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'rc_control_node = drone_control.rc_control_node:main',
            'nmpc_node_v2 = drone_control.nmpc_node_v2:main',
            'nmpc_with_dob = drone_control.nmpc_with_dob:main',
            'msg_parser_node = drone_control.msg_parser_node:main'
        ],
    },
)
