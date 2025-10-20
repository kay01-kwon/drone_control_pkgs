import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.rc_control import RcControl, RcConverter, FlightMode

from drone_control.utils import math_tool

class RCControlNode(Node):
    def __init__(self):
        super().__init__('rc_control_node')

def main():
    rclpy.init()
    node = RCControlNode()
    print('Starting node ...')

if __name__ == '__main__':
    main()