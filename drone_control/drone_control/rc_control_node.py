import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

import threading

import numpy as np

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.rc_control import RcControl, RcConverter, FlightMode

from drone_control.utils.inverse_dynamics import InverseDynamics
from drone_control.utils import math_tool
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils.circular_buffer import CircularBuffer

class RCControlNode(Node):
    def __init__(self):
        super().__init__('rc_control_node')

        p_world = np.zeros((3,))
        v_body = np.zeros((3,))
        q = np.array([1,0,0,0])
        w = np.zeros((3,))

        self.state = np.concatenate([p_world,v_body,q,w])

        self.tau = np.zeros((3,))
        

        self.odom_sub = self.create_subscription(Odometry,
                                                 '/mavros/local_position/odom',
                                                 self.odom_cb,
                                                 5)

        self.disturbance_sub = self.create_subscription(WrenchStamped,
                                                        '/hgdo/wrench',
                                                        self.wrench_cb,
                                                        5)
        self.rcin_sub = self.create_subscription(RCIn,
                                                 '/mavros/rc/in',
                                                 self.rc_in_cb,
                                                 5)

        self.cmd_pub = self.create_publisher(HexaCmdRaw, '/uav/cmd_raw', 5)

        timer_period = 0.010
        self.timer = self.create_timer(timer_period, self.timer_cb)


    def odom_cb(self, msg):
        print('odom callback')

    def wrench_cb(self, msg):
        print('wrench callback')

    def rc_in_cb(self, msg):
        print('rc callback')

    def timer_cb(self):
        msg = HexaCmdRaw()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(msg)



def main():
    rclpy.init()
    node = RCControlNode()
    print('Starting node ...')

if __name__ == '__main__':
    main()