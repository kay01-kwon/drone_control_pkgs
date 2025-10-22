import rclpy
from rclpy.node import Node
from rclpy.clock import ClockType
from rclpy.qos import QoSProfile, qos_profile_sensor_data

import threading

import numpy as np

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.rc_control import RcControl, RcConverter, FlightMode

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.inverse_dynamics import InverseDynamics
from drone_control.utils import math_tool, MsgParser
from drone_control.utils.cmd_converter import HexaCmdConverter


class RcControlNode(Node):
    def __init__(self):
        super().__init__('rc_control_node',
                         automatically_declare_parameters_from_overrides=True)

        p_world = np.zeros((3,))
        v_body = np.zeros((3,))
        q = np.array([1,0,0,0])
        w = np.zeros((3,))

        converterParam, gainParam, dynParam = self._config()


        self.rc_converter = RcConverter(converterParam)
        self.rc_control = RcControl(gainParam, dynParam)

        self.rc_state = (0, np.zeros((12,)))
        self.mode = FlightMode.MANUAL_STAB
        self.prev_mode = self.mode
        self.state = np.concatenate([p_world,v_body,q,w])
        self.tau = np.zeros((3,))

        self.rc_state_buf = CircularBuffer(capacity=20)
        self.state_buf = CircularBuffer(capacity=20)
        self.wrench_buf = CircularBuffer(capacity=20)

        self.rc_in_sub = self.create_subscription(RCIn,
                                                 '/mavros/rc/in',
                                                 self.rc_in_cb,
                                                 qos_profile_sensor_data)

        self.odom_sub = self.create_subscription(Odometry,
                                                 '/mavros/local_position/odom',
                                                 self.odom_cb,
                                                 qos_profile_sensor_data)

        self.do_sub = self.create_subscription(WrenchStamped,
                                                '/hgdo/wrench',
                                                self.wrench_cb,
                                                qos_profile_sensor_data)


        self.cmd_pub = self.create_publisher(HexaCmdRaw, '/uav/cmd_raw', 5)

        timer_period = 0.010
        self.timer = self.create_timer(timer_period, self.timer_cb)

        self.cmd_msg = HexaCmdRaw()


    def odom_cb(self, msg):
        odom_data = MsgParser.parse_odom_msg(msg)
        # print('odom callback')

    def wrench_cb(self, msg):
        print('wrench callback')

    def rc_in_cb(self, msg):
        rc_tuple = MsgParser.parse_rc_msg(msg)
        rc_time, rc_state = rc_tuple
        self.rc_converter.set_rc(rc_state)
        self.mode, v_des, dpsi_des = self.rc_converter.get_rc_state()

        mode_name = ""
        if self.mode is FlightMode.MANUAL_STAB:
            mode_name = "MANUAL_STAB"
        elif self.mode is FlightMode.AUTO:
            mode_name = "AUTO"
        elif self.mode is FlightMode.ARMED:
            mode_name = "ARMED"
        elif self.mode is FlightMode.KILL:
            mode_name = "KILL"

        if self.mode is not self.prev_mode:
            self.get_logger().info(f'mode: {mode_name}')

        self.prev_mode = self.mode

    def timer_cb(self):
        msg = HexaCmdRaw()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(msg)

    def _config(self):

        self.get_logger().info(f'{self.get_name()}: Initializing...')

        # Get constraint parameters
        vxy_max = self.get_parameter('constraint.vxy_max').value
        vz_max = self.get_parameter('constraint.vz_max').value
        dpsi_dt_max = self.get_parameter('constraint.dpsi_dt_max').value

        ConverterParam = {'vxy_max': vxy_max, 'vz_max': vz_max,
                          'dpsi_dt_max': dpsi_dt_max}

        # Get gain parameters
        KpTransArray = self.get_parameter('gain_param.KpTransArray').value
        KpOriArray = self.get_parameter('gain_param.KpOriArray').value
        KdOriArray = self.get_parameter('gain_param.KdOriArray').value

        # Get dynamic parameters
        m = self.get_parameter('dynamic_param.m').value
        MoiArray = self.get_parameter('dynamic_param.MoiArray').value

        gainParam = {'KpTransArray': KpTransArray,
                     'KpOriArray': KpOriArray,
                     'KdOriArray': KdOriArray}

        dynParam = {'m': m,
                    'MoiArray': MoiArray}

        self.get_logger().info(f'vxy_max: {vxy_max}')
        self.get_logger().info(f'vz_max: {vz_max}')
        self.get_logger().info(f'dpsi_dt_max: {dpsi_dt_max}')

        self.get_logger().info(f'KpTransArray: {KpTransArray}')
        self.get_logger().info(f'KpOriArray: {KpOriArray}')
        self.get_logger().info(f'KdOriArray: {KdOriArray}')

        self.get_logger().info(f'm: {m}')
        self.get_logger().info(f'MoiArray: {MoiArray}')

        return ConverterParam, gainParam, dynParam



def main():
    rclpy.init()
    node = RcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()