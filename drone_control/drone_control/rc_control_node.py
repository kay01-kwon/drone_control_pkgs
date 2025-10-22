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

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.inverse_dynamics import InverseDynamics
from drone_control.utils import math_tool, MsgParser
from drone_control.utils.cmd_converter import HexaCmdConverter


class RcControlNode(Node):
    def __init__(self):
        super().__init__('rc_control_node')

        p_world = np.zeros((3,))
        v_body = np.zeros((3,))
        q = np.array([1,0,0,0])
        w = np.zeros((3,))

        self.rc_converter = [None]
        self.rc_control = [None]

        self.rc_state = (0, np.zeros((12,)))
        self.state = np.concatenate([p_world,v_body,q,w])
        self.tau = np.zeros((3,))

        self.rc_state_buf = CircularBuffer(capacity=20)
        self.state_buf = CircularBuffer(capacity=20)
        self.wrench_buf = CircularBuffer(capacity=20)

        self._config()

        self.rcin_sub = self.create_subscription(RCIn,
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


    def odom_cb(self, msg):
        print('odom callback')

    def wrench_cb(self, msg):
        print('wrench callback')

    def rc_in_cb(self, msg):
        self.rc_state = MsgParser.parse_rc_msg(msg)
        print('rc state: ', self.rc_state)

    def timer_cb(self):
        msg = HexaCmdRaw()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(msg)

    def _config(self):

        node_name = self.get_name()
        print('node_names: ', node_name)

        self.declare_parameter(node_name + '/constraint/vxy_max', 0.3)
        self.declare_parameter(node_name + '/constraint/vz_max', 0.5)
        self.declare_parameter(node_name + '/constraint/dpsi_dt_max', 0.2)

        self.declare_parameter(node_name + '/gain_param/KpTransArray', [2, 2, 2])
        self.declare_parameter(node_name + '/gain_param/KpOriArray', [3, 3, 3])
        self.declare_parameter(node_name + '/gain_param/KdOriArray', [0.5, 0.5, 0.5])

        self.declare_parameter(node_name + '/dynamic_param/m',3)
        self.declare_parameter(node_name + '/dynamic_param/MoiArray', [3, 3, 3])

        vxy_max = (self.get_parameter(node_name + '/constraint/vxy_max')
                   .get_parameter_value()
                   .double_value)

        vz_max = (self.get_parameter(node_name + '/constraint/vz_max')
                  .get_parameter_value()
                  .double_value)

        dpsi_dt_max = (self.get_parameter(node_name + '/constraint/dpsi_dt_max')
                       .get_parameter_value()
                       .double_value)

        KpTransArray = (self.get_parameter(node_name + '/gain_param/KpTransArray')
                        .get_parameter_value()
                        .double_array_value)

        KpOriArray = (self.get_parameter(node_name + '/gain_param/KpOriArray')
                      .get_parameter_value()
                      .double_array_value)

        KdTransArray = (self.get_parameter(node_name + '/gain_param/KdOriArray')
                        .get_parameter_value()
                        .double_array_value)

        m = (self.get_parameter(node_name + '/dynamic_param/m')
             .get_parameter_value()
             .double_value)

        MoiArray = (self.get_parameter(node_name + '/dynamic_param/MoiArray')
                    .get_parameter_value()
                    .double_array_value)

        ConverterParam = {'vxy_max': vxy_max,
                          'vz_max': vz_max,
                          'dpsi_dt_max': dpsi_dt_max}

        GainParam = {'KpTransArray': KpTransArray,
                     'KpOriArray': KpOriArray,
                     'KdOriArray': KdTransArray}

        DynParam = {'m': m,
                    'MoiArray': MoiArray}

        self.rc_converter = RcConverter(ConverterParam)
        self.rc_control = RcControl(GainParam,DynParam)


def main():
    rclpy.init()
    print('Starting node ...')
    node = RcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()