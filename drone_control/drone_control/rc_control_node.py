import rclpy
from mercurial.revset import predicate
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import threading

import numpy as np

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.rc_control import RcControl, RcConverter
from drone_control.rc_control import FlightMode, RcModeStr

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

        # Get parameter for converter and controller
        converterParam, gainParam, dynParam, droneParam = self._config()

        self.rc_converter = RcConverter(converterParam)
        self.rc_control = RcControl(gainParam, dynParam)
        self.inverse_dynamics = InverseDynamics(droneParam)

        # Get parameter for watermark
        sensorTimeParam = self._sensor_time_config()

        self.timeout = np.array([sensorTimeParam['timeout_rc'],
                                 sensorTimeParam['timeout_odom'],
                                 sensorTimeParam['timeout_do']])

        time_now = self._get_time_now()
        self.t_curr = time_now
        self.t_prev = self.t_curr

        self.mode = FlightMode.MANUAL_STAB
        self.prev_mode = self.mode

        self.rc_state_buf = CircularBuffer(capacity=30)
        self.odom_buf = CircularBuffer(capacity=30)
        self.wrench_buf = CircularBuffer(capacity=30)

        self.des_rpm = np.zeros((6,))

        self.rc_in_sub = self.create_subscription(RCIn,
                                                 '/mavros/rc/in',
                                                 self._rc_in_cb,
                                                 qos_profile_sensor_data)

        self.odom_sub = self.create_subscription(Odometry,
                                                 '/S550/ground_truth/odom',
                                                 self._odom_cb,
                                                 qos_profile_sensor_data)

        self.do_sub = self.create_subscription(WrenchStamped,
                                                '/hgdo/wrench',
                                                self._do_cb,
                                                qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(HexaCmdRaw, '/uav/cmd_raw', 5)

        timer_period = 0.010
        self.timer = self.create_timer(timer_period, self._timer_cb)

        self.cmd_msg = HexaCmdRaw()

    def _rc_in_cb(self, msg:RCIn):
        rc_tuple = MsgParser.parse_rc_msg(msg)
        rc_time, rc_state = rc_tuple

        # RC buffer
        self.rc_converter.set_rc(rc_state)
        self.mode, v_des, dpsi_des = self.rc_converter.get_rc_state()
        cmd_vel = np.concatenate([v_des, np.array([dpsi_des])])

        if self.rc_state_buf.is_full():
            self.rc_state_buf.pop()
            self.rc_state_buf.push((rc_time, cmd_vel))
        else:
            self.rc_state_buf.push((rc_time, cmd_vel))

        # Get string typed mode_name
        mode_name = RcModeStr.mode_str(self.mode)
        # When mode is switched, print out the mode
        if self.mode is not self.prev_mode:
            self.get_logger().info(f'mode: {mode_name}')

        self.prev_mode = self.mode

    def _odom_cb(self, msg:Odometry):

        odom_time, odom_data = MsgParser.parse_odom_msg(msg)
        if self.odom_buf.is_full():
            self.odom_buf.pop()
            self.odom_buf.push((odom_time, odom_data))
        else:
            self.odom_buf.push((odom_time, odom_data))

    def _do_cb(self, msg:WrenchStamped):
        do_time, do_state = MsgParser.parse_wrench_msg(msg)

        if self.wrench_buf.is_full():
            self.wrench_buf.pop()
            self.wrench_buf.push((do_time, do_state))
        else:
            self.wrench_buf.push((do_time, do_state))

    def _timer_cb(self):

        self.t_curr = self._get_time_now()

        # self.get_logger().info(f'Timer callback')

        if self.odom_buf.is_empty():
            return

        if self.rc_state_buf.is_empty():
            return

        # if self.wrench_buf.is_empty():
        #     return

        t_diff_odom_abs = np.abs(self.t_curr - self.odom_buf.get_latest()[0])

        if t_diff_odom_abs > self.timeout[1]:
            self.get_logger().info(f'odom : STALE')
            self.get_logger().info(f'Switch to KILL Automatically')
            self.mode = FlightMode.KILL

        if self.mode == FlightMode.KILL:
            for i in range(len(self.des_rpm)):
                self.des_rpm[i] = 0
        elif self.mode == FlightMode.DISARMED:
            for i in range(len(self.des_rpm)):
                self.des_rpm[i] = 0
        elif self.mode == FlightMode.ARMED:
            for i in range(len(self.des_rpm)):
                self.des_rpm[i] = 2000
        elif self.mode == FlightMode.MANUAL_STAB:
            cmd_vel = self.rc_state_buf.get_latest()[1]
            # wrench_recent = self.wrench_buf.get_latest()[1]
            state_recent = self.odom_buf.get_latest()[1]
            self.rc_control.set_ref(cmd_vel,
                                    state_recent,
                                    np.zeros((3,)))
            u = self.rc_control.get_control_input()
            self.get_logger().info(f'u: {u}')
            self.des_rpm = self.inverse_dynamics.compute_des_rpm(u[0],u[1:])

        self.cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.t_curr, self.des_rpm)

        self.cmd_pub.publish(self.cmd_msg)

    def _get_time_now(self):
        clock_now = self.get_clock().now()
        (sec, nsec) = clock_now.seconds_nanoseconds()
        time_now = sec + nsec*1e-9
        return time_now

    def _config(self):
        self.get_logger().info(f'{self.get_name()}: Initializing...')
        self.get_logger().info(f'RC Converter and RC control parameters')

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

        # Get drone parameter
        arm_length = self.get_parameter('drone_param.arm_length').value
        rotor_const = self.get_parameter('drone_param.rotor_const').value
        moment_const = self.get_parameter('drone_param.moment_const').value
        T_max = self.get_parameter('drone_param.T_max').value
        T_min = self.get_parameter('drone_param.T_min').value

        gainParam = {'KpTransArray': KpTransArray,
                     'KpOriArray': KpOriArray,
                     'KdOriArray': KdOriArray}

        dynParam = {'m': m,
                    'MoiArray': MoiArray}

        droneParam = {'arm_length': arm_length,
                      'rotor_const': rotor_const,
                      'moment_const': moment_const,
                      'T_max': T_max,
                      'T_min': T_min}

        self.get_logger().info(f'vxy_max: {vxy_max}')
        self.get_logger().info(f'vz_max: {vz_max}')
        self.get_logger().info(f'dpsi_dt_max: {dpsi_dt_max}')

        self.get_logger().info(f'KpTransArray: {KpTransArray}')
        self.get_logger().info(f'KpOriArray: {KpOriArray}')
        self.get_logger().info(f'KdOriArray: {KdOriArray}')

        self.get_logger().info(f'm: {m}')
        self.get_logger().info(f'MoiArray: {MoiArray}')

        return ConverterParam, gainParam, dynParam, droneParam

    def _sensor_time_config(self):
        self.get_logger().info(f'{self.get_name()}: Initializing...')
        self.get_logger().info(f'sensor time parameters')

        # Get watermark parameters

        timeout_rc = self.get_parameter('sensor_time_param.timeout.rc').value
        timeout_odom = self.get_parameter('sensor_time_param.timeout.odom').value
        timeout_do = self.get_parameter('sensor_time_param.timeout.do').value

        self.get_logger().info(f'timeout_rc: {timeout_rc}')
        self.get_logger().info(f'timeout_odom: {timeout_odom}')
        self.get_logger().info(f'timeout_do: {timeout_do}')

        sensor_time_param = {'timeout_rc': timeout_rc,
                             'timeout_odom': timeout_odom,
                             'timeout_do': timeout_do}

        return sensor_time_param

def main():
    rclpy.init()
    node = RcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()