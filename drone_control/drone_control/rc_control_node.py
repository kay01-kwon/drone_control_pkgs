import rclpy
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
        converterParam, gainParam, dynParam = self._config()

        self.rc_converter = RcConverter(converterParam)
        self.rc_control = RcControl(gainParam, dynParam)

        # Get parameter for watermark
        watermarkParam = self._watermark_config()

        self.period = watermarkParam['period']

        self.timeout = np.array([watermarkParam['timeout_rc'],
                                 watermarkParam['timeout_odom'],
                                 watermarkParam['timeout_do']])

        self.latency = np.array([watermarkParam['latency_rc'],
                                  watermarkParam['latency_odom'],
                                  watermarkParam['latency_do']])

        self.loopback = watermarkParam['loopback']
        self.max_catchup_step =watermarkParam['max_catchup_step']

        self.time_latest = np.array([-np.inf,
                                     -np.inf,
                                     -np.inf])
        self.latest_rx_wall = np.zeros((3,))

        clock_now = self.get_clock().now()
        time_now = self._get_time_from_clock(clock_now)
        self.t_curr = time_now
        self.t_prev = self.t_curr

        self.mode = FlightMode.MANUAL_STAB
        self.prev_mode = self.mode

        self.rc_state_buf = CircularBuffer(capacity=30)
        self.odom_buf = CircularBuffer(capacity=30)
        self.wrench_buf = CircularBuffer(capacity=30)


        self.rc_in_sub = self.create_subscription(RCIn,
                                                 '/mavros/rc/in',
                                                 self._rc_in_cb,
                                                 qos_profile_sensor_data)

        self.odom_sub = self.create_subscription(Odometry,
                                                 '/mavros/local_position/odom',
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

        # Time setup for watermark
        self.time_latest[0] = rc_time - self.latency[0]
        clock_now = self.get_clock().now()
        time_now = self._get_time_from_clock(clock_now)
        self.latest_rx_wall[0] = time_now

        # RC buffer
        self.rc_converter.set_rc(rc_state)
        self.mode, v_des, dpsi_des = self.rc_converter.get_rc_state()
        des = np.concatenate([v_des, np.array([dpsi_des])])

        if self.rc_state_buf.is_full():
            self.rc_state_buf.pop()
            self.rc_state_buf.push((rc_time, des))
        else:
            self.rc_state_buf.push((rc_time, des))

        # Get string typed mode_name
        mode_name = RcModeStr.mode_str(self.mode)
        # When mode is switched, print out the mode
        if self.mode is not self.prev_mode:
            self.get_logger().info(f'mode: {mode_name}')

        self.prev_mode = self.mode

    def _odom_cb(self, msg:Odometry):
        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        # Time setup for watermark
        self.time_latest[1] = odom_time - self.latency[1]
        clock_now = self.get_clock().now()
        time_now = self._get_time_from_clock(clock_now)
        self.latest_rx_wall[1] = time_now

        if self.odom_buf.is_full():
            self.odom_buf.pop()
            self.odom_buf.push((odom_time, odom_data))
        else:
            self.odom_buf.push((odom_time, odom_data))

    def _do_cb(self, msg:WrenchStamped):
        do_time, do_state = MsgParser.parse_wrench_msg(msg)

        # Time setup for watermark
        self.time_latest[2] = do_time - self.latency[2]
        clock_now = self.get_clock().now()
        time_now = self._get_time_from_clock(clock_now)
        self.latest_rx_wall[2] = time_now

        if self.wrench_buf.is_full():
            self.wrench_buf.pop()
            self.wrench_buf.push((do_time, do_state))
        else:
            self.wrench_buf.push((do_time, do_state))

    def _timer_cb(self):
        self.cmd_pub.publish(self.cmd_msg)

    def _control_update(self):
        print('control_update')

    def _prepareBufferNear(self, buffer:CircularBuffer, t_ref: float) -> bool:
        if buffer.is_empty():
            return False
        # Remove data older than cutoff time
        cutoff = t_ref - self.loopback
        while not buffer.is_empty() and buffer.get_oldest()[0] < cutoff:
            buffer.pop()
        return not buffer.is_empty()

    def _watermark_time(self):
        clock_now = self.get_clock().now()
        wall_time_now = self._get_time_from_clock(clock_now)
        fresh_indices = [i for i in range(len(self.time_latest)) if self._freshByTTL(i, wall_time_now)]

        if not fresh_indices:
            return self.t_prev
        elif len(fresh_indices) == 1:
            watermark_now = self.time_latest[fresh_indices[0]]
        elif len(fresh_indices) > 1:
            watermark_now = self.time_latest[fresh_indices[0]]
            for i in fresh_indices[1:]:
                watermark_now = min(watermark_now, self.time_latest[i])
        return watermark_now

    def _freshByTTL(self, i, now_wall):
        is_fresh = ((self.latest_rx_wall[i] >= 0 )
                    and
                    (now_wall - self.time_latest[i] < self.timeout[i]))
        return is_fresh

    def _get_time_from_clock(self, clock):
        (sec, nsec) = clock.seconds_nanoseconds()
        return sec + nsec * 1e-9


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

    def _watermark_config(self):
        self.get_logger().info(f'{self.get_name()}: Initializing...')
        self.get_logger().info(f'water mark parameters')

        # Get watermark parameters
        period = self.get_parameter('watermark_param.period').value

        timeout_rc = self.get_parameter('watermark_param.timeout.rc').value
        timeout_odom = self.get_parameter('watermark_param.timeout.odom').value
        timeout_do = self.get_parameter('watermark_param.timeout.do').value

        lateness_rc = self.get_parameter('watermark_param.latency.rc').value
        lateness_odom = self.get_parameter('watermark_param.latency.odom').value
        lateness_do = self.get_parameter('watermark_param.latency.do').value

        loopback = self.get_parameter('watermark_param.loopback').value
        max_catchup_step = self.get_parameter('watermark_param.max_catchup_step').value

        watermarkParam = {'period': period,
                          'timeout_rc': timeout_rc,
                          'timeout_odom': timeout_odom,
                          'timeout_do': timeout_do,
                          'lateness_rc': lateness_rc,
                          'lateness_odom': lateness_odom,
                          'lateness_do': lateness_do,
                          'loopback': loopback,
                          'max_catchup_step': max_catchup_step}

        return watermarkParam

def main():
    rclpy.init()
    node = RcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()