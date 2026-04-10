"""
Moment excitation Node for mass estimation

This implements moment excitation
- Fixed thrust
- Moment rate generator

Author: Geonwoo Kwon
Date: 2026-03-17

"""


import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser
from drone_control.rc_control import RcConverter, FlightMode, RcModeStr
from uaclient.api.u.apt_news.current_news.v1 import current_news


class MomentExcitation(Node):
    """"
    Moment excitation Node

    FlightMode behavior:
    - KILL: All rotors stop (0 RPM)
    - DISARMED: Idle speed (2000 RPM)
    - AUTO: Excitation applied
    """

    def __init__(self):
        super().__init__('moment_excitation',
                         automatically_declare_parameters_from_overrides=True)
        (dynamic_param, drone_param, moment_ramp_param) = self._load_parameters()

        # Store parameters
        self.dynamic_param = dynamic_param
        self.drone_param = drone_param
        self.moment_ramp_param = moment_ramp_param

        m = dynamic_param['m']
        g = 9.81

        # Control input initialization
        self.f_fix = m*g / 2.0
        self.M = np.zeros((3,))

        self.count = 0

        # Moment ramp up parameters
        self.threshold_angle = moment_ramp_param['threshold_angle']
        self.M_dot = moment_ramp_param['moment_dot']
        self.xy_direction = moment_ramp_param['xy_direction']

        # Create RC converter
        self.rc_converter = RcConverter()

        # Create control allocator
        self.control_allocator = ControlAllocator(DroneParam=drone_param)

        # Buffers
        self.odom_buf = CircularBuffer(capacity=30)
        self.rc_buf = CircularBuffer(capacity=30)

        C_T = drone_param['motor_const']
        rotor_min = drone_param['rotor_min']
        self.f_min = 6.0*C_T*(rotor_min**2)

        # Initialize flag
        self.control_input_locked = False
        self.mode = FlightMode.DISARMED
        self.prev_mode = self.mode

        # Topic name for ros param
        cmd_topic = self.get_parameter('topic_names.cmd_topic').value
        rc_topic = self.get_parameter('topic_names.rc_topic').value
        filtered_odom_topic = self.get_parameter('topic_names.filtered_odom_topic').value

        # Create publisher
        self.cmd_pub = self.create_publisher(HexaCmdRaw, cmd_topic, 5)

        # Create subscribers
        self.rc_sub = self.create_subscription(RCIn,
                                               rc_topic,
                                               callback = self._rc_callback,
                                               qos_profile = qos_profile_sensor_data)

        self.odom_sub = self.create_subscription(Odometry,
                                                 filtered_odom_topic,
                                                 callback = self._odom_callback,
                                                 qos_profile = qos_profile_sensor_data)

        # Create timer
        self.excitation_period = 0.01

        self.excitation_timer = self.create_timer(self.excitation_period,
                                                  self._excitation_callback)

        self.get_logger().info('='*60)
        self.get_logger().info('Moment excitation initialized')
        self.get_logger().info(f'Command topic: {cmd_topic}')
        self.get_logger().info(f'Filtered odom topic: {filtered_odom_topic}')
        self.get_logger().info(f'RC topic: {rc_topic}')
        self.get_logger().info(
            f'Moment ramp: threshold angle = {np.rad2deg(self.threshold_angle): .1f} deg, '
            f'moment dot: {self.M_dot: .2f} Nm/s, ')
        self.get_logger().info('='*60)

    def _odom_callback(self, msg:Odometry):
        """
        Odometry callback function

        state format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        Note: Linear and angular velocity is expressed by body frame
        """

        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        if self.odom_buf.is_full():
            self.odom_buf.pop()
        self.odom_buf.push((odom_time, odom_data))

    def _rc_callback(self, msg:RCIn):
        """
        RC callback function

        """

        rc_time, rc_state = MsgParser.parse_rc_msg(msg)

        # Set RC state and get relevant mode
        self.rc_converter.set_rc(rc_state)
        self.mode, _, _ = self.rc_converter.get_rc_state()

        if self.rc_buf.is_full():
            self.rc_buf.pop()
        self.rc_buf.push((rc_time, rc_state))

        if self.mode is not self.prev_mode:
            mode_name = RcModeStr.mode_str(self.mode)
            self.get_logger().info(f'Mode: {mode_name}')

        self.prev_mode = self.mode

    def _compute_moment_ramp(self, roll:float, pitch:float, dt:float):
        """
        Compute moment ramp
        """

        phi = max(abs(roll), abs(pitch))

        if self.control_input_locked:
            self.f_fix = self.f_min
            self.M = np.zeros((3,))
            return

        if self.xy_direction == 'x':
            self.M[0] += self.M_dot * dt
        elif self.xy_direction == 'y':
            self.M[1] += self.M_dot * dt

        if phi > self.threshold_angle:
            self.f_fix = self.f_min
            self.M = np.zeros((3,))
            self.control_input_locked = True

    def _excitation_callback(self):
        """
        Main excitation loop callback

        KILL: All rotors stop (0 RPM)
        DISARMED: Idle speed (2000 RPM)
        AUTO: Excitation applied
        """

        if self.odom_buf.is_empty():
            return

        if self.rc_buf.is_empty():
            return

        # Get current time
        current_time = self._get_time_now()

        # Check odometry freshness
        odom_age = current_time - self.odom_buf.get_latest()[0]
        if odom_age > 0.05:     # 50 ms threshold
            if self.count % 100 == 0:
                self.get_logger().warn(
                    f'Stale odometry! Age: {odom_age*1000:.1f} ms',
                    throttle_duration_sec = 1.0
                )

        # --- KILL mode ---
        if self.mode == FlightMode.KILL:
            des_rpm = np.zeros((6,))
            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
                self.get_clock().now(),des_rpm
            )
            self.cmd_pub.publish(cmd_msg)
            return

        # --- DISARMED mode ---
        if self.mode == FlightMode.DISARMED:
            if self.control_input_locked:
                self.control_input_locked = False
                self.f_fix = self.dynamic_param['m'] * 9.81 / 2.0
                self.M = np.zeros((3,))
                self.get_logger().info('Control input unlocked (DISARMED)')
            des_rpm = 2000.0*np.ones((6,))
            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
                self.get_clock().now(),des_rpm
            )
            self.cmd_pub.publish(cmd_msg)
            return

        # --- AUTO mode ---
        if self.mode != FlightMode.AUTO:
            return

        # Get the latest state (full 13 dim odom)
        _, state_full = self.odom_buf.get_latest()

        # Extract quaternion and compute roll and pitch
        qw, qx, qy ,qz = state_full[6:10]
        roll = np.arctan2(2.0 * (qw * qx + qy * qz),
                          1.0 - 2.0 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

        # Compute moment ramp
        self._compute_moment_ramp(roll, pitch, self.excitation_period)

        des_rpm = (self.control_allocator
                              .compute_des_rpm(self.f_fix, self.M))

        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
            self.get_clock().now(), des_rpm
        )

        self.cmd_pub.publish(cmd_msg)

        # Use statistics
        self.count += 1

        if self.count % 100 == 0:
            self.get_logger().info(
                f'Stats: f_fix: {self.f_fix:.2f} N,'
                f'M [{self.M[0]:.3f}, {self.M[1]:.3f}] Nm,'
            )


    def _get_time_now(self) -> float:
        """Get the current ROS time as float (seconds)"""
        clock_now = self.get_clock().now()
        sec, nsec = clock_now.seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _load_parameters(self):
        """
        Load parameters from ROS2 paramter server

        Returns:
            Tuple of (dynamic_param, drone_param, moment_ramp_param)

        """

        # Nominal dynamic parameters
        m = self.get_parameter('dynamic_param.m').value
        MoiArray = self.get_parameter('dynamic_param.MoiArray').value

        # Drone parameters
        arm_length = self.get_parameter('drone_param.arm_length').value
        motor_const = self.get_parameter('drone_param.motor_const').value
        moment_const = self.get_parameter('drone_param.moment_const').value
        rotor_max = self.get_parameter('drone_param.rotor_max').value
        rotor_min = self.get_parameter('drone_param.rotor_min').value
        acc_max = self.get_parameter('drone_param.acc_max').value
        acc_min = self.get_parameter('drone_param.acc_min').value

        # Moment ramp parameters
        threshold_angle = self.get_parameter('moment_ramp_param.threshold_angle').value
        moment_dot = self.get_parameter('moment_ramp_param.moment_dot').value
        xy_direction = self.get_parameter('moment_ramp_param.xy_direction').value

        self.get_logger().info('Parameters loaded: ')
        self.get_logger().info(f' Mass: {m:.2f} kg')
        self.get_logger().info(f' Moi: {MoiArray} kg m^2 - Not in use')
        self.get_logger().info(f' Arm length: {arm_length:.3f} m')

        self.get_logger().info(f' Rotor const: {motor_const:.2e} N/(rpm)^2')
        self.get_logger().info(f' Rotor RPM limits: [{rotor_min:.2f}, {rotor_max:.2f}]')

        self.get_logger().info(f' Threshold angle: {threshold_angle*180.0/np.pi:.2f} deg')
        self.get_logger().info(f' moment dot ramp up: {moment_dot:.2f} Nm')
        self.get_logger().info(f' XY direction: {xy_direction}')
        dynamic_param = {
            'm':m,
            'Moi':MoiArray
        }

        drone_param = {
            'arm_length':arm_length,
            'motor_const':motor_const,
            'moment_const':moment_const,
            'rotor_max':rotor_max,
            'rotor_min':rotor_min,
            'acc_max':acc_max,
            'acc_min':acc_min
        }

        moment_ramp_param = {
            'threshold_angle':threshold_angle,
            'moment_dot':moment_dot,
            'xy_direction':xy_direction
        }
        return dynamic_param, drone_param, moment_ramp_param

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None

    try:
        # Create node
        node = MomentExcitation()

        # Use SingleThreadedExecutor for predictable behavior
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Spin
        print('\n[MomentExcitation node] is running. Press Ctrl+C to exit.')
        executor.spin()

    except KeyboardInterrupt:
        print('\n[MomentExcitation node] Keyboard interrupt received.')
    except Exception as ex:
        print(f'\n[MomentExcitation node] Exception occurred: {ex}')
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print('[MomentExcitation node] is shutting down.]')

if __name__ == '__main__':
    main()