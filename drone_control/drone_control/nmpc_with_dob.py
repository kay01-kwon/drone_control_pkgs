"""
ROS2 NMPC Node with Disturbance Observer (DOB)

This implements NMPC with disturbance compensation from DOB:
- Receives disturbances wrench from DOB (HGDO, L1 adaptaiton, EKF/UKF)
- Compensates control input for estimated disturbance
- Timer-based control loop (no manual threading)
- SingleThreadedExecutor for predictable behavior
- RC --> Manual STAB is replaced by LANDING state

Author: Geonwoo Kwon
Date: 2026-03-18
"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry
from drone_msgs.msg import Ref
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw
from ros2_libcanard_msgs.msg import HexaActualRpm

from drone_control.rc_control import RcConverter
from drone_control.rc_control import FlightMode, RcModeStr

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, cleanup_acados_files
from drone_control.utils.low_pass_filter import LowPassFilter
from drone_control.nmpc.ocp.S550_simple_ocp import S550_Ocp

class NmpcWithDOBNode(Node):
    """
    NMPC Node with DOB integration for ROS2
    This node implements Model Predictive control with disturbance compensation.
    Control loop runs at 100 Hz using ROS2 timer callback.
    """
    def __init__(self):
        super().__init__('nmpc_with_dob',
                         automatically_declare_parameters_from_overrides=True)

        # Load parameters for nmpc
        dynamic_param, drone_param, nmpc_param = self._load_parameters()

        # Store parameters as instance variables
        self.dynamic_param = dynamic_param
        self.drone_param = drone_param
        self.nmpc_param = nmpc_param

        # Create RC converter
        self.rc_converter = RcConverter()

        # Create NMPC solver
        self.get_logger().info('Creating NMPC solver...')
        self.nmpc_solver = S550_Ocp(DynParam=dynamic_param,
                                    DroneParam=drone_param,
                                    MpcParam=nmpc_param)

        # Create control allocator
        self.control_allocator = ControlAllocator(DroneParam=drone_param)

        # Flight mode
        self.mode = FlightMode.DISARMED
        self.prev_mode = self.mode

        # Buffers
        self.odom_buffer = CircularBuffer(capacity=30)
        self.wrench_buffer = CircularBuffer(capacity=30)

        # Reference state (p, v, q, w) in 13 dim
        self.ref_state = np.zeros((13,))
        self.ref_state[6] = 1.0     # qw = 1 (Identity quaternion)

        # Statistics for NMPC solver
        self.solve_count = 0
        self.failure_count = 0
        self.total_solve_time = 0.0

        # Flags
        self.moment_ff_flag = dynamic_param['moment_ff']
        self.solver_ready = False
        self.first_solve = True

        m = self.dynamic_param['m'] if hasattr(self, 'dynamic_param') else 3.0
        g = 9.81
        self.W = m * g
        u_hover = m * g / 6.0

        # Feedforward moment
        com_offset = dynamic_param['com_offset']

        x_off = com_offset[0]
        y_off = com_offset[1]

        self.M_ff = np.array([
            self.W*y_off,
            -self.W*x_off,
            0.0
        ])

        self.des_rotor_thrust_mpc = u_hover * np.ones((6,))
        self.des_rotor_rpm_comp = np.zeros_like(self.des_rotor_thrust_mpc)
        self.des_rotor_rpm_comp_prev = np.zeros_like(self.des_rotor_thrust_mpc)
        self.C_T = self.drone_param['motor_const'] if hasattr(self, 'drone_param') else 1.386e-7

        # Actual RPM → total thrust for ground/flight detection
        self.actual_total_thrust = 0.0
        self.was_airborne = False

        # Initial pz offset (from mocap)
        self.pz_offset = None

        # LPF for cmd output
        cmd_lpf_cutoff = self.get_parameter('drone_param.cmd_lpf_cutoff').value
        self.cmd_lpf = LowPassFilter(cutoff_freq=cmd_lpf_cutoff)
        self.get_logger().info(f'CMD LPF cutoff: {cmd_lpf_cutoff} Hz')

        # Topic name from ros param
        cmd_topic = self.get_parameter('topic_names.cmd_topic').value
        rc_topic = self.get_parameter('topic_names.rc_topic').value
        nmpc_topic = self.get_parameter('topic_names.base_line_control_topic').value
        filtered_odom_topic = self.get_parameter('topic_names.filtered_odom_topic').value
        ref_topic = self.get_parameter('topic_names.ref_topic').value
        dob_wrench_topic = self.get_parameter('topic_names.dob_wrench_topic').value

        # Create publisher
        self.cmd_pub = self.create_publisher(HexaCmdRaw,
                                             cmd_topic,
                                             5)

        self.nmpc_pub = self.create_publisher(WrenchStamped,
                                              nmpc_topic,
                                              qos_profile=5)

        # Create subscribers
        self.odom_sub = self.create_subscription(Odometry,
                                                 filtered_odom_topic,
                                                 callback=self._odom_callback,
                                                 qos_profile=qos_profile_sensor_data)

        self.rc_sub = self.create_subscription(RCIn,
                                               rc_topic,
                                               callback=self._rc_callback,
                                               qos_profile=qos_profile_sensor_data)

        self.ref_sub = self.create_subscription(Ref,
                                                ref_topic,
                                                callback=self._ref_callback,
                                                qos_profile=10)

        self.wrench_sub = self.create_subscription(WrenchStamped,
                                                   dob_wrench_topic,
                                                   callback=self._wrench_dob_callback,
                                                   qos_profile=qos_profile_sensor_data)

        self.actual_rpm_sub = self.create_subscription(HexaActualRpm,
                                                       '/uav/actual_rpm',
                                                       callback=self._actual_rpm_callback,
                                                       qos_profile=qos_profile_sensor_data)

        # Create control timer ( 100 Hz )
        self.control_period = 0.01
        self.control_timer = self.create_timer(self.control_period,
                                               self._control_callback)

        self.get_logger().info('='*60)
        self.get_logger().info(f'Command topic: {cmd_topic}')
        self.get_logger().info(f'NMPC topic: {nmpc_topic}')
        self.get_logger().info(f'Filtered odom topic: {filtered_odom_topic}')
        self.get_logger().info(f'Reference topic: {ref_topic}')
        self.get_logger().info(f'DoB wrench topic: {dob_wrench_topic}')
        self.get_logger().info('NMPC With DOB Node initialized successfully')
        self.get_logger().info(f'Control rate: {1.0/self.control_period:.1f} Hz')
        self.get_logger().info(f'Horizon: {nmpc_param["t_horizon"]:.2f}s')
        self.get_logger().info(f'Nodes: {nmpc_param["n_nodes"]}')
        self.get_logger().info(f'Q: {nmpc_param["QArray"]}')
        self.get_logger().info(f'R: {nmpc_param["R"]}')
        self.get_logger().info('='*60)

        # Print out flight mode
        self.get_logger().info(f'mode: {self.mode}')

    def _odom_callback(self, msg: Odometry):
        """
        Odometry callback - stores latest state in buffer

        State format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        Note: Linear and angular velocity is in Body frame from odometry
        """

        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        # Subtract initial pz offset (mocap height above ground)
        if self.pz_offset is None:
            self.pz_offset = odom_data[2]
            self.get_logger().info(f'Initial pz offset: {self.pz_offset:.4f} m')
        odom_data[2] -= self.pz_offset

        if self.odom_buffer.is_full():
            self.odom_buffer.pop()
        self.odom_buffer.push((odom_time, odom_data))

    def _rc_callback(self, msg: RCIn):
        rc_tuple = MsgParser.parse_rc_msg(msg)
        rc_time, rc_state = rc_tuple

        # Get RC mode
        self.rc_converter.set_rc(rc_state)
        self.mode, _, _ = self.rc_converter.get_rc_state()

        # When mode is switched, print out the mode
        if self.mode is not self.prev_mode:
            self.get_logger().info(f'Mode: {self.mode}')
        self.prev_mode = self.mode

    def _ref_callback(self, msg: Ref):
        """
        Reference callback - updates desired state

        Reference format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        """

        # Position and linear velocity (Both -> World frame)
        self.ref_state[0:3] = msg.p
        self.ref_state[3:6] = msg.v

        # Quaternion from yaw angle
        self.ref_state[6] = np.cos(msg.psi/2.0)     # qw
        self.ref_state[7] = 0.0                     # qx
        self.ref_state[8] = 0.0                     # qy
        self.ref_state[9] = np.sin(msg.psi/2.0)     # qz

        self.ref_state[10] = 0.0                    # wx
        self.ref_state[11] = 0.0                    # wy
        self.ref_state[12] = msg.psi_dot            # wz

    def _actual_rpm_callback(self, msg: HexaActualRpm):
        """
        Actual RPM callback - computes total thrust from actual RPM
        for ground/flight state detection
        """
        rpms = np.array(msg.rpm, dtype=np.float64)
        thrusts = self.C_T * rpms ** 2
        self.actual_total_thrust = np.sum(thrusts)

    def _wrench_dob_callback(self, msg:WrenchStamped):
        """
        DOB wrench callback - stores latest disturbance estimate in buffer

        Wrench format: [f_x, f_y, f_z, tau_x, tau_y, tau_z]
        """
        wrench_time, wrench_data = MsgParser.parse_wrench_msg(msg)

        if self.wrench_buffer.is_full():
            self.wrench_buffer.pop()
        self.wrench_buffer.push((wrench_time, wrench_data))

    def _control_callback(self):
        """
        Main control loop callback - runs at 100 Hz

        This is where the NMPC is solved and commands are published
        """

        # Check if we have odomemtry data
        if self.odom_buffer.is_empty():
            return

        if self.mode == FlightMode.KILL:
            self._set_rpm_zero()
            return
        elif self.mode == FlightMode.DISARMED:
            self._set_rpm_zero()
            return
        elif self.mode == FlightMode.ARMED:
            self.get_logger().info(f'LANDING state')
            self.ref_state[2] = 0.0
        elif self.mode == FlightMode.MANUAL_STAB:
            self.get_logger().info(f'LANDING state')
            # Landing (Altitude set to zero)
            self.ref_state[2] = 0.0

        # Get current time
        current_time = self._get_time_now()

        # Check odometry freshness
        odom_age = current_time - self.odom_buffer.get_latest()[0]
        if odom_age > 0.05:     # 50 ms threshold
            if self.solve_count % 100 == 0:
                self.get_logger().warn(
                    f'Stale odometry! Age: {odom_age*1000:.1f} ms',
                    throttle_duration_sec = 1.0
                )
        # Get the latest state
        _, state_body = self.odom_buffer.get_latest()

        # Takeoff condition: ref altitude >= 1cm triggers NMPC solver
        take_off_cond = (self.ref_state[2]) >= 0.01 or (state_body[2] >= 0.01)

        if not take_off_cond:
            # On ground: skip solver, fix thrust at 1N/rotor (6N total)
            self.des_rotor_thrust_mpc = 1.0 * np.ones(6)
            self.nmpc_solver.previous_states = None  # reset warm-start

            f_comp = 1.0 * 6.0
            if self.moment_ff_flag is True:
                M_comp = self.M_ff.copy()
            else:
                M_comp = np.zeros((3,))
            M_comp[2] = 0.0

            u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(
                self.des_rotor_thrust_mpc)

            self.des_rotor_rpm_comp = (self.control_allocator
                                       .compute_relaxed_des_rpm(f_comp, M_comp,
                                                                self.des_rotor_rpm_comp_prev,
                                                                self.control_period))

            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                      self.des_rotor_rpm_comp)

            nmpc_msg = WrenchStamped()
            nmpc_msg.header.stamp = self.get_clock().now().to_msg()
            nmpc_msg.header.frame_id = 'nmpc'
            nmpc_msg.wrench.force.z = f_comp
            nmpc_msg.wrench.torque.x = M_comp[0]
            nmpc_msg.wrench.torque.y = M_comp[1]
            nmpc_msg.wrench.torque.z = M_comp[2]

            self.cmd_pub.publish(cmd_msg)
            self.nmpc_pub.publish(nmpc_msg)
            self.des_rotor_rpm_comp_prev = self.des_rotor_rpm_comp
            return

        # In flight or takeoff: solve NMPC
        solve_start = time.time()
        status, rotor_thrust_nmpc = self.nmpc_solver.solve(
            state = state_body,
            ref = self.ref_state,
            u_prev = self.des_rotor_thrust_mpc
        )
        solve_end = time.time()
        solve_time = (solve_end - solve_start)*1e3  # ms

        # Update control input
        self.des_rotor_thrust_mpc = rotor_thrust_nmpc

        u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(self.des_rotor_thrust_mpc)

        # Check if we have DOB data
        if self.wrench_buffer.is_empty():
            # No DOB data available, just return
            return

        _, wrench_body = self.wrench_buffer.get_latest()
        f_dist = wrench_body[0:3]       # [f_x, f_y, f_z]
        tau_dist = wrench_body[3:6]     # [tau_x, tau_y, tau_z]

        # Airborne detection: actual total thrust >= weight
        airborne = self.actual_total_thrust >= self.W

        if airborne:
            self.was_airborne = True

        # Stay in flight mode during landing until altitude is very low
        in_flight = airborne or (self.was_airborne and state_body[2] > 0.01)

        if in_flight:
            # Full DOB compensation in flight (including landing descent)
            f_comp = u_mpc[0] - f_dist[2]
            M_comp = u_mpc[1:4] - tau_dist
        else:
            if self.was_airborne:
                # Just landed — reset flag
                self.was_airborne = False

            # On ground: NMPC thrust only
            f_comp = u_mpc[0]

            if self.moment_ff_flag is True and state_body[2] < 0.01:
                # On ground with moment feedforward: use feedforward moment without DOB compensation
                M_comp = self.M_ff.copy()
            else:
                # On ground: MPC moment only, no DOB compensation
                M_comp = u_mpc[1:4]

        self.des_rotor_rpm_comp = (self.control_allocator
                                   .compute_relaxed_des_rpm(f_comp, M_comp,
                                                            self.des_rotor_rpm_comp_prev,
                                                            self.control_period))

        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                  self.des_rotor_rpm_comp)

        nmpc_msg = WrenchStamped()
        nmpc_msg.header.stamp = self.get_clock().now().to_msg()
        nmpc_msg.header.frame_id = 'nmpc'
        nmpc_msg.wrench.force.x = 0.0
        nmpc_msg.wrench.force.y = 0.0
        nmpc_msg.wrench.force.z = u_mpc[0]
        nmpc_msg.wrench.torque.x = u_mpc[1]
        nmpc_msg.wrench.torque.y = u_mpc[2]
        nmpc_msg.wrench.torque.z = u_mpc[3]

        self.cmd_pub.publish(cmd_msg)
        self.nmpc_pub.publish(nmpc_msg)

        self.des_rotor_rpm_comp_prev = self.des_rotor_rpm_comp

        # Update statistics
        self.solve_count += 1
        self.total_solve_time += solve_time

        if status != 0:
            self.failure_count += 1
            if self.solve_count % 10 == 0:
                self.get_logger().warn(
                    f'Solver failed! Status: {status}',
                    throttle_duration_sec = 1.0
                )
            return


        # Log statistics periodically (every 100 iterations = 1 second at 100 Hz)
        if self.solve_count % 100 == 0:
            avg_solve_time = self.total_solve_time / self.solve_count
            success_rate = (1.0 - self.failure_count / self.solve_count) * 100.0

            self.get_logger().info(
                f'Stats: solve = {avg_solve_time:.2f} ms, '
                f'success = {success_rate:.1f} %, '
                f'odom_age = {odom_age*1000:.1f} ms'
            )


    def _get_time_now(self) -> float:
        """Get the current ROS time as float (seconds)"""
        clock_now = self.get_clock().now()
        sec, nsec = clock_now.seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _set_rpm_zero(self):
        """Set the cmd rpm to zero"""
        zero_rpm = np.zeros((6,))
        self.des_rotor_rpm_comp_prev[:] = 0
        self.cmd_lpf.reset(zero_rpm)
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                  zero_rpm)
        self.cmd_pub.publish(cmd_msg)

    def _set_idle_rpm(self):
        """Set the idle rpm"""
        idle_rpm = 2000.0 * np.ones((6,))
        self.des_rotor_rpm_comp_prev[:] = 2000
        self.cmd_lpf.reset(idle_rpm)
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                  idle_rpm)
        self.cmd_pub.publish(cmd_msg)

    def _load_parameters(self):
        """
        Load parameters from ROS2 parameter server

        Returns:
            Tuple of (dynamic_param, drone_param, nmpc_param)
        """

        # Dynamic parameters
        m = self.get_parameter('dynamic_param.m').value
        MoiArray = self.get_parameter('dynamic_param.MoiArray').value
        moment_ff = self.get_parameter('dynamic_param.moment_ff').value
        com_offset = self.get_parameter('dynamic_param.com_offset').value

        # Drone parameters
        arm_length = self.get_parameter('drone_param.arm_length').value
        motor_const = self.get_parameter('drone_param.motor_const').value
        moment_const = self.get_parameter('drone_param.moment_const').value
        rotor_max = self.get_parameter('drone_param.rotor_max').value
        rotor_min = self.get_parameter('drone_param.rotor_min').value
        acc_max = self.get_parameter('drone_param.acc_max').value
        acc_min = self.get_parameter('drone_param.acc_min').value

        # NMPC parameters
        t_horizon = self.get_parameter('nmpc_param.t_horizon').value
        n_nodes = self.get_parameter('nmpc_param.n_nodes').value
        QArray = self.get_parameter('nmpc_param.QArray').value
        R = self.get_parameter('nmpc_param.R').value

        # Log parameters
        self.get_logger().info('Parameters loaded:')
        self.get_logger().info(f'  Mass: {m:.2f} kg')
        self.get_logger().info(f'  Inertia: {MoiArray}')
        self.get_logger().info(f'  Moment FF: {moment_ff}')
        self.get_logger().info(f'  Com offset: {com_offset}')
        self.get_logger().info(f'  Arm length: {arm_length:.3f} m')
        self.get_logger().info(f'  Rotor const: {motor_const:.2e}')
        self.get_logger().info(f'  Rotor RPM limits: [{rotor_min:.2f}, {rotor_max:.2f}] N')
        self.get_logger().info(f'  Horizon: {t_horizon:.2f} s, Nodes: {n_nodes}')

        dynamic_param = {
            'm': m,
            'MoiArray': MoiArray,
            'moment_ff': moment_ff,
            'com_offset': com_offset
        }

        drone_param = {
            'arm_length': arm_length,
            'motor_const': motor_const,
            'moment_const': moment_const,
            'rotor_max': rotor_max,
            'rotor_min': rotor_min,
            'acc_max': acc_max,
            'acc_min': acc_min
        }

        nmpc_param = {
            't_horizon': t_horizon,
            'n_nodes': n_nodes,
            'QArray': QArray,
            'R': R
        }

        return dynamic_param, drone_param, nmpc_param

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None
    try:
        # Create node
        node = NmpcWithDOBNode()

        # Use SingleThreadedExecutor for predictable behavior
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Spin
        print('\n[NMPC with DOB] Node running. Press Ctrl+C to stop.\n')
        executor.spin()

    except KeyboardInterrupt:
        print('\n[NMPC with DOB] Keyboard interrupt received')
    except Exception as e:
        print(f'\n[NMPC with DOB] Exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cleanup_acados_files(node.nmpc_solver.get_json_file_name())
        print('[NMPC with DOB] Shutdown complete\n')


if __name__ == '__main__':
    main()