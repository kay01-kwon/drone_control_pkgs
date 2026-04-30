"""
ROS2 PD Position + NMPC Attitude Node with DOB

Cascaded control architecture:
- Outer loop: PD position controller → desired force → desired attitude + thrust
  - DOB force compensation in world frame (all 3 axes including lateral)
  - Optional integral action for steady-state error rejection
- Inner loop: NMPC attitude controller → moment commands → motor allocation
  - DOB torque compensation

This solves the lateral DOB force compensation problem that the full-state
NMPC cannot handle: PD operates in world frame, so body-frame DOB forces
are rotated to world and directly subtracted from the desired force vector.

Author: Geonwoo Kwon
Date: 2026-04-17
"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, WrenchStamped
from drone_msgs.msg import Ref
from mavros_msgs.msg import RCIn
from ros2_libcanard_msgs.msg import HexaCmdRaw
from ros2_libcanard_msgs.msg import HexaActualRpm

from drone_control.rc_control import RcConverter
from drone_control.rc_control import FlightMode

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, cleanup_acados_files
from drone_control.utils.math_tool import quaternion_to_rotm, quaternion_to_yaw, wrap_pi
from drone_control.nmpc.ocp.S550_att_ocp import S550_att_ocp


def rotm_to_quat(R):
    """Convert rotation matrix to quaternion [qw, qx, qy, qz] (Shepperd method)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def force_to_attitude(F_des, psi):
    """Convert desired force vector (world frame) to desired quaternion and thrust.

    Args:
        F_des: [Fx, Fy, Fz] desired force in world frame [N]
        psi: desired yaw angle [rad]

    Returns:
        q_des: [qw, qx, qy, qz] desired quaternion
        f_col: desired collective thrust magnitude [N]
    """
    f_col = np.linalg.norm(F_des)
    if f_col < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0.0

    z_b = F_des / f_col
    x_c = np.array([np.cos(psi), np.sin(psi), 0.0])

    y_b = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b)
    if y_b_norm < 1e-6:
        y_b = np.array([-np.sin(psi), np.cos(psi), 0.0])
    else:
        y_b /= y_b_norm

    x_b = np.cross(y_b, z_b)
    R_des = np.column_stack((x_b, y_b, z_b))
    q_des = rotm_to_quat(R_des)
    return q_des, f_col


class PdNmpcAttWithDOBNode(Node):
    """
    Cascaded PD Position + NMPC Attitude Node with DOB.
    Outer: PD/PID position → desired attitude + thrust (with DOB force comp)
    Inner: NMPC attitude → moments → motor allocation (with DOB torque comp)
    """
    def __init__(self):
        super().__init__('pd_nmpc_att_with_dob',
                         automatically_declare_parameters_from_overrides=True)

        dynamic_param, drone_param, nmpc_param, pd_param = self._load_parameters()

        self.dynamic_param = dynamic_param
        self.drone_param = drone_param
        self.nmpc_param = nmpc_param

        m = dynamic_param['m']
        self.m = m
        self.g = 9.81
        self.W = m * self.g

        # PD gains
        self.Kp = np.array(pd_param['Kp'])
        self.Kd = np.array(pd_param['Kd'])
        self.Ki = np.array(pd_param['Ki'])
        self.anti_windup = np.array(pd_param['anti_windup'])
        self.p_integral = np.zeros(3)

        # COM offset for moment feedforward
        com_offset = dynamic_param['com_offset']

        # Moment feedforward
        self.moment_ff_flag = dynamic_param['moment_ff']
        self.M_ff = np.array([self.W * com_offset[1],
                              -self.W * com_offset[0],
                              0.0])

        # Create RC converter
        self.rc_converter = RcConverter()

        # Create NMPC attitude solver
        self.get_logger().info('Creating NMPC Attitude solver...')
        self.nmpc_solver = S550_att_ocp(DynParam=dynamic_param,
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

        # Reference state: [px, py, pz, vx, vy, vz, psi, psi_dot]
        self.ref_p = np.zeros(3)
        self.ref_v = np.zeros(3)
        self.ref_psi = 0.0        # target yaw from /nmpc/ref
        self.ref_psi_dot = 0.0    # target yaw rate from /nmpc/ref
        self.ref_psi_ramped = 0.0 # rate-limited yaw reference
        self.dpsi_dt_max = pd_param['dpsi_dt_max']
        self.a_xy_max = pd_param['a_xy_max']
        self.a_z_max = pd_param['a_z_max']

        # Statistics
        self.solve_count = 0
        self.failure_count = 0
        self.total_solve_time = 0.0

        # Flags
        self.solver_ready = False
        self.first_solve = True

        u_hover = self.W / 6.0
        self.des_rotor_thrust_mpc = u_hover * np.ones(6)
        self.des_rotor_rpm_comp = np.zeros(6)
        self.des_rotor_rpm_comp_prev = np.zeros(6)
        self.C_T = drone_param['motor_const']

        self.actual_total_thrust = 0.0
        self.was_airborne = False

        # Initial position offset (from mocap)
        self.px_offset = None
        self.py_offset = None
        self.pz_offset = None

        # Publish predicted trajectory
        self.publish_state = nmpc_param.get('publish_state', False)
        self.n_nodes = nmpc_param['n_nodes']
        self.dt_horizon = nmpc_param['t_horizon'] / nmpc_param['n_nodes']

        # Topic names
        cmd_topic = self.get_parameter('topic_names.cmd_topic').value
        rc_topic = self.get_parameter('topic_names.rc_topic').value
        nmpc_topic = self.get_parameter('topic_names.base_line_control_topic').value
        state_topic = self.get_parameter('topic_names.state_topic').value
        filtered_odom_topic = self.get_parameter('topic_names.filtered_odom_topic').value
        ref_topic = self.get_parameter('topic_names.ref_topic').value
        dob_wrench_topic = self.get_parameter('topic_names.dob_wrench_topic').value

        # Publishers
        self.cmd_pub = self.create_publisher(HexaCmdRaw, cmd_topic, 5)
        self.nmpc_pub = self.create_publisher(WrenchStamped, nmpc_topic, 5)
        if self.publish_state:
            self.state_pub = self.create_publisher(Path, state_topic, 5)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, filtered_odom_topic,
            self._odom_callback, qos_profile_sensor_data)
        self.rc_sub = self.create_subscription(
            RCIn, rc_topic,
            self._rc_callback, qos_profile_sensor_data)
        self.ref_sub = self.create_subscription(
            Ref, ref_topic,
            self._ref_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, dob_wrench_topic,
            self._wrench_dob_callback, qos_profile_sensor_data)
        self.actual_rpm_sub = self.create_subscription(
            HexaActualRpm, '/uav/actual_rpm',
            self._actual_rpm_callback, qos_profile_sensor_data)

        # Control timer (100 Hz)
        self.control_period = 0.01
        self.control_timer = self.create_timer(self.control_period,
                                                self._control_callback)

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'PD + NMPC Attitude with DOB')
        self.get_logger().info(f'Kp: {self.Kp.tolist()}, Kd: {self.Kd.tolist()}')
        self.get_logger().info(f'Ki: {self.Ki.tolist()}, anti_windup: {self.anti_windup.tolist()}')
        self.get_logger().info(f'Moment FF: {self.moment_ff_flag}')
        self.get_logger().info(f'Att MPC Q: {nmpc_param["QArray"]}, R: {nmpc_param["R"]}')
        self.get_logger().info(f'Control rate: {1.0/self.control_period:.1f} Hz')
        self.get_logger().info('=' * 60)

    # ─── Callbacks ────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        if self.pz_offset is None:
            self.px_offset = odom_data[0]
            self.py_offset = odom_data[1]
            self.pz_offset = odom_data[2]
            self.get_logger().info(
                f'Initial offset: {self.px_offset:.4f}, '
                f'{self.py_offset:.4f}, {self.pz_offset:.4f} m')
        odom_data[0] -= self.px_offset
        odom_data[1] -= self.py_offset
        odom_data[2] -= self.pz_offset

        if odom_data[2] < 0.0:
            odom_data[2] = 0.0

        if self.odom_buffer.is_full():
            self.odom_buffer.pop()
        self.odom_buffer.push((odom_time, odom_data))

    def _rc_callback(self, msg: RCIn):
        rc_time, rc_state = MsgParser.parse_rc_msg(msg)
        self.rc_converter.set_rc(rc_state)
        self.mode, _, _ = self.rc_converter.get_rc_state()
        if self.mode is not self.prev_mode:
            self.get_logger().info(f'Mode: {self.mode}')
        self.prev_mode = self.mode

    def _ref_callback(self, msg: Ref):
        self.ref_p[0:3] = msg.p
        self.ref_v[0:3] = msg.v
        self.ref_psi = msg.psi
        self.ref_psi_dot = msg.psi_dot

    def _wrench_dob_callback(self, msg: WrenchStamped):
        wrench_time, wrench_data = MsgParser.parse_wrench_msg(msg)
        if self.wrench_buffer.is_full():
            self.wrench_buffer.pop()
        self.wrench_buffer.push((wrench_time, wrench_data))

    def _actual_rpm_callback(self, msg: HexaActualRpm):
        rpms = np.array(msg.rpm, dtype=np.float64)
        self.actual_total_thrust = self.C_T * np.sum(rpms ** 2)

    # ─── Control Loop ─────────────────────────────────────────

    def _control_callback(self):
        if self.odom_buffer.is_empty():
            return

        if self.mode == FlightMode.KILL:
            self._set_rpm_zero()
            self.p_integral[:] = 0.0
            _, st = self.odom_buffer.get_latest()
            self.ref_psi_ramped = quaternion_to_yaw(st[6:10])
            return
        elif self.mode == FlightMode.DISARMED:
            self._set_rpm_zero()
            self.p_integral[:] = 0.0
            _, st = self.odom_buffer.get_latest()
            self.ref_psi_ramped = quaternion_to_yaw(st[6:10])
            return
        elif self.mode in (FlightMode.ARMED, FlightMode.MANUAL_STAB):
            self.ref_p[2] = 0.0

        current_time = self._get_time_now()
        odom_age = current_time - self.odom_buffer.get_latest()[0]
        if odom_age > 0.05 and self.solve_count % 100 == 0:
            self.get_logger().warn(
                f'Stale odometry! Age: {odom_age*1000:.1f} ms',
                throttle_duration_sec=1.0)

        _, state = self.odom_buffer.get_latest()
        p = state[0:3]
        v = state[3:6]
        q = state[6:10]   # [qw, qx, qy, qz]
        w = state[10:13]   # [wx, wy, wz] body frame

        R_wb = quaternion_to_rotm(q)

        take_off_cond = self.ref_p[2] >= 0.01 or state[2] >= 0.01

        if not take_off_cond:
            self.des_rotor_thrust_mpc = 6.0 * np.ones(6)
            self.nmpc_solver.previous_states = None
            self.p_integral[:] = 0.0

            f_comp = 6.0
            if self.moment_ff_flag:
                M_comp = self.M_ff.copy()
            else:
                M_comp = np.zeros(3)
            M_comp[2] = 0.0

            self.des_rotor_rpm_comp = (self.control_allocator
                                       .compute_relaxed_des_rpm(
                                           f_comp, M_comp,
                                           self.des_rotor_rpm_comp_prev,
                                           self.control_period))

            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
                self.get_clock().now(), self.des_rotor_rpm_comp)

            nmpc_msg = WrenchStamped()
            nmpc_msg.header.stamp = self.get_clock().now().to_msg()
            nmpc_msg.header.frame_id = 'pd_nmpc_att'
            nmpc_msg.wrench.force.z = f_comp
            nmpc_msg.wrench.torque.x = M_comp[0]
            nmpc_msg.wrench.torque.y = M_comp[1]
            nmpc_msg.wrench.torque.z = M_comp[2]

            self.cmd_pub.publish(cmd_msg)
            self.nmpc_pub.publish(nmpc_msg)
            self.des_rotor_rpm_comp_prev = self.des_rotor_rpm_comp
            return

        # ── Outer loop: PD position → desired force ──

        e_p = self.ref_p - p
        e_v = self.ref_v - R_wb @ v

        # Integral with anti-windup
        self.p_integral += self.Ki * e_p * self.control_period
        self.p_integral = np.clip(self.p_integral,
                                   -self.anti_windup,
                                    self.anti_windup)

        a_des = self.Kp * e_p + self.Kd * e_v + self.p_integral

        a_xy_norm = np.linalg.norm(a_des[0:2])
        if a_xy_norm > self.a_xy_max:
            a_des[0:2] *= self.a_xy_max / a_xy_norm
        a_des[2] = np.clip(a_des[2], -self.a_z_max, self.a_z_max)

        a_des[2] += self.g

        F_des = self.m * a_des

        # DOB force compensation (body → world)
        if not self.wrench_buffer.is_empty():
            _, wrench_body = self.wrench_buffer.get_latest()
            f_dist = wrench_body[0:3]
            tau_dist = wrench_body[3:6]
            F_des -= R_wb @ f_dist
        else:
            tau_dist = np.zeros(3)

        # Ramp yaw reference
        psi_err = wrap_pi(self.ref_psi - self.ref_psi_ramped)
        max_step = self.dpsi_dt_max * self.control_period
        psi_err = np.clip(psi_err, -max_step, max_step)
        self.ref_psi_ramped = wrap_pi(self.ref_psi_ramped + psi_err)

        # Force → desired attitude + thrust
        q_des, f_col = force_to_attitude(F_des, self.ref_psi_ramped)

        # Desired angular velocity reference (yaw rate only)
        ref_att = np.array([q_des[0], q_des[1], q_des[2], q_des[3],
                            0.0, 0.0, self.ref_psi_dot])

        # ── Inner loop: NMPC attitude ──

        state_att = np.concatenate((q, w))

        solve_start = time.time()
        status, rotor_thrust_nmpc = self.nmpc_solver.solve(
            state=state_att,
            ref=ref_att,
            u_prev=self.des_rotor_thrust_mpc,
            f_col=f_col
        )
        solve_end = time.time()
        solve_time = (solve_end - solve_start) * 1e3

        self.solve_count += 1
        self.total_solve_time += solve_time

        if status != 0:
            self.failure_count += 1
            if self.solve_count % 10 == 0:
                self.get_logger().warn(
                    f'Solver failed! Status: {status}',
                    throttle_duration_sec=1.0)
            return

        self.des_rotor_thrust_mpc = rotor_thrust_nmpc
        u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(
            self.des_rotor_thrust_mpc)

        # Airborne detection
        airborne = self.actual_total_thrust >= self.W
        if airborne:
            self.was_airborne = True
        in_flight = airborne or (self.was_airborne and state[2] > 0.01)

        if in_flight:
            f_comp_final = f_col
            M_comp = u_mpc[1:4] - tau_dist
        else:
            if self.was_airborne:
                self.was_airborne = False
            f_comp_final = u_mpc[0]
            if self.moment_ff_flag and state[2] < 0.01:
                M_comp = self.M_ff.copy()
            else:
                M_comp = u_mpc[1:4] - tau_dist
                M_comp[2] = 0.0

        self.des_rotor_rpm_comp = (self.control_allocator
                                   .compute_relaxed_des_rpm(
                                       f_comp_final, M_comp,
                                       self.des_rotor_rpm_comp_prev,
                                       self.control_period))

        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
            self.get_clock().now(), self.des_rotor_rpm_comp)

        nmpc_msg = WrenchStamped()
        nmpc_msg.header.stamp = self.get_clock().now().to_msg()
        nmpc_msg.header.frame_id = 'pd_nmpc_att'
        nmpc_msg.wrench.force.x = F_des[0]
        nmpc_msg.wrench.force.y = F_des[1]
        nmpc_msg.wrench.force.z = f_col
        nmpc_msg.wrench.torque.x = u_mpc[1]
        nmpc_msg.wrench.torque.y = u_mpc[2]
        nmpc_msg.wrench.torque.z = u_mpc[3]

        self.cmd_pub.publish(cmd_msg)
        self.nmpc_pub.publish(nmpc_msg)

        if self.publish_state and self.nmpc_solver.previous_states is not None:
            self._publish_predicted_path()

        self.des_rotor_rpm_comp_prev = self.des_rotor_rpm_comp

        if self.solve_count % 100 == 0:
            avg_solve_time = self.total_solve_time / self.solve_count
            success_rate = (1.0 - self.failure_count / self.solve_count) * 100.0
            self.get_logger().info(
                f'Stats: solve = {avg_solve_time:.2f} ms, '
                f'success = {success_rate:.1f} %, '
                f'odom_age = {odom_age*1000:.1f} ms, '
                f'f_col = {f_col:.1f} N, '
                f'int = [{self.p_integral[0]:.3f}, {self.p_integral[1]:.3f}, {self.p_integral[2]:.3f}]'
            )

    # ─── Helpers ──────────────────────────────────────────────

    def _publish_predicted_path(self):
        now = self.get_clock().now()
        sec, nsec = now.seconds_nanoseconds()
        path_msg = Path()
        path_msg.header.stamp = now.to_msg()
        path_msg.header.frame_id = 'S550/odom'
        for i, x in enumerate(self.nmpc_solver.previous_states):
            ps = PoseStamped()
            t_offset_ns = int(i * self.dt_horizon * 1e9)
            ps.header.stamp.sec = sec + (nsec + t_offset_ns) // 1_000_000_000
            ps.header.stamp.nanosec = (nsec + t_offset_ns) % 1_000_000_000
            ps.header.frame_id = 'S550/odom'
            ps.pose.orientation.w = x[0]
            ps.pose.orientation.x = x[1]
            ps.pose.orientation.y = x[2]
            ps.pose.orientation.z = x[3]
            path_msg.poses.append(ps)
        self.state_pub.publish(path_msg)

    def _get_time_now(self) -> float:
        clock_now = self.get_clock().now()
        sec, nsec = clock_now.seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _set_rpm_zero(self):
        zero_rpm = np.zeros(6)
        self.des_rotor_rpm_comp_prev[:] = 0
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
            self.get_clock().now(), zero_rpm)
        self.cmd_pub.publish(cmd_msg)

    def _load_parameters(self):
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

        # NMPC attitude parameters
        publish_state = self.get_parameter('nmpc_param.publish_state').value
        t_horizon = self.get_parameter('nmpc_param.t_horizon').value
        n_nodes = self.get_parameter('nmpc_param.n_nodes').value
        QArray = self.get_parameter('nmpc_param.QArray').value
        R = self.get_parameter('nmpc_param.R').value
        tanh_k = self.get_parameter('nmpc_param.tanh_k').value

        # PD parameters
        Kp = self.get_parameter('pd_param.Kp').value
        Kd = self.get_parameter('pd_param.Kd').value
        Ki = self.get_parameter('pd_param.Ki').value
        anti_windup = self.get_parameter('pd_param.anti_windup').value
        dpsi_dt_max = self.get_parameter('pd_param.dpsi_dt_max').value
        a_xy_max = self.get_parameter('pd_param.a_xy_max').value
        a_z_max = self.get_parameter('pd_param.a_z_max').value
        self.get_logger().info('Parameters loaded:')
        self.get_logger().info(f'  Mass: {m:.2f} kg')
        self.get_logger().info(f'  Inertia: {MoiArray}')
        self.get_logger().info(f'  Arm length: {arm_length:.3f} m')
        self.get_logger().info(f'  PD Kp: {Kp}, Kd: {Kd}')
        self.get_logger().info(f'  PD Ki: {Ki}, anti_windup: {anti_windup}')
        self.get_logger().info(f'  dpsi_dt_max: {dpsi_dt_max} rad/s')
        self.get_logger().info(f'  a_xy_max: {a_xy_max} m/s^2, a_z_max: {a_z_max} m/s^2')

        dynamic_param = {
            'm': m, 'MoiArray': MoiArray,
            'moment_ff': moment_ff, 'com_offset': com_offset,
        }
        drone_param = {
            'arm_length': arm_length, 'motor_const': motor_const,
            'moment_const': moment_const, 'rotor_max': rotor_max,
            'rotor_min': rotor_min, 'acc_max': acc_max, 'acc_min': acc_min,
        }
        nmpc_param = {
            'publish_state': publish_state,
            't_horizon': t_horizon, 'n_nodes': n_nodes,
            'QArray': QArray, 'R': R, 'tanh_k': tanh_k,
        }
        pd_param = {
            'Kp': Kp, 'Kd': Kd, 'Ki': Ki, 'anti_windup': anti_windup,
            'dpsi_dt_max': dpsi_dt_max,
            'a_xy_max': a_xy_max, 'a_z_max': a_z_max,
        }
        return dynamic_param, drone_param, nmpc_param, pd_param


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PdNmpcAttWithDOBNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        print('\n[PD+NMPC Att with DOB] Node running. Press Ctrl+C to stop.\n')
        executor.spin()
    except KeyboardInterrupt:
        print('\n[PD+NMPC Att with DOB] Keyboard interrupt received')
    except Exception as e:
        print(f'\n[PD+NMPC Att with DOB] Exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            cleanup_acados_files(node.nmpc_solver.get_json_file_name())
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('[PD+NMPC Att with DOB] Shutdown complete\n')


if __name__ == '__main__':
    main()
