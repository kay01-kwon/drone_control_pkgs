"""
ROS2 NMPC ATTITUDE Node with Disturbance Observer (DOB)

This implements NMPC with disturbance compensation from DOB:
- Receives desired collective thrust (Float64) from external source
- Receives rotational disturbances from DOB (HGDO, L1 adaptaiton, EKF/UKF)
- Compensates control input for estimated disturbance (moment only)
- Reference mode: fixed hover ref or external des_odom (Odometry)
- FlightMode-based control (DISARMED: idle, AUTO: NMPC attitude)
- Timer-based control loop (no manual threading)
- SingleThreadedExecutor for predictable behavior

Author: Geonwoo Kwon
Date: 2026-03-05
"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped
from mavros_msgs.msg import RCIn
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, math_tool, cleanup_acados_files
from drone_control.nmpc.ocp.S550_att_ocp import S550_att_ocp
from drone_control.rc_control import RcConverter, FlightMode, RcModeStr

class NMPCAttitudeWithDOB(Node):
    """
    NMPC Attitude Node with DOB integration for ROS2
    This node implements Model Predictive control with disturbance compensation.

    FlightMode behavior:
    - KILL: All rotors stop (0 RPM)
    - DISARMED: Idle speed (2000 RPM)
    - AUTO: NMPC attitude control with external thrust command

    Control loop runs at 100 Hz using ROS2 timer callback.
    """
    def __init__(self):
        super().__init__('nmpc_att_with_dob',
                         automatically_declare_parameters_from_overrides=True)

        # Load parameters
        (dynamc_param, drone_param, nmpc_param,
         rc_converter_param, reference_param) = self._load_parameters()

        # Store parameters as instance variables
        self.dynamic_param = dynamc_param
        self.drone_param = drone_param
        self.nmpc_param = nmpc_param

        m = self.dynamic_param['m']
        self.mg = m * 9.81

        # Reference mode
        self.use_fixed_ref = reference_param['use_fixed_ref']

        # DOB compensation enable flag
        self.use_dob_compensation = reference_param['use_dob_compensation']

        # Create RC converter
        self.rc_converter = RcConverter(rc_converter_param)

        # Create NMPC solver
        self.get_logger().info('Creating NMPC Att solver...')
        self.nmpc_solver = S550_att_ocp(DynParam=dynamc_param,
                                        DroneParam=drone_param,
                                        MpcParam=nmpc_param)

        # Create control allocator
        self.control_allocator = ControlAllocator(DroneParam=drone_param)

        # Odometry buffer
        self.odom_buffer = CircularBuffer(capacity=30)

        # Wrench buffer (f_x, f_y, f_z, tau_x, tau_y, tau_z)
        self.wrench_buffer = CircularBuffer(capacity=30)

        # RC buffer
        self.rc_buffer = CircularBuffer(capacity=30)

        # Reference state (q, w) in 7 dim
        # Default reference: q = (1, 0, 0, 0), w = (0, 0, 0)
        self.ref_state = np.zeros((7,))
        self.ref_state[0] = 1.0  # qw = 1 (Identity quaternion)

        # Statistics for NMPC solver
        self.solve_count = 0
        self.failure_count = 0
        self.total_solve_time = 0.0

        # Flags
        self.solver_ready = False
        self.first_solve = True

        u_hover = m * 9.81 / 6.0
        self.des_rotor_thrust_mpc = u_hover * np.ones((6,))
        self.des_rotor_rpm_comp = np.zeros_like(self.des_rotor_thrust_mpc)
        self.des_rotor_rpm_comp_prev = np.zeros_like(self.des_rotor_thrust_mpc)
        self.C_T = self.drone_param['motor_const']

        # Desired collective thrust (from external Float64 topic)
        self.f_col = 0.0

        # Flight mode
        self.mode = FlightMode.DISARMED
        self.prev_mode = self.mode

        # Topic name from ros param
        cmd_topic = self.get_parameter('topic_names.cmd_topic').value
        nmpc_topic = self.get_parameter('topic_names.base_line_control_topic').value
        filtered_odom_topic = self.get_parameter('topic_names.filtered_odom_topic').value
        rc_topic = self.get_parameter('topic_names.rc_topic').value
        dob_wrench_topic = self.get_parameter('topic_names.dob_wrench_topic').value
        des_thrust_topic = self.get_parameter('topic_names.des_thrust_topic').value

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

        if self.use_dob_compensation:
            self.wrench_sub = self.create_subscription(WrenchStamped,
                                                       dob_wrench_topic,
                                                       callback=self._wrench_dob_callback,
                                                       qos_profile=qos_profile_sensor_data)

        # Desired thrust subscriber (Float64)
        self.des_thrust_sub = self.create_subscription(
            Float64,
            des_thrust_topic,
            callback=self._des_thrust_callback,
            qos_profile=qos_profile_sensor_data)

        # Desired odom subscriber (for ref quaternion + angular velocity)
        if not self.use_fixed_ref:
            des_odom_topic = self.get_parameter('topic_names.des_odom_topic').value
            self.des_odom_sub = self.create_subscription(
                Odometry,
                des_odom_topic,
                callback=self._des_odom_callback,
                qos_profile=qos_profile_sensor_data)
            self.get_logger().info(f'Des odom topic: {des_odom_topic}')

        # Create control timer ( 100 Hz )
        self.control_period = 0.01
        self.control_timer = self.create_timer(self.control_period,
                                               self._control_callback)

        self.get_logger().info('='*60)
        self.get_logger().info(f'Command topic: {cmd_topic}')
        self.get_logger().info(f'NMPC topic: {nmpc_topic}')
        self.get_logger().info(f'Filtered odom topic: {filtered_odom_topic}')
        self.get_logger().info(f'RC topic: {rc_topic}')
        self.get_logger().info(f'DoB wrench topic: {dob_wrench_topic}')
        self.get_logger().info(f'Des thrust topic: {des_thrust_topic}')
        self.get_logger().info(f'Use fixed ref: {self.use_fixed_ref}')
        self.get_logger().info(f'Use DOB compensation: {self.use_dob_compensation}')
        self.get_logger().info('NMPC Attitude With DOB Node initialized successfully')
        self.get_logger().info(f'Control rate: {1.0/self.control_period:.1f} Hz')
        self.get_logger().info(f'Horizon: {nmpc_param["t_horizon"]:.2f}s')
        self.get_logger().info(f'Nodes: {nmpc_param["n_nodes"]}')
        self.get_logger().info(f'Q: {nmpc_param["QArray"]}')
        self.get_logger().info(f'R: {nmpc_param["R"]}')
        self.get_logger().info('='*60)

    def _odom_callback(self, msg: Odometry):
        """
        Odometry callback - stores latest state in buffer

        State format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        Note: Linear and angular velocity is in Body frame from odometry
        """

        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        if self.odom_buffer.is_full():
            self.odom_buffer.pop()
        self.odom_buffer.push((odom_time, odom_data))

    def _rc_callback(self, msg: RCIn):
        """
        RC callback - updates flight mode from RC switches.
        """
        rc_time, rc_state = MsgParser.parse_rc_msg(msg)

        self.rc_converter.set_rc(rc_state)
        self.mode, _, _ = self.rc_converter.get_rc_state()

        if self.rc_buffer.is_full():
            self.rc_buffer.pop()
        self.rc_buffer.push((rc_time, rc_state))

        if self.mode is not self.prev_mode:
            mode_name = RcModeStr.mode_str(self.mode)
            self.get_logger().info(f'Mode: {mode_name}')

        self.prev_mode = self.mode

    def _wrench_dob_callback(self, msg: WrenchStamped):
        """
        DOB wrench callback - stores latest disturbance estimate in buffer

        Wrench format: [f_x, f_y, f_z, tau_x, tau_y, tau_z]
        Only moments (tau_x, tau_y, tau_z) are used for compensation.
        """
        wrench_time, wrench_data = MsgParser.parse_wrench_msg(msg)

        if self.wrench_buffer.is_full():
            self.wrench_buffer.pop()
        self.wrench_buffer.push((wrench_time, wrench_data))

    def _des_thrust_callback(self, msg: Float64):
        """
        Desired collective thrust callback (Float64).
        Receives total thrust [N] from external position controller.
        """
        self.f_col = msg.data

    def _des_odom_callback(self, msg: Odometry):
        """
        Desired odometry callback for reference state.
        Extracts quaternion (qw, qx, qy, qz) and angular velocity (wx, wy, wz)
        from the Odometry message to set as NMPC reference.
        """
        q = msg.pose.pose.orientation
        w = msg.twist.twist.angular
        self.ref_state[0] = q.w
        self.ref_state[1] = q.x
        self.ref_state[2] = q.y
        self.ref_state[3] = q.z
        self.ref_state[4] = w.x
        self.ref_state[5] = w.y
        self.ref_state[6] = w.z

    def _control_callback(self):
        """
        Main control loop callback - runs at 100 Hz

        KILL: 0 RPM
        DISARMED: idle 2000 RPM
        AUTO: NMPC attitude + DOB compensation (thrust from external topic)
        """

        # Check if we have data
        if self.odom_buffer.is_empty():
            return

        if self.rc_buffer.is_empty():
            return

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

        # --- KILL mode ---
        if self.mode == FlightMode.KILL:
            des_rpm = np.zeros((6,))
            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
                self.get_clock().now(), des_rpm)
            self.cmd_pub.publish(cmd_msg)
            return

        # --- DISARMED mode ---
        if self.mode == FlightMode.DISARMED:
            des_rpm = 2000.0 * np.ones((6,))
            cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
                self.get_clock().now(), des_rpm)
            self.cmd_pub.publish(cmd_msg)
            return

        # --- AUTO mode: NMPC attitude with external thrust ---
        if self.mode != FlightMode.AUTO:
            return

        # Get the latest state (full 13-dim odom)
        _, state_full = self.odom_buffer.get_latest()

        # Extract attitude state: [qw, qx, qy, qz, wx, wy, wz]
        state_att = np.concatenate((state_full[6:10], state_full[10:13]))

        # Solve NMPC with collective thrust constraint
        solve_start = time.time()
        status, rotor_thrust_nmpc = self.nmpc_solver.solve(
            state = state_att,
            ref = self.ref_state,
            u_prev = self.des_rotor_thrust_mpc,
            f_col = self.f_col
        )
        solve_end = time.time()
        solve_time = (solve_end - solve_start)*1e3  # ms

        # Update control input
        self.des_rotor_thrust_mpc = rotor_thrust_nmpc

        u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(self.des_rotor_thrust_mpc)

        # Compensate: total thrust from external, moments from NMPC (minus DOB)
        f_comp = self.f_col

        if self.use_dob_compensation:
            # Check if we have DOB data
            if self.wrench_buffer.is_empty():
                return

            _, wrench_body = self.wrench_buffer.get_latest()
            tau_dist = wrench_body[3:6]     # [tau_x, tau_y, tau_z]

            M_comp = u_mpc[1:4] - tau_dist
        else:
            # No DOB compensation: use NMPC moments directly
            M_comp = u_mpc[1:4]

        self.des_rotor_rpm_comp = (self.control_allocator
                                   .compute_des_rpm(f_comp, M_comp))

        # Convert to RPM and publish
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                  self.des_rotor_rpm_comp)

        nmpc_msg = WrenchStamped()
        nmpc_msg.header.stamp = self.get_clock().now().to_msg()
        nmpc_msg.header.frame_id = 'nmpc_att'
        nmpc_msg.wrench.force.x = 0.0
        nmpc_msg.wrench.force.y = 0.0
        nmpc_msg.wrench.force.z = u_mpc[0]
        nmpc_msg.wrench.torque.x = u_mpc[1]
        nmpc_msg.wrench.torque.y = u_mpc[2]
        nmpc_msg.wrench.torque.z = u_mpc[3]

        self.cmd_pub.publish(cmd_msg)
        self.nmpc_pub.publish(nmpc_msg)

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
                f'odom_age = {odom_age*1000:.1f} ms, '
                f'f_col = {self.f_col:.2f} N'
            )

    def _get_time_now(self) -> float:
        """Get the current ROS time as float (seconds)"""
        clock_now = self.get_clock().now()
        sec, nsec = clock_now.seconds_nanoseconds()
        return sec + nsec * 1e-9

    def _load_parameters(self):
        """
        Load parameters from ROS2 parameter server

        Returns:
            Tuple of (dynamic_param, drone_param, nmpc_param,
                      rc_converter_param, reference_param)
        """

        # Dynamic parameters
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

        # NMPC parameters
        t_horizon = self.get_parameter('nmpc_param.t_horizon').value
        n_nodes = self.get_parameter('nmpc_param.n_nodes').value
        QArray = self.get_parameter('nmpc_param.QArray').value
        R = self.get_parameter('nmpc_param.R').value

        # RC converter parameters
        vxy_max = self.get_parameter('rc_converter_param.vxy_max').value
        vz_max = self.get_parameter('rc_converter_param.vz_max').value
        dpsi_dt_max = self.get_parameter('rc_converter_param.dpsi_dt_max').value

        # Reference parameters
        use_fixed_ref = self.get_parameter('reference_param.use_fixed_ref').value

        # DOB compensation enable flag (default True for backward compat)
        if self.has_parameter('reference_param.use_dob_compensation'):
            use_dob_compensation = self.get_parameter(
                'reference_param.use_dob_compensation').value
        else:
            use_dob_compensation = True

        # Log parameters
        self.get_logger().info('Parameters loaded:')
        self.get_logger().info(f'  Mass: {m:.2f} kg')
        self.get_logger().info(f'  Inertia: {MoiArray}')
        self.get_logger().info(f'  Arm length: {arm_length:.3f} m')
        self.get_logger().info(f'  Rotor const: {motor_const:.2e}')
        self.get_logger().info(f'  Rotor RPM limits: [{rotor_min:.2f}, {rotor_max:.2f}] N')
        self.get_logger().info(f'  Horizon: {t_horizon:.2f} s, Nodes: {n_nodes}')
        self.get_logger().info(f'  Use fixed ref: {use_fixed_ref}')

        dynamic_param = {
            'm': m,
            'MoiArray': MoiArray,
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

        rc_converter_param = {
            'vxy_max': vxy_max,
            'vz_max': vz_max,
            'dpsi_dt_max': dpsi_dt_max
        }

        reference_param = {
            'use_fixed_ref': use_fixed_ref,
            'use_dob_compensation': use_dob_compensation
        }

        return dynamic_param, drone_param, nmpc_param, rc_converter_param, reference_param

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None
    try:
        # Create node
        node = NMPCAttitudeWithDOB()

        # Use SingleThreadedExecutor for predictable behavior
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Spin
        print('\n[NMPC Att with DOB] Node running. Press Ctrl+C to stop.\n')
        executor.spin()

    except KeyboardInterrupt:
        print('\n[NMPC Att with DOB] Keyboard interrupt received')
    except Exception as e:
        print(f'\n[NMPC Att with DOB] Exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if node is not None:
            cleanup_acados_files(node.nmpc_solver.get_json_file_name())
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print('[NMPC Att with DOB] Shutdown complete\n')

if __name__ == '__main__':
    main()
