"""
ROS2 NMPC Node with Disturbance Observer (DOB)

This implements NMPC with disturbance compensation from DOB:
-
-
-
-

"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry
from drone_msgs.msg import Ref
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, math_tool, cleanup_acados_files
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

        # Load parameters
        dynamc_param, drone_param, nmpc_param = self._load_parameters()

        # Create NMPC solver
        self.get_logger().info('Creating NMPC solver...')
        self.nmpc_solver = S550_Ocp(DynParam=dynamc_param,
                                    DroneParam=drone_param,
                                    MpcParam=nmpc_param)

        # Create control allocator
        self.control_allocator = ControlAllocator(DroneParam=drone_param)

        # Odometry buffer
        self.odom_buffer = CircularBuffer(capacity=30)

        # Wrench buffer (f_x, f_y, f_z, tau_x, tau_y, tau_z)
        self.wrench_buffer = CircularBuffer(capacity=30)

        # Reference state (p, v, q, w) in 13 dim
        self.ref_state = np.zeros((13,))
        self.ref_state[6] = 1.0     # qw = 1 (Identity quaternion)

        # Statistics for NMPC solver
        self.solve_count = 0
        self.failure_count = 0
        self.total_solve_time = 0.0

        # Flags
        self.solver_ready = False
        self.first_solve = True

        m = self.dynamic_param['m'] if hasattr(self, 'dynamic_param') else 2.9
        u_hover = m * 9.81 / 6.0
        self.des_rotor_thrust_mpc = u_hover * np.ones((6,))
        self.des_rotor_rpm_comp = np.zeros_like(self.des_rotor_thrust_mpc)
        self.des_rotor_rpm_comp_prev = np.zeros_like(self.des_rotor_thrust_mpc)
        self.C_T = self.drone_param['rotor_const'] if hasattr(self, 'drone_param') else 1.465e-7

        # Create publisher
        self.cmd_pub = self.create_publisher(HexaCmdRaw,
                                             '/uav/cmd_raw',
                                             5)

        # Create subscribers
        self.odom_sub = self.create_subscription(Odometry,
                                                 '/filtered_odom',
                                                 callback=self._odom_callback,
                                                 qos_profile=qos_profile_sensor_data)

        self.ref_sub = self.create_subscription(Ref,
                                                '/nmpc/ref',
                                                callback=self._ref_callback,
                                                qos_profile=10)

        self.wrench_sub = self.create_subscription(WrenchStamped,
                                                   '/hgdo/wrench',
                                                   callback=self._wrench_dob_callback,
                                                   qos_profile=qos_profile_sensor_data)

        # Create control timer ( 100 Hz )
        self.control_period = 0.01
        self.control_timer = self.create_timer(self.control_period,
                                               self._control_callback)

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

    def _wrench_dob_callback(self, msg:WrenchStamped):
        """
        DOB wrench callback - stores latest disturbance estimate in buffer

        Wrench format: [f_x, f_y, f_z, tau_x, tau_y, tau_z]
        """
        wrench_time, wrench_data = MsgParser.parse_wrench_msg(msg)

        if self.wrench_buffer.is_full():
            self.wrench_buffer.pop()
        self.wrench_buffer.push((wrench_time, wrench_data))

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

    def _control_callback(self):
        """
        Main control loop callback - runs at 100 Hz

        This is where the NMPC is solved and commands are published
        """

        # Check if we have odomemtry data
        if self.odom_buffer.is_empty():
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
        # Get the latest state
        _, state_body = self.odom_buffer.get_latest()
        state_current = state_body.copy()

        # Transform linear velocity from body to world frame
        v_body = state_current[3:6]
        q = state_current[6:10]
        R_world_body = math_tool.quaternion_to_rotm(q)
        v_world = R_world_body @ v_body
        state_current[3:6] = v_world

        # Solve NMPC
        solve_start = time.time()
        status, rotor_thrust_nmpc = self.nmpc_solver.solve(
            state = state_current,
            ref = self.ref_state,
            u_prev = self.des_rotor_thrust_mpc
        )
        solve_end = time.time()
        solve_time = (solve_end - solve_start)*1e3  # ms

        # Update control input
        self.des_rotor_thrust_mpc = rotor_thrust_nmpc

        u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(self.des_rotor_thrust_mpc)

        # Log for MPC solver output (force and moment)
        # self.get_logger().info('u_mpc = {}'.format(u_mpc))

        # Check if we have DOB data
        if self.wrench_buffer.is_empty():
            # No DOB data available, just return
            return

        _, wrench_body = self.wrench_buffer.get_latest()
        f_dist = wrench_body[0:3]       # [f_x, f_y, f_z]
        tau_dist = wrench_body[3:6]     # [tau_x, tau_y, tau_z]

        f_comp = u_mpc[0] - f_dist[2]
        M_comp = u_mpc[1:4] - tau_dist

        self.get_logger().info(f'f_comp: {f_comp:.3f} N, '
                               f'tau_dist: {M_comp} Nm')


        self.des_rotor_rpm_comp = (self.control_allocator
                                   .compute_relaxed_des_rpm(f_comp, M_comp,
                                    self.des_rotor_rpm_comp,
                                    self.control_period))

        # Convert to RPM and publish
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(),
                                                  self.des_rotor_rpm_comp)
        self.cmd_pub.publish(cmd_msg)

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

    def _load_parameters(self):
        """
        Load parameters from ROS2 parameter server

        Returns:
            Tuple of (dynamic_param, drone_param, nmpc_param)
        """

        # Dynamic parameters
        m = self.get_parameter('dynamic_param.m').value
        MoiArray = self.get_parameter('dynamic_param.MoiArray').value

        # Drone parameters
        arm_length = self.get_parameter('drone_param.arm_length').value
        rotor_const = self.get_parameter('drone_param.rotor_const').value
        moment_const = self.get_parameter('drone_param.moment_const').value
        T_max = self.get_parameter('drone_param.T_max').value
        T_min = self.get_parameter('drone_param.T_min').value
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
        self.get_logger().info(f'  Arm length: {arm_length:.3f} m')
        self.get_logger().info(f'  Rotor const: {rotor_const:.2e}')
        self.get_logger().info(f'  Thrust limits: [{T_min:.2f}, {T_max:.2f}] N')
        self.get_logger().info(f'  Horizon: {t_horizon:.2f} s, Nodes: {n_nodes}')

        dynamic_param = {
            'm': m,
            'MoiArray': MoiArray
        }

        drone_param = {
            'arm_length': arm_length,
            'rotor_const': rotor_const,
            'moment_const': moment_const,
            'T_max': T_max,
            'T_min': T_min,
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

        cleanup_acados_files()
        print('[NMPC with DOB] Shutdown complete\n')


if __name__ == '__main__':
    main()