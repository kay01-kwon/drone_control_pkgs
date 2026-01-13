"""
ROS2 NMPC Controller Node

This is a clean rewrite of the NMPC controller for ROS2, following ROS2 best practices:
- Timer-based control loop (no manual threading)
- SingleThreadedExecutor for predictable behavior
- Proper state management
- Clean initialization and shutdown

Author: Claude
Date: 2025-12-28
"""

import os
import shutil
import numpy as np
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from nav_msgs.msg import Odometry
from drone_msgs.msg import Ref
from ros2_libcanard_msgs.msg import HexaCmdRaw
from geometry_msgs.msg import WrenchStamped

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, math_tool, cleanup_acados_files
from drone_control.nmpc.ocp.S550_simple_ocp import S550_Ocp

class NmpcNodeV2(Node):
    """
    NMPC Controller Node for ROS2

    This node implements Model Predictive Control using Acados solver.
    Control loop runs at 100Hz using ROS2 timer callback.
    """

    def __init__(self):
        super().__init__('nmpc_node_v2',
                         automatically_declare_parameters_from_overrides=True)

        # Load parameters and initialize solver
        self.dynamic_param, self.drone_param, self.nmpc_param = self._load_parameters()

        # Initialize state variables
        self._initialize_state()

        # Create NMPC solver
        self.get_logger().info('Creating NMPC solver...')
        self.nmpc_solver = S550_Ocp(
            DynParam=self.dynamic_param,
            DroneParam=self.drone_param,
            MpcParam=self.nmpc_param
        )

        self.control_allocator = ControlAllocator(self.drone_param)

        self.get_logger().info('NMPC solver created successfully')

        # Topic name from ros param
        cmd_topic = self.get_parameter('topic_names.cmd_topic').value
        nmpc_topic = self.get_parameter('topic_names.base_line_control_topic').value
        filtered_odom_topic = self.get_parameter('topic_names.filtered_odom_topic').value
        ref_topic = self.get_parameter('topic_names.ref_topic').value

        # Create publishers
        self.cmd_pub = self.create_publisher(
            HexaCmdRaw,
            cmd_topic,
            5
        )

        self.nmpc_pub = self.create_publisher(WrenchStamped,
                                              nmpc_topic,
                                              qos_profile=5)

        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            filtered_odom_topic,
            self._odom_callback,
            5
        )

        self.ref_sub = self.create_subscription(
            Ref,
            ref_topic,
            self._ref_callback,
            5
        )

        # Create control timer (100 Hz)
        self.control_dt = 0.01  # 10ms
        self.control_timer = self.create_timer(
            self.control_dt,
            self._control_callback
        )

        # Statistics
        self.solve_count = 0
        self.fail_count = 0
        self.total_solve_time = 0.0

        self.get_logger().info('='*60)
        self.get_logger().info(f'Command topic: {cmd_topic}')
        self.get_logger().info(f'NMPC topic: {nmpc_topic}')
        self.get_logger().info(f'Filtered odom topic: {filtered_odom_topic}')
        self.get_logger().info(f'Reference topic: {ref_topic}')
        self.get_logger().info('NMPC Node V2 initialized successfully')
        self.get_logger().info(f'Control rate: {1.0/self.control_dt:.1f} Hz')
        self.get_logger().info(f'Horizon: {self.nmpc_param["t_horizon"]:.2f}s')
        self.get_logger().info(f'Nodes: {self.nmpc_param["n_nodes"]}')
        self.get_logger().info('='*60)

    def _initialize_state(self):
        """Initialize all state variables"""
        # Odometry buffer
        self.odom_buffer = CircularBuffer(capacity=30)
        self.last_odom_time = 0.0

        # Reference state (p, v, q, w)
        self.ref_state = np.zeros(13)
        self.ref_state[6] = 1.0  # qw = 1 (identity quaternion)

        # Control input (rotor thrusts)
        m = self.dynamic_param['m'] if hasattr(self, 'dynamic_param') else 2.9
        self.u_hover = m * 9.81 / 6.0
        self.des_rotor_thrust = self.u_hover * np.ones(6)

        # Rotor constant
        self.C_T = self.drone_param['motor_const'] if hasattr(self, 'drone_param') else 1.465e-7

        # Flags
        self.solver_ready = False
        self.first_solve = True

    def _load_parameters(self) -> Tuple[dict, dict, dict]:
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
        motor_const = self.get_parameter('drone_param.motor_const').value
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
        self.get_logger().info(f'  Rotor const: {motor_const:.2e}')
        self.get_logger().info(f'  Thrust limits: [{T_min:.2f}, {T_max:.2f}] N')
        self.get_logger().info(f'  Horizon: {t_horizon:.2f} s, Nodes: {n_nodes}')

        dynamic_param = {
            'm': m,
            'MoiArray': MoiArray
        }

        drone_param = {
            'arm_length': arm_length,
            'motor_const': motor_const,
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

    def _odom_callback(self, msg: Odometry):
        """
        Odometry callback - stores latest state in buffer

        State format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        Note: Velocity is in BODY frame from odometry
        """
        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        # Update buffer
        if self.odom_buffer.is_full():
            self.odom_buffer.pop()
        self.odom_buffer.push((odom_time, odom_data))

        self.last_odom_time = odom_time

    def _ref_callback(self, msg: Ref):
        """
        Reference callback - updates desired state

        Reference format: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        """
        # Position and velocity
        self.ref_state[0:3] = msg.p
        self.ref_state[3:6] = msg.v

        # Quaternion from yaw angle
        psi = msg.psi
        self.ref_state[6] = np.cos(psi / 2.0)  # qw
        self.ref_state[7] = 0.0                 # qx
        self.ref_state[8] = 0.0                 # qy
        self.ref_state[9] = np.sin(psi / 2.0)  # qz

        # Angular velocity
        self.ref_state[10] = 0.0      # wx
        self.ref_state[11] = 0.0      # wy
        self.ref_state[12] = msg.psi_dot  # wz

    def _control_callback(self):
        """
        Main control loop callback - runs at 100 Hz

        This is where the NMPC is solved and commands are published
        """
        # Check if we have odometry data
        if self.odom_buffer.is_empty():
            return

        # Get current time
        current_time = self._get_time_now()

        # Check odometry freshness
        odom_age = current_time - self.last_odom_time
        if odom_age > 0.05:  # 50ms threshold
            if self.solve_count % 100 == 0:  # Log every second
                self.get_logger().warn(
                    f'Stale odometry! Age: {odom_age*1000:.1f} ms',
                    throttle_duration_sec=1.0
                )
            return

        # Get latest state
        _, state_body = self.odom_buffer.get_latest()
        state_current = state_body.copy()

        # Transform velocity from body to world frame
        v_body = state_current[3:6]
        q = state_current[6:10]
        R_world_body = math_tool.quaternion_to_rotm(q)
        v_world = R_world_body @ v_body
        state_current[3:6] = v_world

        # Solve NMPC
        solve_start = time.time()
        status, T_rotor = self.nmpc_solver.solve(
            state=state_current,
            ref=self.ref_state,
            u_prev=self.des_rotor_thrust
        )
        solve_end = time.time()
        solve_time = (solve_end - solve_start)*1e3  # ms

        u_mpc = self.control_allocator.compute_u_from_rotor_thrusts(T_rotor)

        # Update statistics
        self.solve_count += 1
        self.total_solve_time += solve_time

        if status != 0:
            self.fail_count += 1
            if self.solve_count % 10 == 0:
                self.get_logger().warn(
                    f'Solver failed! Status: {status}',
                    throttle_duration_sec=1.0
                )
            return

        # Update control input
        self.des_rotor_thrust = T_rotor

        # Convert to RPM and publish
        des_rpm = np.sqrt(self.des_rotor_thrust / self.C_T)
        cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(
            self.get_clock().now(),
            des_rpm
        )

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

        # Log statistics periodically (every 100 iterations = 1 second at 100Hz)
        if self.solve_count % 100 == 0:
            avg_solve_time = self.total_solve_time / self.solve_count
            success_rate = 100.0 * (1.0 - self.fail_count / self.solve_count)

            self.get_logger().info(
                f'Stats: solve={avg_solve_time:.2f}ms, '
                f'success={success_rate:.1f}%, '
                f'odom_age={odom_age*1000:.1f}ms'
            )

    def _get_time_now(self) -> float:
        """Get current ROS time as float (seconds)"""
        clock_now = self.get_clock().now()
        sec, nsec = clock_now.seconds_nanoseconds()
        return sec + nsec * 1e-9

    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info('Shutting down NMPC node...')

        # Print final statistics
        if self.solve_count > 0:
            avg_solve_time = self.total_solve_time / self.solve_count
            success_rate = 100.0 * (1.0 - self.fail_count / self.solve_count)

            self.get_logger().info('='*60)
            self.get_logger().info('Final Statistics:')
            self.get_logger().info(f'  Total solves: {self.solve_count}')
            self.get_logger().info(f'  Average solve time: {avg_solve_time:.2f} ms')
            self.get_logger().info(f'  Success rate: {success_rate:.1f}%')
            self.get_logger().info('='*60)

        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None
    try:
        # Create node
        node = NmpcNodeV2()

        # Use SingleThreadedExecutor for predictable behavior
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Spin
        print('\n[NMPC] Node running. Press Ctrl+C to stop.\n')
        executor.spin()

    except KeyboardInterrupt:
        print('\n[NMPC] Keyboard interrupt received')
    except Exception as e:
        print(f'\n[NMPC] Exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cleanup_acados_files()
        print('[NMPC] Shutdown complete\n')


if __name__ == '__main__':
    main()
