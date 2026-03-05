"""
ROS2 NMPC ATTITUDE Node with Disturbance Observer (DOB)

This implements NMPC with disturbance compensation from DOB:
- Get total thrust
- Receives rotational disturbances from DOB (HGDO, L1 adaptaiton, EKF/UKF)
- Compensates control input for estimated disturbance
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
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.control_allocator import ControlAllocator
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, math_tool, cleanup_acados_files
from drone_control.nmpc.ocp.S550_att_ocp import S550_att_ocp

class NMPCAttitudeWithDOB(Node):
    """
    NMPC Attitude Node with DOB integration for ROS2
    This node implements Model Predictive control with disturbance compensation.
    Control loop runs at 100 Hz using ROS2 timer callback.
    """
    def __init__(self):
        super().__init__('nmpc_att_with_dob',
                         automatically_declare_parameters_from_overrides=True)

        # Load parameters
        dynamc_param, drone_param, nmpc_param = self._load_parameters()

        # Store parameters as instance variables
        self.dynamic_param = dynamc_param
        self.drone_param = drone_param
        self.nmpc_param = nmpc_param

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

        # Reference state (q, w) in 7 dim
        self.ref_state = np.zeros((7,))
        self.ref_state[0] = 1.0  # qw = 1 (Identity quaternion)

        # Statistics for NMPC solver
        self.solve_count = 0
        self.failure_count = 0
        self.total_solve_time = 0.0

        # Flags
        self.solver_ready = False
        self.first_solve = True


    def _load_parameter(self):
        """
        Load paramters from ROS2 parameter server

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
        self.get_logger().info(f'  Arm length: {arm_length:.3f} m')
        self.get_logger().info(f'  Rotor const: {motor_const:.2e}')
        self.get_logger().info(f'  Rotor RPM limits: [{rotor_min:.2f}, {rotor_max:.2f}] N')
        self.get_logger().info(f'  Horizon: {t_horizon:.2f} s, Nodes: {n_nodes}')

        dynamic_param = {
            'm': m,
            'MoiArray': MoiArray
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
