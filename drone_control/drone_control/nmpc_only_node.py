import os
import shutil
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np

from mavros_msgs.msg import RCIn
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from drone_msgs.msg import Ref
from ros2_libcanard_msgs.msg import HexaCmdRaw

from drone_control.utils.circular_buffer import CircularBuffer
from drone_control.utils.cmd_converter import HexaCmdConverter
from drone_control.utils import MsgParser, math_tool
from drone_control.nmpc.ocp.S550_simple_ocp import S550_Ocp

class NmpcOnlyNode(Node):
    def __init__(self):
        super().__init__('nmpc_only_node',
                         automatically_declare_parameters_from_overrides=True)
        dynamic_param, drone_param, nmpc_param = self._config()

        u_hover = dynamic_param['m'] * 9.81 / 6.0 * np.ones((6,))

        self.C_T = drone_param['rotor_const']

        self.nmpc_solver = S550_Ocp(DynParam=dynamic_param,
                                    DroneParam=drone_param,
                                    MpcParam=nmpc_param)

        self.odom_buf = CircularBuffer(capacity=30)
        self.ref = np.zeros((13,))
        self.ref[6] = 1.0
        self.des_rotor_thrust = u_hover  # ROS1 matches: initialized as zeros
        self.cmd_msg = HexaCmdRaw()

        self.group_sub = MutuallyExclusiveCallbackGroup()

        self.odom_sub = self.create_subscription(Odometry,
                                                 '/filtered_odom',
                                                 self._odom_cb,
                                                 qos_profile=qos_profile_sensor_data,
                                                 callback_group=self.group_sub)

        self.ref_sub = self.create_subscription(Ref,
                                                '/nmpc/ref',
                                                self._ref_cb,
                                                1)

        self.cmd_pub = self.create_publisher(HexaCmdRaw,
                                             '/uav/cmd_raw',
                                            5)

        # Control loop thread (ROS1-style)
        self.control_rate = 100.0  # Hz
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        self.t_curr = time.time()
        self.t_prev = self.t_curr

        self.get_logger().info('NMPC control thread started at 100 Hz')

    def _odom_cb(self, msg):

        odom_time, odom_data = MsgParser.parse_odom_msg(msg)

        if self.odom_buf.is_full():
            self.odom_buf.pop()

        self.odom_buf.push((odom_time, odom_data))


    def _ref_cb(self, msg):

        p = msg.p
        v = msg.v
        psi = msg.psi
        psi_dot = msg.psi_dot

        self.ref[:3] = p
        self.ref[3:6] = v

        self.ref[6] = np.cos(psi/2.0)
        self.ref[7] = 0.0
        self.ref[8] = 0.0
        self.ref[9] = np.sin(psi/2.0)

        self.ref[10] = 0.0
        self.ref[11] = 0.0
        self.ref[12] = psi_dot

    def _control_loop(self):
        """
        ROS1-style control loop running in separate thread.
        This ensures regular, blocking execution like ROS1 while loop.
        """
        period = 1.0 / self.control_rate  # 0.01s for 100Hz

        while self.running and rclpy.ok():
            self.t_curr = time.time()
            loop_start = time.time()

            # Skip if no odometry data yet
            if self.odom_buf.is_empty():
                time.sleep(period)
                continue

            # Check odom data freshness
            time_diff = self._get_time_now() - self.odom_buf.get_latest()[0]
            if time_diff > 0.05:  # Warn if odom is older than 50ms
                self.get_logger().warn(f'Odom data age: {time_diff * 1000:.2f} ms (stale!)')

            # Get latest state and transform velocity to world frame
            state_recent = self.odom_buf.get_latest()[1].copy()
            v_Body = state_recent[3:6]
            q = state_recent[6:10]
            R = math_tool.quaternion_to_rotm(q)
            v_World = R @ v_Body
            state_recent[3:6] = v_World

            # Solve NMPC
            time_now = time.time()
            status, u = self.nmpc_solver.solve(state=state_recent,
                                   ref=self.ref,
                                   u_prev=self.des_rotor_thrust)  # ROS1 matches: u_prev not used
            dt = time.time() - time_now
            self.des_rotor_thrust = u

            # Log solver time (less frequently to avoid spam)
            # if int(loop_start * 10) % 10 == 0:  # Every 1 second
            #     self.get_logger().info(f'solver time: {dt * 1000:.2f} ms, status: {status}')

            dt = self.t_curr - self.t_prev
            # self.get_logger().info(f'NMPC solver status: {dt*1000:.2f} ms')

            # Publish control command
            self.des_rotor_thrust = u
            des_rpm = np.zeros((6,))

            if status == 0:
                for i in range(6):
                    des_rpm[i] = np.sqrt(self.des_rotor_thrust[i]/self.C_T)

                self.cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(), des_rpm)
                self.cmd_pub.publish(self.cmd_msg)
            else:
                self.get_logger().warn(f'Solver failed with status {status}!')

            # Sleep to maintain control rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, period - elapsed)
            self.t_prev = self.t_curr
            time.sleep(sleep_time)

    def _get_time_now(self):
        clock_now = self.get_clock().now()
        (sec, nsec) = clock_now.seconds_nanoseconds()
        time_now = sec + nsec * 1e-9
        return time_now


    def _config(self):
        '''
        Get several parameters from yaml file
        :return: dynamic_param, drone_param, nmpc_param
        '''

        # 1. Get drone dynamics parameter
        m = self.get_parameter('dynamic_param.m').value
        MoiArray = self.get_parameter('dynamic_param.MoiArray').value

        # 2. Get drone paramter
        arm_length = self.get_parameter('drone_param.arm_length').value
        rotor_const = self.get_parameter('drone_param.rotor_const').value
        moment_const = self.get_parameter('drone_param.moment_const').value
        T_max = self.get_parameter('drone_param.T_max').value
        T_min = self.get_parameter('drone_param.T_min').value

        # 3. Get nmpc param
        t_horizon = self.get_parameter('nmpc_param.t_horizon').value
        n_nodes = self.get_parameter('nmpc_param.n_nodes').value
        QArray = self.get_parameter('nmpc_param.QArray').value
        R = self.get_parameter('nmpc_param.R').value

        self.get_logger().info(f'{self.get_name()}: Initializing...')
        self.get_logger().info(f'{self.get_name()}: m: {m}')
        self.get_logger().info(f'{self.get_name()}: MoiArray: {MoiArray}')
        self.get_logger().info(f'{self.get_name()}: arm_length: {arm_length}')
        self.get_logger().info(f'{self.get_name()}: rotor_const: {rotor_const}')
        self.get_logger().info(f'{self.get_name()}: moment_const: {moment_const}')
        self.get_logger().info(f'{self.get_name()}: T_max: {T_max}')
        self.get_logger().info(f'{self.get_name()}: T_min: {T_min}')
        self.get_logger().info(f'{self.get_name()}: t_horizon: {t_horizon}')
        self.get_logger().info(f'{self.get_name()}: n_nodes: {n_nodes}')
        self.get_logger().info(f'{self.get_name()}: QArray: {QArray}')
        self.get_logger().info(f'{self.get_name()}: R: {R}')

        dynamic_param = {'m': m,
                         'MoiArray': MoiArray}
        drone_param = {'arm_length': arm_length,
                       'rotor_const': rotor_const,
                       'moment_const': moment_const,
                       'T_max': T_max,
                       'T_min': T_min}

        nmpc_param = {'t_horizon': t_horizon,
                      'n_nodes': n_nodes,
                      'QArray': QArray,
                      'R': R}

        return dynamic_param, drone_param, nmpc_param

def cleanup():
    curr_dir = os.getcwd()
    target_list = ['c_generated_code', 'acados_ocp.json']
    for target in target_list:
        path = os.path.join(curr_dir, target)
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    print(f'\n[Cleanup] Removed acados files in {curr_dir}')

def main():
    rclpy.init()
    node = NmpcOnlyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print('\n[Shutdown] Stopping control thread...')
        node.running = False
        if node.control_thread.is_alive():
            node.control_thread.join(timeout=2.0)
        cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()