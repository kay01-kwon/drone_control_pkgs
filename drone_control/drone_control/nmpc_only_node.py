import os
import shutil

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

        self.u_hover = dynamic_param['m'] * 9.81 / 6.0 * np.ones((6,))

        self.C_T = drone_param['rotor_const']

        self.nmpc_solver = S550_Ocp(DynParam=dynamic_param,
                                    DroneParam=drone_param,
                                    MpcParam=nmpc_param)

        self.odom_buf = CircularBuffer(capacity=30)
        self.ref = np.zeros((13,))
        self.ref[6] = 1.0
        self.des_rotor_thrust = self.u_hover.copy()  # Initialize with hover thrust
        self.cmd_msg = HexaCmdRaw()

        self.group_sub = MutuallyExclusiveCallbackGroup()
        self.group_pub = MutuallyExclusiveCallbackGroup()

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
        time_period = 0.01
        self.timer = self.create_timer(time_period,
                                       self._time_cb,
                                       callback_group=self.group_pub)

        self.t_curr = self._get_time_now()
        self.t_prev = self.t_curr

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

    def _time_cb(self):

        self.t_curr = self._get_time_now()

        if self.odom_buf.is_empty():
            return

        time_diff = self._get_time_now() - self.odom_buf.get_latest()[0]
        # self.get_logger().info(f'solver time: {time_diff * 1000:.2f} ms')

        state_recent = self.odom_buf.get_latest()[1]
        # Transform linear velocity
        # from body to world frame
        v_Body = state_recent[3:6]
        q = state_recent[6:10]
        R = math_tool.quaternion_to_rotm(q)
        v_World = R @ v_Body
        state_recent[3:6] = v_World
        # time_now = self._get_time_now()
        status, u = self.nmpc_solver.solve(state=state_recent,
                               ref=self.ref,
                               u_prev=self.des_rotor_thrust)
        # dt = self._get_time_now() - time_now
        # # Assuming dt is in seconds
        # self.get_logger().info(f'solver time: {dt * 1000:.2f} ms')

        self.des_rotor_thrust = u
        des_rpm = np.zeros((6,))

        if status == 0:
            for i in range(6):
                des_rpm[i] = np.sqrt(self.des_rotor_thrust[i]/self.C_T)

            self.cmd_msg = HexaCmdConverter.Rpm_to_cmd_raw(self.get_clock().now(), des_rpm)
            self.cmd_pub.publish(self.cmd_msg)
        else:
            self.get_logger().warn(f'Solver failed with status {status}!')
        self.t_prev = self.t_curr

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
        # rclpy.spin(node)
        executor.spin()
    except KeyboardInterrupt:
        cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()