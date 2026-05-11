import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped
from ros2_libcanard_msgs.msg import HexaCmdRaw

import numpy
from drone_control.utils.msg_parser import MsgParser

class MsgParserNode(Node):

    def __init__(self):
        super().__init__('msg_parser_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/mavros/odom',
            self.odom_cb,
            5
        )

        self.parser = MsgParser()

    def odom_cb(self, msg: Odometry):
        self.get_logger().info('Odometry received')
        odom = MsgParser.parse_odom_msg(msg)
        # time = math_tool.stamp_to_time(odom[0], odom[1])
        # self.get_logger().info('Odometry time stamp: ' + str(time))
        self.get_logger().info('Pose: {}'.format(odom))
        (now_sec, now_nanosec) = self.get_clock().now().seconds_nanoseconds()
        time_now = now_sec + now_nanosec*1e-9
        print(time_now)


def main():
    rclpy.init()
    node = MsgParserNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()