import numpy as np
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from geometry_msgs.msg import WrenchStamped

def stamp_to_time(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class MsgParser:
    """ Static msg parser"""

    @staticmethod
    def parse_rc_msg(msg:RCIn):
        """ Parse a RC message"""
        return (stamp_to_time(msg.header.stamp),
                np.array(msg.channels, dtype=np.int32))

    @staticmethod
    def parse_odom_msg(msg:Odometry):
        """ Parse a Odometry message"""
        pose = msg.pose.pose
        twist = msg.twist.twist

        return (stamp_to_time(msg.header.stamp),
                np.array([pose.position.x,
                          pose.position.y,
                          pose.position.z,
                          twist.linear.x,
                          twist.linear.y,
                          twist.linear.z,
                          pose.orientation.w,
                          pose.orientation.x,
                          pose.orientation.y,
                          pose.orientation.z,
                          twist.angular.x,
                          twist.angular.y,
                          twist.angular.z],
                          dtype=np.float64),
                )

    @staticmethod
    def parse_wrench_msg(msg:WrenchStamped):
        """ Parse a WrenchStamped message"""
        return (stamp_to_time(msg.header.stamp),
                np.array([msg.wrench.force.x,
                          msg.wrench.force.y,
                          msg.wrench.force.z,
                          msg.wrench.torque.x,
                          msg.wrench.torque.y,
                          msg.wrench.torque.z],
                          dtype=np.float64)
                )