import numpy as np
from ros2_libcanard_msgs.msg import HexaCmdRaw
from ros2_libcanard_msgs.msg import QuadCmdRaw

MaxBit: int = 8191
MaxRpm: int = 9800

class HexaCmdConverter:
    @staticmethod
    def Rpm_to_cmd_raw(time_stamp, des_rpm):
        cmd_msg = HexaCmdRaw()
        (sec, nsec) = time_stamp.seconds_nanoseconds()
        cmd_msg.header.stamp.sec = sec
        cmd_msg.header.stamp.nanosec = nsec
        for i in range(6):
            cmd_msg.cmd_raw[i] = int(des_rpm[i]*MaxBit/MaxRpm)

        return cmd_msg

class QuadCmdConverter:
    @staticmethod
    def Rpm_to_cmd_raw(time_stamp, des_rpm):
        cmd_msg = QuadCmdRaw()
        cmd_msg.header.stamp = time_stamp

        for i in range(4):
            cmd_msg.cmd_raw[i] = int(des_rpm[i]*MaxBit/MaxRpm)
        return cmd_msg