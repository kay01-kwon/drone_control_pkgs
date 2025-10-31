import numpy as np
from enum import Enum

class FlightMode(Enum):
    ARMED = 0
    DISARMED = 1
    AUTO = 2
    MANUAL_STAB = 3
    KILL = 4

class RcModeStr:
    @staticmethod
    def mode_str(mode:FlightMode):
        ret=""
        if mode == FlightMode.ARMED:
            ret = "ARMED"
        elif mode == FlightMode.DISARMED:
            ret = "DISARMED"
        elif mode == FlightMode.AUTO:
            ret = "AUTO"
        elif mode == FlightMode.MANUAL_STAB:
            ret = "MANUAL_STAB"
        elif mode == FlightMode.KILL:
            ret = "KILL"
        else:
            ret = "UNKNOWN"
        return ret


class RcConverter:
    def __init__(self, ConverterParam):
        # Set flight mode
        self.mode = FlightMode.AUTO

        # Set maximum acceleration and altitude
        self.vxy_max = ConverterParam['vxy_max']
        self.vz_max = ConverterParam['vz_max']

        self.R_max = np.sqrt(2*self.vxy_max**2)

        self.dpsi_dt_max = ConverterParam['dpsi_dt_max']

        self.rc_in_mid = 1500
        self.rc_in_delta = 512
        self.rc_in_min = self.rc_in_mid - self.rc_in_delta
        self.rc_in_max = self.rc_in_mid + self.rc_in_delta

        self.v_des = np.zeros((3,))
        self.dpsi_dt_des = np.zeros((1,))

    def set_rc(self, rc_in):
        vx_temp =  self.vxy_max * self._constrain(rc_in[1])
        vy_temp = -self.vxy_max * self._constrain(rc_in[0])

        R_temp = np.sqrt(vx_temp**2 + vy_temp**2)

        if R_temp > self.R_max and R_temp > 1e-9:
            scale = self.R_max / R_temp
            vx_des = scale * vx_temp
            vy_des = scale * vy_temp
        else:
            vx_des = vx_temp
            vy_des = vy_temp

        self.v_des[0] = vx_des
        self.v_des[1] = vy_des
        self.v_des[2] = (self.vz_max
                       * self._vz_constrain(rc_in[2]))
        self.dpsi_dt_des = (- self.dpsi_dt_max
                             * self._constrain(rc_in[3]))


        if self._two_pos(rc_in[7]) == "HIGH":
            self.mode = FlightMode.ARMED
            if self._three_pos(rc_in[8]) == "LOW":
                self.mode = FlightMode.KILL
            else:
                if self._three_pos(rc_in[5]) == "LOW":
                    self.mode = FlightMode.ARMED
                elif self._three_pos(rc_in[5]) == "MID":
                    self.mode = FlightMode.MANUAL_STAB
                elif self._three_pos(rc_in[5]) == "HIGH":
                    self.mode = FlightMode.AUTO
        elif self._two_pos(rc_in[7]) == "LOW":
            self.mode = FlightMode.DISARMED

    def get_rc_state(self):
        return self.mode, self.v_des, self.dpsi_dt_des

    def _constrain(self, input):
        temp = (float(input-self.rc_in_mid)/
        float(self.rc_in_delta))
        return temp

    def _vz_constrain(self, input):

        vz_input = (float(input - self.rc_in_mid)/
                    float(self.rc_in_delta))
        return vz_input

    def _two_pos(self, pwm:int) -> str:
        '''
        Return 'LOW' | 'HIGH'
        :param pwd:
        :return:
        '''
        if pwm < 1200:
            return 'LOW'
        elif pwm > 1700:
            return 'HIGH'

    def _three_pos(self, pwm:int) -> str:
        '''
        Return 'LOW' | 'MID' |'HIGH'
        :param pwm:
        :return:
        '''
        if pwm < 1200:
            return 'LOW'
        elif pwm > 1700:
            return 'HIGH'
        else:
            return 'MID'