import numpy as np
from ..utils import math_tool

class RcControl():

    def __init__(self, GainParam, DynParam):

        KpTransArray = GainParam['KpTransArray']
        KiTransArray = GainParam['KiTransArray']
        KdTransArray = GainParam['KdTransArray']
        IntegralMaxArray = GainParam['IntegralMaxArray']

        KpOriArray = GainParam['KpOriArray']
        KdOriArray = GainParam['KdOriArray']
        AccelMaxArray = GainParam['AccelMaxArray']

        self.KpTransDiag = np.diag(KpTransArray)
        self.KiTransDiag = np.diag(KiTransArray)
        self.KdTransDiag = np.diag(KdTransArray)
        self.IntegralMaxArray = np.array(IntegralMaxArray)
        self.KpOriDiag = np.diag(KpOriArray)
        self.KdOriDiag = np.diag(KdOriArray)

        self.AccelMaxArray = AccelMaxArray

        self.m = DynParam['m']
        MoiArray = DynParam['MoiArray']
        self.J = np.diag(MoiArray)

        self.g_vec = np.array([0, 0, -9.81])

        self.psi_des = 0.0
        self.p_des = np.zeros((3,))
        self.p_err_integral = np.zeros((3,))
        self.initialized = False
        self.axis_des = np.zeros((3,))

        # Force and moment
        self.u = np.zeros((4,))

    def reset(self, state):
        '''
        Reset desired position and yaw from current state.
        Call this when entering MANUAL_STAB or after landing.
        '''
        self.p_des = state[0:3].copy()
        self.p_err_integral = np.zeros((3,))
        q = state[6:10]
        R = math_tool.quaternion_to_rotm(q)
        self.psi_des = np.arctan2(R[1, 0], R[0, 0])
        self.initialized = True

    def set_ref(self, ref, state, dt ,tau):
        '''

        :param ref: vx_des, vy_des, vz_des, dpsi_dt_des
        :param state: p, v, q, w
        :param tau: orientational disturbance
        :return: None
        '''
        vx_des, vy_des, vz_des = ref[0], ref[1], ref[2]
        dpsi_dt_des = ref[3]
        cmd_vel = np.array([vx_des, vy_des, vz_des])
        w_des = np.array([0, 0, dpsi_dt_des])

        p = state[0:3]
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]

        R = math_tool.quaternion_to_rotm(q)

        # Initialize p_des from current state on first call
        if not self.initialized:
            self.p_des = p.copy()
            self.p_err_integral = np.zeros((3,))
            self.psi_des = np.arctan2(R[1, 0], R[0, 0])
            self.initialized = True

        # Integrate velocity command to get desired position
        # cmd_vel is in body frame, convert to world frame for integration
        v_cmd_world = R @ cmd_vel
        self.p_des += v_cmd_world * dt
        self.psi_des += dpsi_dt_des * dt

        # Position error in world frame
        p_err = p - self.p_des

        # Accumulate integral of position error
        self.p_err_integral += p_err * dt

        # Anti-windup: clamp integral term
        for i in range(3):
            self.p_err_integral[i] = np.clip(
                self.p_err_integral[i],
                -self.IntegralMaxArray[i],
                self.IntegralMaxArray[i]
            )

        # Velocity error in world frame (for damping)
        v_err_body = v - cmd_vel
        v_err = R @ v_err_body

        # PID position control
        accelCommand = -(self.KpTransDiag @ p_err
                         + self.KiTransDiag @ self.p_err_integral
                         + self.KdTransDiag @ v_err) / self.m
        accelCommand = self._accel_command_clamp(accelCommand)

        accelCommand = accelCommand - self.g_vec

        f_des = self.m * accelCommand
        self.u[0] = f_des.dot(R[:,2])

        # b1, b2, b3 : desired frame
        b3 = accelCommand/np.linalg.norm(accelCommand)

        c1 = np.array([np.cos(self.psi_des), np.sin(self.psi_des), 0])

        if np.linalg.norm(np.cross(c1, b3)) < 1e-3:
            c1 = np.array([1, 0, 0])

        b2 = np.cross(b3, c1)
        b2 = b2 / np.linalg.norm(b2)
        b1 = np.cross(b2, b3)

        # Construct desired rotation matrix
        R_des = np.column_stack((b1, b2, b3))

        angle_error_matrix = 0.5*(R_des.transpose()@R - R.transpose()@R_des)
        e_R = math_tool.skew_symm_to_vec(angle_error_matrix)
        e_w = w - R.transpose() @ R_des @ w_des

        accel_ang = -(self.KpOriDiag @ e_R + self.KdOriDiag @ e_w)

        M_control = self.J @ accel_ang - tau

        self.u[1:] = M_control

    def get_control_input(self):
        return self.u

    def _accel_command_clamp(self, accelCommand):

        for i in range(3):
            if np.abs(accelCommand[i]) >= self.AccelMaxArray[i]:
                accelCommand[i] = (self.AccelMaxArray[i]
                                   * math_tool.signum(accelCommand[i]))
        return accelCommand
