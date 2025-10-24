import numpy as np
from ..utils import math_tool

class RcControl():

    def __init__(self, GainParam, DynParam):

        KpTransArray = GainParam['KpTransArray']

        KpOriArray = GainParam['KpOriArray']
        KdOriArray = GainParam['KdOriArray']
        AccelMaxArray = GainParam['AccelMaxArray']

        self.KpTransDiag = np.diag(KpTransArray)

        self.KpOriDiag = np.diag(KpOriArray)
        self.KdOriDiag = np.diag(KdOriArray)
        self.AccelMaxArray = AccelMaxArray

        self.m = DynParam['m']
        MoiArray = DynParam['MoiArray']
        self.J = np.diag(MoiArray)

        self.g_vec = np.array([0, 0, -9.81])

        self.axis_des = np.zeros((3,))

        # Force and moment
        self.u = np.zeros((4,))


    def set_ref(self, ref, state, tau):
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

        v = state[3:6]
        q = state[6:10]
        w = state[10:13]

        R = math_tool.quaternion_to_rotm(q)
        v_err_body = v - cmd_vel
        v_err = R @ v_err_body

        # P gain for velocity control
        accelCommand = -self.KpTransDiag @ v_err / self.m
        accelCommand = self._accel_command_clamp(accelCommand)

        accelCommand = accelCommand - self.g_vec
        accelCommand_norm = np.linalg.norm(accelCommand)

        f_des = self.m * accelCommand
        self.u[0] = np.sqrt(f_des.dot(f_des))

        # b1, b2, b3 : desired frame
        b3 = accelCommand/accelCommand_norm

        b1 = R[:,0]

        tol = 1e-3
        if np.linalg.norm(np.cross(b1, b3)) < tol:
            b1 = R[:,1]

            if(np.linalg.norm(np.cross(b1, b3)) < tol):
                b1 = R[:,2]

        b2 = np.cross(b3,b1)

        # Construct desired rotation matrix
        R_des = np.column_stack((b1, b2, b3))

        angle_error_matrix = 0.5*(R_des.transpose()@R
                                  -R.transpose()@R_des)
        e_R = math_tool.skew_symm_to_vec(angle_error_matrix)
        e_w = w - R.transpose() @ R_des @ w_des


        M_pd = -self.J @ (self.KpOriDiag @ e_R
                          + self.KdOriDiag @ e_w)

        self.u[1:] = M_pd - tau

    def get_control_input(self):
        return self.u

    def _accel_command_clamp(self, accelCommand):

        for i in range(3):
            if np.abs(accelCommand[i]) < self.AccelMaxArray[i]:
                accelCommand[i] = (self.AccelMaxArray[i]
                                   * math_tool.signum(accelCommand[i]))
        return accelCommand