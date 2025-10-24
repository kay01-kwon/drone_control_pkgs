import numpy as np
from ..utils import math_tool

class RcControl():

    def __init__(self, GainParam, DynParam):

        KpTransArray = GainParam['KpTransArray']

        KpOriArray = GainParam['KpOriArray']
        KdOriArray = GainParam['KdOriArray']

        self.KpTransDiag = np.diag(KpTransArray)

        self.KpOriDiag = np.diag(KpOriArray)
        self.KdOriDiag = np.diag(KdOriArray)

        self.m = DynParam['m']
        MoiArray = DynParam['MoiArray']
        self.J = np.diag(MoiArray)

        self.g_vec = np.array([0, 0, -9.81])

        self.axis_des = np.zeros((3,))

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
        v_des = np.array([vx_des, vy_des, vz_des])
        w_des = np.array([0, 0, dpsi_dt_des])

        v = state[3:6]
        q = state[6:10]
        w = state[10:13]

        R = math_tool.quaternion_to_rotm(q)
        v_err_body = v - v_des
        v_err = R @ v_err_body

        accelCommand = -(self.KpTransDiag @ v_err / self.m
                        + self.g_vec)

        accelCommand_norm = np.linalg.norm(accelCommand)

        f_des = self.m * accelCommand

        self.u[0] = f_des.dot(R[:,2])

        r = accelCommand/accelCommand_norm
        rx = r[0]
        ry = r[1]
        rz = r[2]

        cos_half_phi = np.sqrt((1+rz)/2.0)
        sin_half_phi = np.sqrt((1+rz)/2.0)
        sin_phi = np.sqrt(1-rz**2)

        if np.abs(sin_phi) > 1e-30:
            self.axis_des[0] = -1/sin_phi * ry
            self.axis_des[1] = 1/sin_phi * rx
        else:
            self.axis_des[0] = 0
            self.axis_des[1] = 0

        q_rp_des = np.array([cos_half_phi,
                             sin_half_phi*self.axis_des[0],
                             sin_half_phi*self.axis_des[1],
                             sin_half_phi*self.axis_des[2]])

        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        half_psi = 0.5*np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
        cos_half_psi = np.cos(half_psi)
        sin_half_psi = np.sin(half_psi)
        q_yaw = np.array([cos_half_psi, 0, 0, sin_half_psi])

        q_des = math_tool.otimes(q_rp_des, q_yaw)
        q_des_conj = math_tool.conjugate(q_des)
        q_err = math_tool.otimes(q_des_conj, q)

        error_R = math_tool.quaternion_to_angle_axis_vec(q_err)

        R_des = math_tool.quaternion_to_rotm(q_des)

        w_err = w - R.transpose() @ R_des @ w_des

        M_pd =  self.J @ (-self.KpOriDiag @ error_R
                -self.KdOriDiag @ w_err)

        self.u[1:] = M_pd - tau

    def get_control_input(self):
        return self.u

    def _signum(self, x):
        return 1 if x >= 0 else -1