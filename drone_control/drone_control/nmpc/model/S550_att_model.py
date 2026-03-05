from acados_template import AcadosModel
import casadi as cs
import numpy as np
from drone_control.nmpc.cs_utils import cs_math_tool

class S550_att_model:
    def __init__(self, DynParam, DroneParam):
        '''
        DynParam: m, MOI
        DroneParam: arm_length, motor_const, moment_const
        '''

        self.model_name = 'S550_att_model'
        self.model = AcadosModel()

        # Pass Dynamic parameter
        self.J = DynParam['MoiArray']

        # Pass Drone parameter
        self.l = DroneParam['arm_length']
        self.C_T = DroneParam['motor_const']
        self.k_m = DroneParam['moment_const']

        # State x (q, w)
        self.q = cs.MX.sym('q',4)
        self.w = cs.MX.sym('w',3)
        self.x = cs.vertcat(self.q, self.w)
        self.x_dim = 7

        # Control input (Rotor thrust vector)
        self.u1 = cs.MX.sym('u1')
        self.u2 = cs.MX.sym('u2')
        self.u3 = cs.MX.sym('u3')
        self.u4 = cs.MX.sym('u4')
        self.u5 = cs.MX.sym('u5')
        self.u6 = cs.MX.sym('u6')
        self.u = cs.vertcat(self.u1, self.u2, self.u3,
                            self.u4, self.u5, self.u6)
        self.u_dim = 6

        # Set collective thrust parameter variable
        self.f_col_param = cs.MX.sym('f_col')
        self.f_col_param_dim = 1

        # Time derivative of state
        self.dqdt = cs.MX.sym('dqdt', 4)
        self.dwdt = cs.MX.sym('dwdt', 3)
        self.xdot = cs.vertcat(self.dqdt, self.dwdt)

    def export_acados_model(self) -> AcadosModel:
        '''
        Export acados model
        :return: acados_model
        '''

        print('Exporting acados model...')

        f_expl = cs.vertcat(self._q_dynamics,
                            self._w_dynamics)

        f_impl = self.xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.p = self.f_col_param
        self.model.name = self.model_name

    def _q_dynamics(self):
        '''
        dqdt = 0.5 * otimes(q, w_quat)
        '''
        w_quat = cs.vertcat(0.0, self.w)

        dqdt = 0.5 * cs_math_tool.otimes(self.q, w_quat)
        return dqdt

    def _w_dynamics(self):
        '''
        dwdt = J_inv @ M - cross(w,J@w)
        :return:
        '''

        Jxx = self.J[0]
        Jyy = self.J[1]
        Jzz = self.J[2]

        Mx, My, Mz = self._control_alloc_moment()

        m_vec = cs.vertcat(Mx/Jxx, My/Jyy, Mz/Jzz)

        # Get angular velocity
        w_x = self.w[0]
        w_y = self.w[1]
        w_z = self.w[2]

        # inertial effect = J_inv @ (w x (J*w))
        inertial_effect = cs.vertcat((Jzz-Jyy)/Jxx*w_y*w_z,
                                     (Jxx-Jzz)/Jyy*w_x*w_z,
                                     (Jyy-Jxx)/Jzz*w_x*w_y)

        dwdt = m_vec - inertial_effect
        return dwdt

    def _control_alloc_moment(self):
        '''
        Thrust to moment
        :return:
        '''
        cos_pi_3 = np.cos(np.pi/3)
        sin_pi_3 = np.sin(np.pi/3)

        Mx = self.l * (
        (self.u1 + self.u3)*cos_pi_3 + self.u2
        -(self.u4 + self.u6)*cos_pi_3 - self.u5
        )

        My = self.l * (
            -(self.u1 + self.u6)*sin_pi_3
            +(self.u3 + self.u4)*sin_pi_3
        )

        Mz = self.k_m * (
            -self.u1 + self.u2 - self.u3
            +self.u4 - self.u5 + self.u6
        )

        return Mx, My, Mz

    def _col_thrust_constraint(self):
        '''

        '''
        f_col = (self.u1 + self.u2 + self.u3
                 + self.u4 + self.u5 + self.u6)
        h = f_col - self.f_col_param
        return h