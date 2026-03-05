from acados_template import AcadosOcp, AcadosOcpSolver
from drone_control.nmpc.model.S550_simple import S550_model
from drone_control.utils.math_tool import quaternion_to_rotm
from scipy.linalg import block_diag
import numpy as np

class S550_att_ocp:
    def __init__(self, DynParam = None, DroneParam = None, MpcParam = None):
        '''
        :param DynParam: MoiArray
        :param DroneParam: arm_length, motor_const, moment_const, rotor_max, rotor_min
        :param MpcParam: t_horizon, n_nodes, QArray, RArray
        '''

        if DynParam is None:
            m = 3.23
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': 3.23, 'MoiArray': J}

        if DroneParam is None:
            l = 0.265
            C_T = 1.365e-7
            k_m = 0.01569
            DroneParam = {'arm_length': l,
                          'motor_const': C_T,
                          'moment_const': k_m}
        else:
            C_T = DroneParam['motor_const']
            u_max = C_T*(DroneParam['rotor_max'])**2
            u_min = C_T*(DroneParam['rotor_min'])**2
