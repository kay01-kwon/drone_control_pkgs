from acados_template import AcadosOcp, AcadosOcpSolver
from drone_control.nmpc.model.S550_att_model import S550_att_model
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
            # Mass not in use
            m = 3.23
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': 3.23, 'MoiArray': J}

        if DroneParam is None:
            l = 0.265
            C_T = 1.365e-7
            k_m = 0.01569
            rotor_max = 7300.0
            rotor_min = 2000.0
            DroneParam = {'arm_length': l,
                          'motor_const': C_T,
                          'moment_const': k_m}

        else:
            C_T = DroneParam['motor_const']
            rotor_max = DroneParam['rotor_max']
            rotor_min = DroneParam['rotor_min']

        # Convert to Maximum and minimum rotor thrust
        u_max = C_T*(rotor_max)**2
        u_min = C_T*(rotor_min)**2

        if MpcParam is None:
            t_horizon = 1.0
            n_nodes = 20
            Q = np.array(
                [8.0, 8.0, 8.0, 8.0,    # qw, qx, qy, qz
                 5.0, 5.0, 5.0]             # wx, wy, wz
            )
        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R'][0]*np.eye(6)

        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_att_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        # Put acados model into ocp model
        self.ocp.model = acados_model

        # Dimension info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        ny = nx + nu

        n_param = acados_model.p.rows()

        # Initial reference and param
        x0 = np.array([
            1.0, 0.0, 0.0, 0.0,     # qw, qx, qy, qz
            0.0, 0.0, 0.0           # wx, wy ,wz
        ])
        p0 = np.array([0.0])

        # 1. Cost

        # 1.1 Cost type
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # 1.2 Vx setup
        self.ocp.cost.Vx = np.zeros((ny,nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        # 1.3 Vu setup
        self.ocp.cost.Vu = np.zeros((ny,nu))
        self.ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        # 1.4 Weight setup
        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        # 1.5 Reference setup
        self.ocp.cost.yref = np.concatenate((x0, np.zeros(nu)))
        self.ocp.cost.yref_e = x0

        # 2. Set ocp constraints

        # 2.1 Initial state
        self.ocp.constraints.x0 = x0

        # 2.2 Control input bound
        self.ocp.constraints.lbu = np.array([u_min]*nu)
        self.ocp.constraints.ubu = np.array([u_max]*nu)
        self.ocp.constraints.idxbu = np.arange(nu)

        # 2.3 Equality constraint (Collective thrust condition)
        self.ocp.constraints.lh = np.array([0.0])
        self.ocp.constraints.uh = np.array([0.0])

        # 3. Set solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.print_level = 0     # Do not print
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nmp_solver_max_iter = 100
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'PROJECT'
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        AcadosOcpSolver.generate(self.ocp, json_file = self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)
        # Store state trajectory for warm start
        self.previous_states = None

    def solve(self, state, ref, u_prev = None, f_col = None):
        '''
        Solve OCP problem with warm start
        :param state: q, w (Body)
        :param ref: q, w (Body)
        :param u_prev: u1...u6 (Rotor thrust)
        :return: status, u(u1...u6)
        '''

        if u_prev is None:
            u_prev = np.zeros((6,))

        param = np.array([f_col])
        N = self.ocp.solver_options.N

        y_ref = np.concatenate((ref, u_prev))
        y_ref_N = ref

        #Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state)
        self.ocp_solver.set(0, 'ubx', state)

        if self.previous_states is not None:
            # Set previous state computed by NMPC as reference
            for stage in range(N):

                self.ocp_solver.set(stage, 'p', param)

                if stage < N-1:
                    prev_state = self.previous_states[stage + 1]
                    y_ref_warm = np.concatenate((y_ref, prev_state))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    self.ocp_solver.set(stage, 'y_ref', y_ref)

            self.ocp_solver.set(N, 'p', param)
            self.ocp_solver.set(N, 'y_ref', y_ref_N)

        else:
            # Just set the final reference as reference
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', param)
                self.ocp_solver.set(stage, 'y_ref', y_ref)

            self.ocp_solver.set(N, 'p', param)
            self.ocp_solver.set(N, 'y_ref', y_ref_N)

        status = self.ocp_solver.solve()

        self.previous_states = [
            self.ocp_solver.get(stage, 'x').copy() for stage in range(N + 1)
        ]

        u = self.ocp_solver.get(0, 'u')

        return status, u

    def get_json_file_name(self):
        return self.solver_json




