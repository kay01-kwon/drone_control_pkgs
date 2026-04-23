from acados_template import AcadosOcp, AcadosOcpSolver
from drone_control.nmpc.model.S550_att_model import S550_att_model
from drone_control.nmpc.cs_utils import cs_math_tool
from scipy.linalg import block_diag
import casadi as cs
import numpy as np

class S550_att_ocp:
    def __init__(self, DynParam = None, DroneParam = None, MpcParam = None):
        '''
        :param DynParam: MoiArray
        :param DroneParam: arm_length, motor_const, moment_const, rotor_max, rotor_min
        :param MpcParam: t_horizon, n_nodes, QArray, RArray, tanh_k
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
            Q = np.diag(
                [8.0, 8.0, 8.0,             # q_e_x, q_e_y, q_e_z
                 5.0, 5.0, 5.0]             # wx, wy, wz
            )
            R = 10.0*np.eye(6)
            tanh_k = 100.0
        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R'][0]*np.eye(6)
            tanh_k = MpcParam.get('tanh_k', 100.0)

        self.tanh_k = tanh_k

        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_att_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        # --- Extend parameters with q_ref and tanh_k for NONLINEAR_LS cost ---
        q_ref_sym = cs.MX.sym('q_ref', 4)
        tanh_k_sym = cs.MX.sym('tanh_k', 1)
        # p = [f_col, qw_ref, qx_ref, qy_ref, qz_ref, tanh_k]
        acados_model.p = cs.vertcat(acados_model.p, q_ref_sym, tanh_k_sym)

        # --- Quaternion error cost expression ---
        q = acados_model.x[0:4]
        w = acados_model.x[4:7]

        # q_e = conj(q_ref) otimes q
        q_ref_conj = cs_math_tool.conjugate(q_ref_sym)
        q_e = cs_math_tool.otimes(q_ref_conj, q)

        # Sign approximation for double cover prevention
        sign_approx = cs.tanh(tanh_k_sym * q_e[0])
        q_e_vec_signed = sign_approx * q_e[1:4]

        # Cost residual: [q_e_vec_signed(3), w(3), u(6)] = 12-dim
        acados_model.cost_y_expr = cs.vertcat(q_e_vec_signed, w, acados_model.u)
        acados_model.cost_y_expr_e = cs.vertcat(q_e_vec_signed, w)

        # Put acados model into ocp model
        self.ocp.model = acados_model

        # Dimension info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        n_q_e = 3   # quaternion error vector dim
        n_w = 3     # angular velocity dim
        ny = n_q_e + n_w + nu      # 12
        ny_e = n_q_e + n_w          # 6

        n_param = acados_model.p.rows()

        # Initial reference and param
        x0 = np.array([
            1.0, 0.0, 0.0, 0.0,     # qw, qx, qy, qz
            0.0, 0.0, 0.0           # wx, wy ,wz
        ])
        # p0 = [f_col, qw_ref, qx_ref, qy_ref, qz_ref, tanh_k]
        p0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, tanh_k])

        # 1. Cost

        # 1.1 Cost type: NONLINEAR_LS for quaternion error cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # 1.2 Weight setup
        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        # 1.3 Reference setup
        # y_ref = [q_e_vec_ref(0,0,0), w_ref, u_ref]
        self.ocp.cost.yref = np.zeros(ny)
        self.ocp.cost.yref_e = np.zeros(ny_e)

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
        self.ocp.parameter_values = p0

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

        # Copy ref to avoid mutating the caller's buffer
        ref = ref.copy()

        if u_prev is None:
            u_prev = np.zeros((6,))

        # Extract q_ref and w_ref from ref
        q_ref = ref[0:4]
        w_ref = ref[4:7]

        # Parameter: [f_col, q_ref(4), tanh_k]
        param = np.array([f_col,
                          q_ref[0], q_ref[1], q_ref[2], q_ref[3],
                          self.tanh_k])

        N = self.ocp.solver_options.N_horizon

        # y_ref: [q_e_vec_ref(0,0,0), w_ref, u_prev]
        y_ref = np.concatenate(([0.0, 0.0, 0.0], w_ref, u_prev))
        y_ref_N = np.concatenate(([0.0, 0.0, 0.0], w_ref))

        #Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state)
        self.ocp_solver.set(0, 'ubx', state)

        if self.previous_states is not None:
            # Warm start: use previous trajectory for per-stage q_ref and w_ref
            for stage in range(N):

                # Per-stage q_ref from shifted previous trajectory
                prev_q = self.previous_states[min(stage + 1, N)][0:4]
                param_stage = np.array([f_col,
                                        prev_q[0], prev_q[1], prev_q[2], prev_q[3],
                                        self.tanh_k])
                self.ocp_solver.set(stage, 'p', param_stage)

                if stage < N-1:
                    prev_w = self.previous_states[stage + 1][4:7]
                    y_ref_warm = np.concatenate(([0.0, 0.0, 0.0], prev_w, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    self.ocp_solver.set(stage, 'y_ref', y_ref)

            # Final stage: actual target q_ref
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
