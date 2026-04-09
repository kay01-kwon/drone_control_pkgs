from acados_template import AcadosOcp, AcadosOcpSolver
from drone_control.nmpc.model.S550_simple import S550_model
from drone_control.nmpc.cs_utils import cs_math_tool
from drone_control.utils.math_tool import quaternion_to_rotm
from scipy.linalg import block_diag
import casadi as cs
import numpy as np

class S550_Ocp:
    def __init__(self, DynParam = None, DroneParam = None, MpcParam = None):
        '''
        Constructor
        :param DynParam: m, MoiArray (Jxx, Jyy, Jzz)
        :param DroneParam: arm_length, rotor_const, moment_const, T_max, T_min
        :param MpcParam: t_horizon, n_nodes, QArray, RArray, tanh_k
        '''
        if DynParam is None:
            m = 3.0
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': m, 'MoiArray': J}

        if DroneParam is None:
            l = 0.265
            C_T = 1.465e-7
            k_m = 0.01569

            rotor_max = 7300.0
            rotor_min = 2000.0

            DroneParam = {'arm_length':l,
                          'motor_const':C_T,
                          'moment_const':k_m}
        else:
            C_T = DroneParam['motor_const']
            rotor_max = DroneParam['rotor_max']
            rotor_min = DroneParam['rotor_min']

        u_max = C_T * (rotor_max)**2
        u_min = C_T * (rotor_min)**2

        if MpcParam is None:
            t_horizon = 1.0                # Time horizon
            n_nodes = 20                    # Number of nodes
            Q = np.diag([2.0, 2.0, 10.0,           # px, py, pz
                        1.0, 1.0, 7.0,              # vx, vy, vz
                        8.0, 8.0, 8.0,              # q_e_x, q_e_y, q_e_z
                        5.0, 5.0, 5.0])             # wx, wy ,wz
            R = np.diag([5.0]*6)           # u1...u6
            roll_max_deg = 15.0
            pitch_max_deg = 15.0
            pitch_soft_Zl = 1000.0
            pitch_soft_Zu = 1000.0
            pitch_soft_zl = 100.0
            pitch_soft_zu = 100.0
            roll_soft_Zl = 1000.0
            roll_soft_Zu = 1000.0
            roll_soft_zl = 100.0
            roll_soft_zu = 100.0
            tanh_k = 100.0

        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R'][0]*np.eye(6)
            roll_max_deg = MpcParam.get('roll_max_deg', 15.0)
            pitch_max_deg = MpcParam.get('pitch_max_deg', 15.0)
            pitch_soft_Zl = MpcParam.get('pitch_soft_Zl', 1000.0)
            pitch_soft_Zu = MpcParam.get('pitch_soft_Zu', 1000.0)
            pitch_soft_zl = MpcParam.get('pitch_soft_zl', 100.0)
            pitch_soft_zu = MpcParam.get('pitch_soft_zu', 100.0)
            roll_soft_Zl = MpcParam.get('roll_soft_Zl', 1000.0)
            roll_soft_Zu = MpcParam.get('roll_soft_Zu', 1000.0)
            roll_soft_zl = MpcParam.get('roll_soft_zl', 100.0)
            roll_soft_zu = MpcParam.get('roll_soft_zu', 100.0)
            tanh_k = MpcParam.get('tanh_k', 100.0)

        self.tanh_k = tanh_k

        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        # --- Add parameters for NONLINEAR_LS cost: q_ref(4) + tanh_k(1) ---
        q_ref_sym = cs.MX.sym('q_ref', 4)
        tanh_k_sym = cs.MX.sym('tanh_k', 1)
        # p = [qw_ref, qx_ref, qy_ref, qz_ref, tanh_k]
        acados_model.p = cs.vertcat(q_ref_sym, tanh_k_sym)

        # --- Quaternion error cost expression ---
        p_pos = acados_model.x[0:3]    # position
        v = acados_model.x[3:6]        # velocity
        q = acados_model.x[6:10]       # quaternion
        w = acados_model.x[10:13]      # angular velocity

        # q_e = conj(q_ref) otimes q
        q_ref_conj = cs_math_tool.conjugate(q_ref_sym)
        q_e = cs_math_tool.otimes(q_ref_conj, q)

        # Sign approximation for double cover prevention
        sign_approx = cs.tanh(tanh_k_sym * q_e[0])
        q_e_vec_signed = sign_approx * q_e[1:4]

        # Cost residual: [p(3), v(3), q_e_vec_signed(3), w(3), u(6)] = 18-dim
        acados_model.cost_y_expr = cs.vertcat(p_pos, v, q_e_vec_signed, w, acados_model.u)
        acados_model.cost_y_expr_e = cs.vertcat(p_pos, v, q_e_vec_signed, w)

        # Put acados model into ocp model
        self.ocp.model = acados_model

        x0 = np.array([
            0.0, 0.0, 0.0,          # px, py, pz
            0.0, 0.0, 0.0,          # vx, vy, vz
            1.0, 0.0, 0.0, 0.0,     # qw, qx, qy, qz
            0.0, 0.0, 0.0           # wx, wy, wz
        ])

        # Dim info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        n_pos = 3
        n_vel = 3
        n_q_e = 3   # quaternion error vector dim
        n_w = 3
        ny = n_pos + n_vel + n_q_e + n_w + nu     # 18
        ny_e = n_pos + n_vel + n_q_e + n_w          # 12

        # 1. Cost setup

        # 1.1 Declare type of cost: NONLINEAR_LS for quaternion error cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # 1.2 Weight setup
        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        # 1.3 Reference setup
        # y_ref = [p_ref(3), v_ref(3), q_e_vec_ref(0,0,0), w_ref(3), u_ref(6)]
        self.ocp.cost.yref = np.zeros(ny)
        self.ocp.cost.yref_e = np.zeros(ny_e)

        # 2. Set ocp constraints
        self.ocp.constraints.x0 = x0
        self.ocp.constraints.lbu = np.array([u_min]*nu)
        self.ocp.constraints.ubu = np.array([u_max]*nu)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # 2.1 Nonlinear attitude constraints (roll/pitch)
        # h1 = 2*(qw*qy - qx*qz) = sin(pitch)
        # h2 = 2*(qw*qx + qy*qz) = sin(roll)*cos(pitch)
        x = self.ocp.model.x
        qw = x[6]
        qx = x[7]
        qy = x[8]
        qz = x[9]

        sin_pitch = 2 * (qw * qy - qx * qz)
        sin_roll_cos_pitch = 2 * (qw * qx + qy * qz)

        self.ocp.model.con_h_expr = cs.vertcat(sin_pitch, sin_roll_cos_pitch)

        sin_pitch_max = np.sin(np.deg2rad(pitch_max_deg))
        sin_roll_max = np.sin(np.deg2rad(roll_max_deg))

        self.ocp.constraints.lh = np.array([-sin_pitch_max, -sin_roll_max])
        self.ocp.constraints.uh = np.array([sin_pitch_max, sin_roll_max])

        # Terminal attitude constraint (same bounds)
        self.ocp.model.con_h_expr_e = cs.vertcat(sin_pitch, sin_roll_cos_pitch)
        self.ocp.constraints.lh_e = np.array([-sin_pitch_max, -sin_roll_max])
        self.ocp.constraints.uh_e = np.array([sin_pitch_max, sin_roll_max])

        # 2.2 Soft constraints on pitch (index 0) and roll (index 1) in con_h_expr
        # Slack variables allow temporary violation with L1 + L2 penalty
        self.ocp.constraints.idxsh = np.array([0, 1])
        self.ocp.cost.zl = np.array([pitch_soft_zl, roll_soft_zl])     # L1 lower penalty
        self.ocp.cost.zu = np.array([pitch_soft_zu, roll_soft_zu])     # L1 upper penalty
        self.ocp.cost.Zl = np.array([pitch_soft_Zl, roll_soft_Zl])     # L2 lower penalty
        self.ocp.cost.Zu = np.array([pitch_soft_Zu, roll_soft_Zu])     # L2 upper penalty

        # Terminal soft constraints on pitch and roll
        self.ocp.constraints.idxsh_e = np.array([0, 1])
        self.ocp.cost.zl_e = np.array([pitch_soft_zl, roll_soft_zl])
        self.ocp.cost.zu_e = np.array([pitch_soft_zu, roll_soft_zu])
        self.ocp.cost.Zl_e = np.array([pitch_soft_Zl, roll_soft_Zl])
        self.ocp.cost.Zu_e = np.array([pitch_soft_Zu, roll_soft_Zu])

        # 3. Set ocp solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4       # RK4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.print_level = 0                 # 0 : Do not print
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 100
        # self.ocp.solver_options.qp_solver_cond_N = n_nodes
        # self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'PROJECT'
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        # p0 = [qw_ref, qx_ref, qy_ref, qz_ref, tanh_k]
        p0 = np.array([1.0, 0.0, 0.0, 0.0, tanh_k])
        self.ocp.parameter_values = p0

        # self.ocp_solver = AcadosOcpSolver(self.ocp)
        # generate json file and generate cython
        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        AcadosOcpSolver.generate(self.ocp, json_file = self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)
        # Store state trajectory for warm start
        self.previous_states = None

    def solve(self, state, ref, u_prev=None):
        '''
        Solve OCP problem with warm start
        :param state: p (World), v (Body), q, w (Body)
        :param ref: p (World), v (World), q, w (Body)
        :param u_prev: u1...u6 (Rotor thrust)
        :return: status, u(u1...u6)
        '''

        # Copy state and ref to avoid mutating the caller's buffer
        state = state.copy()
        ref = ref.copy()

        # Transform velocity from body frame to world frame
        v_body = state[3:6]
        q = state[6:10]
        R_b_w = quaternion_to_rotm(q)
        v_world = R_b_w @ v_body
        state[3:6] = v_world

        # Extract references
        p_ref = ref[0:3]
        v_ref = ref[3:6]
        q_ref = ref[6:10]
        w_ref = ref[10:13]

        if u_prev is None:
            u_prev = np.zeros((6,))

        # Parameter: [q_ref(4), tanh_k]
        param = np.array([q_ref[0], q_ref[1], q_ref[2], q_ref[3],
                          self.tanh_k])

        # y_ref: [p_ref(3), v_ref(3), q_e_vec_ref(0,0,0), w_ref(3), u_prev(6)]
        y_ref = np.concatenate((p_ref, v_ref, [0.0, 0.0, 0.0], w_ref, u_prev))
        y_ref_N = np.concatenate((p_ref, v_ref, [0.0, 0.0, 0.0], w_ref))

        # Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state)
        self.ocp_solver.set(0, 'ubx', state)

        N = self.ocp.solver_options.N_horizon

        # Warm start: Use previous state trajectory as reference
        if self.previous_states is not None:
            # Shift previous trajectory forward by one step
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', param)

                if stage < N - 1:
                    # Use next state from previous trajectory
                    prev_state = self.previous_states[stage + 1]
                    p_prev = prev_state[0:3]
                    v_prev = prev_state[3:6]
                    w_prev = prev_state[10:13]
                    y_ref_warm = np.concatenate((p_prev, v_prev, [0.0, 0.0, 0.0], w_prev, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    # Last stage: use terminal reference
                    self.ocp_solver.set(stage, 'y_ref', y_ref)

            # Set terminal reference
            self.ocp_solver.set(N, 'p', param)
            self.ocp_solver.set(N, 'y_ref', y_ref_N)
        else:
            # First solve: use constant reference
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', param)
                self.ocp_solver.set(stage, 'y_ref', y_ref)

            # Set y ref at the terminal stage
            self.ocp_solver.set(N, 'p', param)
            self.ocp_solver.set(N, 'y_ref', y_ref_N)

        status = self.ocp_solver.solve()

        # Store current state trajectory for next iteration
        self.previous_states = []
        for stage in range(N + 1):
            x_stage = self.ocp_solver.get(stage, 'x')
            self.previous_states.append(x_stage.copy())

        # Get control input at the first stage
        u = self.ocp_solver.get(0, 'u')

        return status, u

    def get_json_file_name(self):
        return self.solver_json
