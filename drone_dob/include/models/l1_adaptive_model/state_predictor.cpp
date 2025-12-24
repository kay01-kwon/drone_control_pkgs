#include "state_predictor.hpp"

StatePredictor::StatePredictor(){
    // Constructor
    initialize_state();

    // Default Hurwitz matrix
    As_.setIdentity();
    As_ = -5.0 * As_;

    ode_solver_ = new OdeRk4Solver<Vector6d>();
}

StatePredictor::StatePredictor(const DroneParam& drone_param,
                               const Matrix6x6d& As)
{
    initialize_state();
    ode_solver_ = new OdeRk4Solver<Vector6d>();
    configure(drone_param, As);
}

StatePredictor::~StatePredictor(){
    // Destructor
    delete ode_solver_;
}

void StatePredictor::configure(const DroneParam& drone_param,
                               const Matrix6x6d& hurwitz_matrix)
{
    m_ = drone_param.m;
    J_ = drone_param.J;
    J_inv_.setZero();

    for(int i=0; i<3; i++)
    {
        J_inv_(i,i) = 1.0 / J_(i,i);
    }
    
    As_ = hurwitz_matrix;
}

void StatePredictor::update(const double &t_prev,
                            const double &t_curr,
                            const StateData &state_meas,
                            const Vector4d& u_BL,
                            const Vector4d& u_L1,
                            const Vector6d &sigma)
{
    t_prev_ = t_prev;
    t_curr_ = t_curr;
    state_meas_ = state_meas;
    u_BL_ = u_BL;
    u_L1_ = u_L1;
    sigma_ = sigma;

    if(t_curr_ <= t_prev_)
        return;
    else
    {
        double dt = t_curr_ - t_prev_;
        
        ode_solver_->do_step(
            [this](const Vector6d& z_hat,
                   Vector6d& z_hat_dot,
                   const double &t_prev)
            {
                compute_dynamics(z_hat, z_hat_dot, t_prev);
            },
            z_hat_, t_prev_, dt);

    }
}

Vector6d StatePredictor::get_predicted_state() const{
    return z_hat_;
}

Vector6d StatePredictor::get_z_tilde() const{
    Vector6d z_tilde;
    z_tilde.head<3>() = z_hat_.head<3>() - state_meas_.v;
    z_tilde.tail<3>() = z_hat_.tail<3>() - state_meas_.w;
    return z_tilde;
}

void StatePredictor::initialize_state(){

    state_meas_.p.setZero();
    state_meas_.v.setZero();
    state_meas_.q.setIdentity();
    state_meas_.w.setZero();

    z_hat_.setZero();

    sigma_.setZero();
    u_BL_.setZero();
    u_L1_.setZero();
}

void StatePredictor::compute_dynamics(const Vector6d& z_hat,
                                      Vector6d& z_hat_dot,
                                      const double &t_prev)
{
    // Unpack measured state
    Quaterniond q_meas;
    Vector3d v_meas, w_meas;
    q_meas = state_meas_.q;
    v_meas = state_meas_.v;
    w_meas = state_meas_.w;

    // Compute state prediction error
    Vector6d z_tilde;
    z_tilde.head<3>() = z_hat.head<3>() - v_meas;
    z_tilde.tail<3>() = z_hat.tail<3>() - w_meas;

    // Rotation matrix from Body to Inertial frame
    Matrix3x3d R_B_I = q_meas.toRotationMatrix();

    // Extract body frame unit vectors
    Vector3d e_x_B = R_B_I.col(0);
    Vector3d e_y_B = R_B_I.col(1);
    Vector3d e_z_B = R_B_I.col(2);

    // Compute function f (Base line control input effects)
    Vector6d func_f;
    func_f.head<3>() = g_vec + u_BL_(0)/m_*e_z_B;
    func_f.tail<3>() = J_inv_ * (u_BL_.tail<3>());

    // Compute function g (L1 adaptive control input effects)
    Matrix6x4d func_g;
    Matrix6x2d func_g_perp;

    func_g.setZero();
    func_g_perp.setZero();

    func_g.block<3,1>(0,0) = (1.0/m_)*e_z_B;
    func_g.block<3,3>(3,1) = J_inv_;

    func_g_perp.block<3,1>(0,0) = (1.0/m_)*e_x_B;
    func_g_perp.block<3,1>(0,1) = (1.0/m_)*e_y_B;

    // Compute z_hat_dot
    z_hat_dot = func_f
                + func_g * (u_L1_ + sigma_.head<4>())
                + func_g_perp * (sigma_.tail<2>())
                + As_ * z_tilde;

}