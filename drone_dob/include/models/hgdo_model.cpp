#include "hgdo_model.hpp"
#include "utils/math_tool.hpp"

HgdoModel::HgdoModel()
{
    initialize_state();
}

HgdoModel::HgdoModel(const DroneParam& drone_param,
                     const HgdoParam& hgdo_param)
{
    configure_internal(drone_param, hgdo_param);
    initialize_state();

    ode_solver_ = new OdeRk4Solver<Vector6d>();
}

void HgdoModel::configure(const DroneParam& drone_param,
                            const HgdoParam& hgdo_param)
{
    configure_internal(drone_param, hgdo_param);
    initialize_state();
}

HgdoModel::~HgdoModel()
{
    if(ode_solver_ != nullptr)
        delete ode_solver_;
}

void HgdoModel::update(const double &t_prev,
                       const double &t_curr,
                       const Vector3d& v,
                       const Vector3d& w,
                       const Quaterniond& q,
                       const Vector4d& u)
{
    t_prev_ = t_prev;
    t_curr_ = t_curr;
    v_ = v;
    w_ = w;
    q_ = q;
    u_ = u;

    if(t_curr_ <= t_prev_)
        return;
    else
    {
        dt_ = t_curr_ - t_prev_;
        
        ode_solver_->do_step(
            [this](const Vector6d& gamma,
                   Vector6d& gamma_dot,
                   const double &t_prev)
            {
                compute_dynamics(gamma, gamma_dot, t_prev);
            },
            gamma_, t_prev_, dt_);

    }
}

Vector6d HgdoModel::get_disturbance_estimate() const
{
    Vector6d disturbance_estimate;
    
    disturbance_estimate.head<3>()
    = m_*(gamma_.head<3>() + 1.0/hgdo_param_.eps_f*v_);
    
    disturbance_estimate.tail<3>()
    = J_*(gamma_.tail<3>() + 1.0/hgdo_param_.eps_tau*w_);

    return disturbance_estimate;
}

void HgdoModel::compute_dynamics(const Vector6d& gamma,
                                 Vector6d& gamma_dot,
                                 const double &t_prev)
{
    Vector3d collective_thrust(0.0, 0.0, u_(0)/m_);

    Matrix3x3d R_wb = q_.toRotationMatrix();

    // Body to World frame
    collective_thrust = R_wb * collective_thrust;

    gamma_dot.head<3>()
    = - (1.0 / hgdo_param_.eps_f) * 
    (gamma.head<3>() + 1.0/hgdo_param_.eps_f*v_)
    + 1.0/hgdo_param_.eps_f*(-collective_thrust - g_vec);

    gamma_dot.tail<3>() = -1.0/hgdo_param_.eps_tau *
    (gamma.tail<3>() + 1.0/hgdo_param_.eps_tau*w_)
    + 1.0/hgdo_param_.eps_tau*
    (-J_inv_*u_.tail<3>() + J_inv_*w_.cross(J_*w_));

}

void HgdoModel::initialize_state()
{
    gamma_.setZero();
    w_.setZero();
    v_.setZero();
    q_.setIdentity();
    u_.setZero();
}

void HgdoModel::configure_internal(const DroneParam& drone_param,
                                   const HgdoParam& hgdo_param)
{
    // Configure model parameters
    J_ = drone_param.J;
    J_inv_ << 1/J_(0,0), 0, 0,
               0, 1/J_(1,1), 0,
               0, 0, 1/J_(2,2);

    m_ = drone_param.m;

    hgdo_param_ = hgdo_param;
}

