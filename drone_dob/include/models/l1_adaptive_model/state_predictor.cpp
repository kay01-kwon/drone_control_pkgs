#include "state_predictor.hpp"

StatePredictor::StatePredictor(){
    // Constructor
}

StatePredictor::~StatePredictor(){
    // Destructor
}

void StatePredictor::configure(const DroneParam& drone_param,
                               const Matrix6x6d& hurwitz_matrix)
{
    As_ = hurwitz_matrix;
}

void StatePredictor::update(const double &t_prev,
                            const double &t_curr,
                            const Vector6d &state_meas,
                            const Vector4d& u_BL,
                            const Vector4d& u_L1,
                            const Vector6d &sigma)
{

}

Vector6d StatePredictor::get_predicted_state() const{
    return z_hat_;
}


void StatePredictor::compute_dynamics(const Vector6d& z_hat,
                                      Vector6d& z_hat_dot,
                                      const double &t_prev)
{
    // Dynamics computation logic goes here
}