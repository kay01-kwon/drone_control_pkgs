#include "adaptation_law.hpp"

AdaptationLaw::AdaptationLaw()
{

}

AdaptationLaw::~AdaptationLaw()
{

}

void AdaptationLaw::update(const double &t_prev,
                           const double &t_curr,
                           const Vector6d& z_tilde,
                           const Quaterniond& q_meas)
{
    // Implementation of the update method goes here

    double dt = t_curr - t_prev;

    Matrix6x6d Phi, Phi_inv;

    double exp_As_dt;

    for(size_t i = 0; i < 6; ++i)
    {
        exp_As_dt = exp(dt*As_(i,i));
        Phi(i,i) = 1/As_(i,i)*(exp_As_dt - 1.0);
        Phi_inv(i,i) = 1/Phi(i,i);
        mu_(i) = exp_As_dt*z_tilde(i);
    }
    
    Matrix3x3d R_B_I;
    R_B_I = q_meas.toRotationMatrix();
    Vector3d e_x_B, e_y_B, e_z_B;
    e_x_B = R_B_I.col(0);
    e_y_B = R_B_I.col(1);
    e_z_B = R_B_I.col(2);
    
    Matrix6x6d G_inv;
    G_inv.setZero();

    G_inv.block<1,3>(0,0) = e_x_B.transpose() * m_;
    G_inv.block<3,3>(1,3) = J_;
    G_inv.block<1,3>(4,0) = e_y_B.transpose() * m_;
    G_inv.block<1,3>(5,0) = e_z_B.transpose() * m_;

    sigma_hat_ = -G_inv * Phi_inv * mu_;

}

Vector6d AdaptationLaw::get_sigma_hat() const
{
    return sigma_hat_;
}
