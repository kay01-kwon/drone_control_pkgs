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

}

Vector6d AdaptationLaw::get_sigma_hat() const
{
    return sigma_hat_;
}
