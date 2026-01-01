#include "adaptation_law.hpp"

AdaptationLaw::AdaptationLaw()
{
    sigma_hat_.setZero();
    mu_.setZero();
}

AdaptationLaw::AdaptationLaw(const DroneParam& drone_param,
                               const Matrix6x6d& As)
{
    configure(drone_param, As);
    sigma_hat_.setZero();
    mu_.setZero();
}

AdaptationLaw::~AdaptationLaw()
{

}

void AdaptationLaw::configure(const DroneParam& drone_param,
                              const Matrix6x6d& As)
{
    m_ = drone_param.m;
    J_ = drone_param.J;
    As_ = As;
    As_inv_ = Matrix6x6d::Zero();
    for(size_t i = 0; i < 6; ++i)
    {
        As_inv_(i,i) = 1.0/As_(i,i);
    }
    std::cout << "As_inv_: \n" << As_inv_ << std::endl;
}

void AdaptationLaw::update(const double &t_prev,
                           const double &t_curr,
                           const Vector6d& z_tilde,
                           const Quaterniond& q_meas)
{
    // Implementation of the update method goes here

    double dt = t_curr - t_prev;

    if (dt <= 0.0)
    {
        return;
    }

    Matrix6x6d Phi_inv;

    Phi_inv.setZero();

    double exp_As_dt;
    double Phi_ii;

    for(size_t i = 0; i < 6; ++i)
    {
        exp_As_dt = exp(dt*As_(i,i));
        Phi_ii = As_inv_(i,i)*(exp_As_dt - 1.0);
        Phi_inv(i,i) = 1.0/Phi_ii;
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

    G_inv.block<1,3>(0,0) = e_z_B.transpose() * m_;
    G_inv.block<3,3>(1,3) = J_;
    G_inv.block<1,3>(4,0) = e_x_B.transpose() * m_;
    G_inv.block<1,3>(5,0) = e_y_B.transpose() * m_;

    sigma_hat_ = -G_inv * Phi_inv * mu_;

    // std::cout << "G_inv: \n" << G_inv << std::endl;
    // std::cout << "Phi_inv: \n" << Phi_inv << std::endl;
    // std::cout << "mu_: \n" << mu_ << std::endl;
    // std::cout << "sigma_hat_: \n" << sigma_hat_ << std::endl;

}

Vector6d AdaptationLaw::get_sigma_hat() const
{
    return sigma_hat_;
}
