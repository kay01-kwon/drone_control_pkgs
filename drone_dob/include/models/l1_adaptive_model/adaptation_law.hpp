#ifndef ADAPTATION_LAW_HPP
#define ADAPTATION_LAW_HPP

#include "utils/state_def.hpp"

class AdaptationLaw {

    public:

    AdaptationLaw();

    ~AdaptationLaw();

    void update(const double &t_prev,
                const double &t_curr,
                const Vector6d& z_tilde,
                const Quaterniond& q_meas);

    Vector6d get_sigma_hat() const;
    private:

    double m_{3.0};

    Matrix3x3d J_;
    Matrix6x6d As_, As_inv_;

    Quaterniond q_meas_;
    Vector6d mu_;

    Vector6d sigma_hat_;

};

#endif // ADAPTATION_LAW_HPP