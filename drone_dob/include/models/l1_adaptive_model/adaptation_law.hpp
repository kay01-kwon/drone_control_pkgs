#ifndef ADAPTATION_LAW_HPP
#define ADAPTATION_LAW_HPP

#include "utils/state_def.hpp"

class AdaptationLaw {

    public:

    AdaptationLaw();

    AdaptationLaw(const DroneParam& drone_param,
                    const Matrix6x6d& As);

    ~AdaptationLaw();

    void configure(const DroneParam& drone_param,
                   const Matrix6x6d& As);

    /**
     * @brief Updates the adaptation law based on the previous and current time, the error state, and the measured quaternion.
     * 
     * @param t_prev : Previous time
     * @param t_curr : Current time
     * @param z_tilde : Error state (z_hat - z_meas)
     * @param q_meas : Measured quaternion
     */
    void update(const double &t_prev,
                const double &t_curr,
                const Vector6d& z_tilde,
                const Quaterniond& q_meas);

    /**
     * @brief Get the noisy uncertainty estimate sigma_hat
     * 
     * @return Vector6d (Matched and unmatched uncertainty estimate)
     */
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