#ifndef STATE_PREDICTOR_HPP
#define STATE_PREDICTOR_HPP

#include "utils/state_def.hpp"
#include "utils/rk4_ode_solver.hpp"

// Matrix 6 by 6 definition for Hurwitz matrix
typedef Eigen::Matrix<double, 6, 6> Matrix6x6d;

class StatePredictor{

    public:

    // Constructor
    StatePredictor();

    // Destructor
    ~StatePredictor();

    void configure(const DroneParam& drone_param,
                   const Matrix6x6d& hurwitz_matrix);

    void update(const double &t_prev,
                const double &t_curr,
                const Vector6d &state_meas,
                const Vector4d& u_BL,
                const Vector4d& u_L1,
                const Vector6d &sigma);


    Vector6d get_predicted_state() const;

    private:

    void compute_dynamics(const Vector6d& z_hat,
                          Vector6d& z_hat_dot,
                          const double &t_prev);

    Vector6d state_meas_;
    Vector6d z_hat_;

    Vector6d sigma_;
    Vector4d u_BL_;
    Vector4d u_L1_;

    Matrix6x6d As_;

    OdeRk4Solver<Vector6d> rk4_solver_;

    double t_curr_{0.0};
    double t_prev_{0.0};

};


#endif // STATE_PREDICTOR_HPP