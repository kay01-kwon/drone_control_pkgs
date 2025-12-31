#ifndef STATE_PREDICTOR_HPP
#define STATE_PREDICTOR_HPP

#include "utils/state_def.hpp"
#include "utils/rk4_ode_solver.hpp"

// Matrix 6 by 4 definition for control input mapping
typedef Eigen::Matrix<double, 6, 4> Matrix6x4d;

// Matrix 6 by 2 definition for control input mapping
typedef Eigen::Matrix<double, 6, 2> Matrix6x2d;

class StatePredictor{

    public:

    // Constructor
    StatePredictor();

    StatePredictor(const DroneParam& drone_param,
                   const Matrix6x6d& As);

    // Destructor
    ~StatePredictor();

    void configure(const DroneParam& drone_param,
                   const Matrix6x6d& hurwitz_matrix);

    /**
     * @brief Updates the state predictor with the latest measurements and inputs
     *
     * @param t_prev : previous time
     * @param t_curr : current time
     * @param state_meas : measured state vector
     * @param u_BL : baseline control input
     * @param u_L1 : L1 adaptive control input
     * @param sigma : Matched (Fz, Mx, My, Mz) and unmatched (Fx, Fy) uncertainties
     */
    void update(const double &t_prev,
                const double &t_curr,
                const StateData &state_meas,
                const Vector4d& u_BL,
                const Vector4d& u_L1,
                const Vector6d &sigma);


    Vector6d get_predicted_state() const;

    Vector6d get_z_tilde() const;

    private:

    void initialize_state();

    /**
     * @brief Computes the dynamics of the predicted state
     * 
     * @param z_hat : Estimated state vector
     * @param z_hat_dot : Time derivative of the estimated state vector
     * @param t_prev : Previous time
     */
    void compute_dynamics(const Vector6d& z_hat,
                          Vector6d& z_hat_dot,
                          const double &t_prev);

    OdeRk4Solver<Vector6d> *ode_solver_;

    double m_{3.0};             // Mass
    Matrix3x3d J_, J_inv_;              // Inertia matrix


    StateData state_meas_;
    Vector6d z_hat_;

    Vector6d sigma_;
    Vector4d u_BL_;
    Vector4d u_L1_;

    Matrix6x6d As_;

    OdeRk4Solver<Vector6d> *rk4_solver_;

    double t_curr_{0.0};
    double t_prev_{0.0};

};


#endif // STATE_PREDICTOR_HPP