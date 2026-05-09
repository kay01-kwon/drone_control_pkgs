#ifndef HGDO_MODEL_HPP
#define HGDO_MODEL_HPP

#include "utils/state_def.hpp"
#include "utils/rk4_ode_solver.hpp"

struct HgdoParam{
    double eps_tau{0.01};
    double eps_f{0.01};
};

class HgdoModel {

    public: 

    // Default constructor
    HgdoModel();

    // Parameterized constructor
    HgdoModel(const DroneParam& drone_param,
              const HgdoParam& hgdo_param);
    
    void configure(const DroneParam& drone_param,
                   const HgdoParam& hgdo_param);
    
    ~HgdoModel();
    
    void update(const double &t_prev,
                const double &t_curr,
                const Vector3d& v,
                const Vector3d& w,
                const Quaterniond& q,
                const Vector4d& u);
    
    Vector6d get_disturbance_estimate() const;

    private:

    void compute_dynamics(const Vector6d& gamma,
                          Vector6d& gamma_dot,
                          const double &t_prev);

    void initialize_state();

    void configure_internal(const DroneParam& drone_param,
                            const HgdoParam& hgdo_param);

    OdeRk4Solver<Vector6d> *ode_solver_;

    double m_{3.0};
    Matrix3x3d J_, J_inv_;

    Vector6d gamma_;

    // Linear velocities (Odom frame)
    Vector3d v_;

    // Angular velocities (Body frame)
    Vector3d w_;
    // Orientation (Body frame w.r.t Odom frame)
    Quaterniond q_;

    // Control inputs (forces and torques in Body frame)
    Vector4d u_;

    HgdoParam hgdo_param_;

    double t_prev_{0.0};
    double t_curr_{0.0};
    double dt_{0.01};

};

#endif // HGDO_MODEL_HPP