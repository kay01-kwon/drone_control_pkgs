#ifndef ODE_RK4_SOLVER_H
#define ODE_RK4_SOLVER_H

#include <iostream>

template <typename StateInOut>
class OdeRk4Solver{

    public:

    OdeRk4Solver();

    template <typename SystemDynamics>
    void do_step(SystemDynamics system_dynamics,
                 StateInOut &state,
                 const double &prev_time,
                 const double &dt);

    private:

};

#endif

// Constructor for OdeRk4Solver
template <typename StateInOut>
OdeRk4Solver<StateInOut>::OdeRk4Solver()
{
}

template <typename StateInOut>
template <typename SystemDynamics>
void OdeRk4Solver<StateInOut>::do_step(SystemDynamics system_dynamics,
                              StateInOut &s,
                              const double &prev_time,
                              const double &dt)
{
    StateInOut k1, k2, k3, k4;

    // Compute the four stages of RK4

    // 1. First stage
    system_dynamics(s, k1, prev_time);

    // 2. Second stage
    StateInOut s_temp = s + 0.5*dt*k1;
    system_dynamics(s_temp, k2, prev_time + 0.5*dt);
    

    // 3. Third stage
    s_temp = s + 0.5*dt*k2;
    system_dynamics(s_temp, k3, prev_time + 0.5*dt);

    // 4. Fourth stage
    s_temp = s + dt*k3;
    system_dynamics(s_temp, k4, prev_time + dt);

    // Combine the results of the four stages of RK4
    s += (k1*1.0 + k2*2.0 + k3*2.0 + k4*1.0)/6.0 * dt;
    
}