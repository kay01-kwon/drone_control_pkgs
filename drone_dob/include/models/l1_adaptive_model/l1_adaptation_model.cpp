#include "l1_adaptation_model.hpp"

L1AdaptationModel::L1AdaptationModel()
{
    initialize_state();
    for(size_t i = 0; i < 6; ++i)
    {
        lpf_sigma_hat_[i] = new LowPassFilter();
    }
}

L1AdaptationModel::L1AdaptationModel(const DroneParam& drone_param,
                                     const L1DobParam& l1_dob_param)
{
    initialize_state();
    for(size_t i = 0; i < 6; ++i)
    {
        lpf_sigma_hat_[i] = new LowPassFilter();
    }
    configure(drone_param, l1_dob_param);
}

L1AdaptationModel::~L1AdaptationModel()
{
    for(size_t i = 0; i < 6; ++i)
    {
        delete lpf_sigma_hat_[i];
    }
}
void L1AdaptationModel::configure(const DroneParam& drone_param,
                                  const L1DobParam& l1_dob_param)
{
    state_predictor_.configure(drone_param, l1_dob_param.As);
    adaptation_law_.configure(drone_param, l1_dob_param.As);

    for(size_t i = 0; i < 3; ++i)
    {
        lpf_sigma_hat_[i]->setCutoffFrequency(l1_dob_param.freq_cutoff_trans);
    }

    for(size_t i = 3; i < 6; ++i)
    {
        lpf_sigma_hat_[i]->setCutoffFrequency(l1_dob_param.freq_cutoff_rot);
    }
}

void L1AdaptationModel::update(const double &t_prev,
                               const double &t_curr,
                               const StateData& state_meas,
                               const Vector4d& u_BL)
{
    state_predictor_.update(t_prev, t_curr, 
        state_meas, u_BL, u_L1_, sigma_hat_);
    
    Vector6d z_tilde = state_predictor_.get_z_tilde();

    adaptation_law_.update(t_prev, t_curr, z_tilde, state_meas.q);
    sigma_hat_ = adaptation_law_.get_sigma_hat();

    double dt = t_curr - t_prev;
    for(size_t i = 0; i < 6; ++i)
    {
        lpf_sigma_hat_[i]->update(sigma_hat_(i), dt);

        if (i < 4)
        {
            u_L1_(i) = -lpf_sigma_hat_[i]->getOutput();
        }
    }
    
}

Vector6d L1AdaptationModel::get_u_L1() const
{
    return u_L1_;
}

void L1AdaptationModel::initialize_state()
{
    u_L1_.setZero();
    sigma_hat_.setZero();
}