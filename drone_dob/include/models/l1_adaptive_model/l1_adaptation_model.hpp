#ifndef L1_ADAPTATION_MODEL_HPP
#define L1_ADAPTATION_MODEL_HPP

#include "state_predictor.hpp"
#include "adaptation_law.hpp"
#include "utils/LowPassFilter.hpp"

struct L1DobParam{
    Matrix6x6d As;                     // Hurwitz matrix for L1 DOB
    double freq_cutoff_trans{60};       // Cutoff frequency for translational uncertainty low-pass filter
    double freq_cutoff_rot{60};         // Cutoff frequency for rotational uncertainty low-pass filter
};


class L1AdaptationModel {
    public:

    L1AdaptationModel();

    L1AdaptationModel(const DroneParam& drone_param,
                    const L1DobParam& l1_dob_param);

    ~L1AdaptationModel();

    void configure(const DroneParam& drone_param,
                   const L1DobParam& l1_dob_param);

    void update(const double &t_prev,
                const double &t_curr,
                const StateData& state_meas,
                const Vector4d& u_BL);

    Vector6d get_u_L1() const;

    private:

    void initialize_state();

    Vector4d u_L1_;
    Vector6d sigma_hat_;

    StatePredictor state_predictor_;
    AdaptationLaw adaptation_law_;
    LowPassFilter *lpf_sigma_hat_[6];

};


#endif // L1_ADAPTATION_MODEL_HPP