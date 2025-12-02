#ifndef ADAPTATION_LAW_HPP
#define ADAPTATION_LAW_HPP

#include "utils/state_def.hpp"

class AdaptationLaw {

    public:

    AdaptationLaw();

    ~AdaptationLaw();

    private:

    double m_{3.0};

    Matrix3x3d J_;

    

    Quaterniond q_meas_;
    Vector6d mu_;

};

#endif // ADAPTATION_LAW_HPP