#ifndef HEXA_ROTOR_RPM_TO_CMD_HPP
#define HEXA_ROTOR_RPM_TO_CMD_HPP

#include "utils/state_def.hpp"

typedef Eigen::Matrix<double, 4, 6> Matrix4x6d;



class HexaRotorRpmToCmd{
    public:

    HexaRotorRpmToCmd();
    HexaRotorRpmToCmd(const DroneParam& drone_param);

    void configure(const DroneParam& drone_param);
    
    // Convert motor RPMs to control commands (thrust and torques)
    Vector4d convert(const Vector6int16& rpm);

    ~HexaRotorRpmToCmd();

    private:

    void configure_mapping(const DroneParam& drone_param);


    DroneType drone_type_;

    double C_T_{1.465e-07};
    double k_m_{0.01569};

    Matrix4x6d rpm_to_cmd_matrix_;

};


#endif // HEXA_ROTOR_RPM_TO_CMD_HPP