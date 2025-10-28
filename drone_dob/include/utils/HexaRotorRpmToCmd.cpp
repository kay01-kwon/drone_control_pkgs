#include "utils/HexaRotorRpmToCmd.hpp"

HexaRotorRpmToCmd::HexaRotorRpmToCmd() 
{
    DroneParam default_param;
    default_param.J << 0.06, 0.0, 0.0,
                        0.0, 0.06, 0.0,
                        0.0, 0.0, 0.08;
    configure(default_param);
}

HexaRotorRpmToCmd::HexaRotorRpmToCmd(const DroneParam& drone_param) 
{
    configure(drone_param);
}

void HexaRotorRpmToCmd::configure(const DroneParam& drone_param)
{
    configure_mapping(drone_param);
}

Vector4d HexaRotorRpmToCmd::convert(const Vector6int16& rpm)
{
    Vector4d cmd;

    Vector6d rotor_thrust;

    for(size_t i = 0; i < 6; ++i){
        rotor_thrust[i] = C_T_ * double(rpm[i]*rpm[i]);
    }

    cmd = rpm_to_cmd_matrix_ * rotor_thrust;

    return cmd;
}

HexaRotorRpmToCmd::~HexaRotorRpmToCmd() 
{
}

void HexaRotorRpmToCmd::configure_mapping(const DroneParam& drone_param)
{

    // Get the motor and moment constants
    C_T_ = drone_param.C_T;
    k_m_ = drone_param.k_m;

    // Compute the location of each rotor
    double l = drone_param.l;

    double cos_pi_3 = cos(M_PI / 3.0);
    double sin_pi_3 = sin(M_PI / 3.0);

    double ly1, ly2, ly3, ly4, ly5, ly6;
    double lx1, lx2, lx3, lx4, lx5, lx6;

    ly1 = l * cos_pi_3;
    ly2 = l;
    ly3 = l * cos_pi_3;

    ly4 = -l * cos_pi_3;
    ly5 = -l;
    ly6 = -l * cos_pi_3;

    lx1 = l * sin_pi_3;
    lx2 = 0.0;
    lx3 = -l * sin_pi_3;
    
    lx4 = -l * sin_pi_3;
    lx5 = 0.0;
    lx6 = l * sin_pi_3;

    // Configure the mapping matrix from rotor RPM to command
    rpm_to_cmd_matrix_ <<
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ly1, ly2, ly3, ly4, ly5, ly6,
    -lx1, -lx2, -lx3, -lx4, -lx5, -lx6,
    -k_m_, k_m_, -k_m_, k_m_, -k_m_, k_m_;

}