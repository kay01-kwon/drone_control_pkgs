#ifndef STATE_DEF_HPP
#define STATE_DEF_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, 13, 1> StateVector13d;

typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Quaternion<double> Quaterniond;

typedef Eigen::Matrix<double, 3, 3> Matrix3x3d;

typedef Eigen::Matrix<int16_t, 6, 1> Vector6int16;

static Vector3d g_vec{0.0, 0.0, -9.81}; // Gravity vector

static Matrix3x3d Eye = Matrix3x3d::Identity(); // Identity matrix


enum class DroneType{
    Quad,
    Hexa,
};

struct DroneParam{
    double m{3.0};              // Mass
    double l{0.265};            // Arm length
    Matrix3x3d J;               // Inertia matrix
    double C_T{1.465e-07};      // Motor constant
    double k_m{0.01569};        // Moment constant
};

struct OdomData{
    double timestamp{0.0};
    Vector3d position{Vector3d::Zero()};
    Vector3d linear_velocity{Vector3d::Zero()};
    Quaterniond orientation{Quaterniond::Identity()};
    Vector3d angular_velocity{Vector3d::Zero()};
};

struct RpmData{
    double timestamp{0.0};
    Vector6int16 rpm{Vector6int16::Zero()};
};

#endif // STATE_DEF_HPP