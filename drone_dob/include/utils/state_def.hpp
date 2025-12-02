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

// Matrix 6 by 6 definition for Hurwitz matrix
typedef Eigen::Matrix<double, 6, 6> Matrix6x6d;

typedef Eigen::Matrix<int16_t, 6, 1> Vector6int16;

static Vector3d g_vec{0.0, 0.0, -9.81}; // Gravity vector

static Matrix3x3d Eye = Matrix3x3d::Identity(); // Identity matrix

#define BIT_TO_RPM (9800.0/8191.0) // Conversion factor from command bit to RPM

enum class DroneType{
    Quad,
    Hexa,
};

/**
 * @brief Parameters for the drone
 * m: Mass
 * l: Arm length
 * J: Inertia matrix
 * C_T: Motor constant (N / RPM^2)
 * k_m: Moment constant (Nm / N)
 */
struct DroneParam{
    double m{3.0};              // Mass
    double l{0.265};            // Arm length
    Matrix3x3d J;               // Inertia matrix
    double C_T{1.465e-07};      // Motor constant
    double k_m{0.01569};        // Moment constant
};


/**
 * @brief State data structure
 * p: Position vector (World frame)
 * v: Linear velocity vector (World frame)
 * q: Orientation quaternion (World to Body)
 * w: Angular velocity vector (Body frame)
 */
struct StateData{
    Vector3d p{Vector3d::Zero()};          // Position (World frame)
    Vector3d v{Vector3d::Zero()};          // Linear velocity (World frame)
    Quaterniond q{Quaterniond::Identity()}; // Orientation (World to Body)
    Vector3d w{Vector3d::Zero()};          // Angular velocity (Body
};

/**
 * @brief Odometry data structure
 * timestamp: Time stamp
 * position: Position vector (World frame)
 * linear_velocity: Linear velocity vector (World frame)
 * orientation: Orientation quaternion (World to Body)
 * angular_velocity: Angular velocity vector (Body frame)
 */
struct OdomData{
    double timestamp{0.0};
    Vector3d position{Vector3d::Zero()};
    Vector3d linear_velocity{Vector3d::Zero()};
    Quaterniond orientation{Quaterniond::Identity()};
    Vector3d angular_velocity{Vector3d::Zero()};
};

/**
 * @brief RPM data structure (Hexacopter)
 * timestamp: Time stamp
 * rpm: RPM vector for each motor
 */
struct RpmData{
    double timestamp{0.0};
    Vector6int16 rpm{Vector6int16::Zero()};
};

#endif // STATE_DEF_HPP