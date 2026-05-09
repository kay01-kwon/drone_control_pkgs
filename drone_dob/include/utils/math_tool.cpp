#include "math_tool.hpp"

Quaterniond otimes(const Quaterniond& q1, 
                   const Quaterniond& q2) 
{
    Quaterniond result;
    result.w() = q1.w() * q2.w() - q1.vec().dot(q2.vec());
    result.vec() = q1.w() * q2.vec() + q2.w() * q1.vec() + 
                   q1.vec().cross(q2.vec());
    return result;
}

Matrix3x3d skewSymmetric(const Eigen::Vector3d& v) {
    Matrix3x3d skew;
    double vx, vy, vz;
    vx = v(0);
    vy = v(1);
    vz = v(2);
    skew << 0, -vz, vy,
            vz, 0, -vx,
            -vy, vx, 0;
    return skew;
}

Matrix3x3d w_cross_Jw_deriv(const Matrix3x3d& J, 
                             const Eigen::Vector3d& w) {
    

    Matrix3x3d result;

    double wx, wy, wz;
    wx = w(0);
    wy = w(1);
    wz = w(2);

    double Jxx, Jyy, Jzz;
    Jxx = J(0, 0);
    Jyy = J(1, 1);
    Jzz = J(2, 2);

    double J_z_y = Jzz - Jyy;
    double J_x_z = Jxx - Jzz;
    double J_y_x = Jyy - Jxx;

    result << 0, J_z_y * wz, J_y_x * wy,
            J_x_z * wz, 0, J_z_y * wx,
            J_y_x * wy, J_x_z * wx, 0;

    return result;
}

Vector3d clampVector3d(const Vector3d& v, 
                           double min_val, 
                           double max_val) {
    Vector3d clamped;
    clamped(0) = std::max(min_val, std::min(max_val, v(0)));
    clamped(1) = std::max(min_val, std::min(max_val, v(1)));
    clamped(2) = std::max(min_val, std::min(max_val, v(2)));
    return clamped;
}