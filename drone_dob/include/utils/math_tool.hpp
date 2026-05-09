#ifndef MATH_TOOL_HPP
#define MATH_TOOL_HPP

#include "state_def.hpp"

Quaterniond otimes(const Quaterniond& q1, 
                   const Quaterniond& q2);

Matrix3x3d skewSymmetric(const Eigen::Vector3d& v);

Matrix3x3d w_cross_Jw_deriv(const Matrix3x3d& J, 
                            const Eigen::Vector3d& w);

Vector3d clampVector3d(const Vector3d& v, 
                           double min_val, 
                           double max_val);

#endif // MATH_TOOL_HPP