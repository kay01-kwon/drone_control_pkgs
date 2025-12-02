#ifndef L1_DOB_NODE_HPP
#define L1_DOB_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <chrono>

#include "nav_msgs/msg/odometry.hpp"
#include <ros2_libcanard_msgs/msg/hexa_actual_rpm.hpp>
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "utils/CircularBuffer.hpp"
#include "utils/HexaRotorRpmToCmd.hpp"

#include "models/l1_adaptive_model/l1_adaptation_model.hpp"


class L1DobNode : public rclcpp::Node {
    
    public:

    L1DobNode();
    
    ~L1DobNode();
    
    private:

    // Sensor Msg data callbacks
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void hexaActualRpmCallback(const ros2_libcanard_msgs::msg::HexaActualRpm::SharedPtr msg);

    // DOB estimate loop
    void dobEstimateLoopCallback();

    void dob_estimate();

    Vector6int16 get_rpm_near_odom(const double &odom_time_stamp);

    Vector6int16 rpm_linear_interpolation(const double &odom_time_stamp,
                                        const Vector6int16 &rpm_before,
                                        const double &time_before,
                                        const Vector6int16 &rpm_after,
                                        const double &time_after);

    double get_rpm_time_stamp(CircularBuffer<RpmData>& buffer, size_t index);

    void configure_parameters();

    void print_parameters(const DroneParam &drone_param,
                          const L1DobParam &l1_dob_param,
                          const double &lin_cutoff_freq,
                          const double &ang_cutoff_freq,
                          const double &disturbance_force_cutoff,
                          const double &adaptation_gain);
};

#endif // L1_DOB_NODE_HPP