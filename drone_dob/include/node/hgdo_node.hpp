#ifndef HGDO_NODE_HPP
#define HGDO_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <chrono>

#include "nav_msgs/msg/odometry.hpp"
#include <ros2_libcanard_msgs/msg/hexa_actual_rpm.hpp>
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "utils/math_tool.hpp"
#include "utils/CircularBuffer.hpp"
#include "utils/HexaRotorRpmToCmd.hpp"
#include "utils/LowPassFilter.hpp"

#include "models/hgdo_model/hgdo_model.hpp"

using namespace std::chrono_literals;

using nav_msgs::msg::Odometry;
using ros2_libcanard_msgs::msg::HexaActualRpm;
using geometry_msgs::msg::WrenchStamped;

class HgdoNode : public rclcpp::Node {
    
    public:

    HgdoNode();
    
    ~HgdoNode();
    
    private:

    // Sensor Msg data callbacks
    void odomCallback(const Odometry::SharedPtr msg);
    void hexaActualRpmCallback(const HexaActualRpm::SharedPtr msg);

    // DOB estimate loop
    void dobEstimateLoopCallback();

    void odom_filter();

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
                          const HgdoParam &hgdo_param,
                          const double &lin_cutoff_freq,
                          const double &ang_cutoff_freq,
                          const double &disturbance_force_cutoff,
                          const double &disturbance_torque_cutoff);

    rclcpp::TimerBase::SharedPtr control_loop_timer_;
    
    rclcpp::Subscription<Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<HexaActualRpm>::SharedPtr hexa_actual_rpm_subscriber_;

    rclcpp::Publisher<Odometry>::SharedPtr filtered_odom_publisher_;
    rclcpp::Publisher<WrenchStamped>::SharedPtr dob_publisher_;
    
    double t_curr_{0.0};
    double t_prev_{0.0};
    double dob_looptime_{0.01};

    CircularBuffer<OdomData> odom_buffer_;
    CircularBuffer<RpmData> hexa_rpm_buffer_;

    Vector3d lin_vel_filtered_{Vector3d::Zero()};
    Vector3d ang_vel_filtered_{Vector3d::Zero()};

    HgdoModel *hgdo_model_{nullptr};
    HexaRotorRpmToCmd *rpm_to_cmd_converter_{nullptr};

    LowPassFilter *angular_velocity_lpf_[3]{nullptr};
    LowPassFilter *linear_velocity_lpf_[3]{nullptr};

    LowPassFilter *disturbance_force_lpf_[3]{nullptr};
    LowPassFilter *disturbance_torque_lpf_[3]{nullptr};

    Vector6d disturbance_estimate_{Vector6d::Zero()};
    
    nav_msgs::msg::Odometry filtered_odom_msg_;
    WrenchStamped dob_msg_;


};

#endif // HGDO_NODE_HPP