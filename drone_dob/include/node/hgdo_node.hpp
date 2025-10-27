#ifndef HGDO_NODE_HPP
#define HGDO_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <chrono>

#include "nav_msgs/msg/odometry.hpp"
#include "ros2_libcanard_msgs/msg/hexa_actual_rpm.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "utils/CircularBuffer.hpp"
#include "utils/HexaRotorRpmToCmd.hpp"
#include "utils/LowPassFilter.hpp"

#include "models/hgdo_model.hpp"

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
    void hexaRpmCallback(const HexaActualRpm::SharedPtr msg);

    // DOB estimate loop
    void dobEstimateLoopCallback();

    rclcpp::TimerBase::SharedPtr control_loop_timer_;
    
    rclcpp::Subscription<Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<HexaActualRpm>::SharedPtr hexa_rpm_subscriber_;

    rclcpp::Publisher<Odometry>::SharedPtr filtered_odom_publisher_;
    rclcpp::Publisher<WrenchStamped>::SharedPtr dob_publisher_;
    

    CircularBuffer<OdomData> odom_buffer_;
    CircularBuffer<RpmData> hexa_rpm_buffer_;

    HgdoModel *hgdo_model_{nullptr};
    HexaRotorRpmToCmd *rpm_to_cmd_converter_{nullptr};


};

#endif // HGDO_NODE_HPP