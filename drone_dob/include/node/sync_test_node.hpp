#ifndef SYNC_TEST_NODE_HPP
#define SYNC_TEST_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <chrono>

#include "ros2_libcanard_msgs/msg/hexa_cmd_raw.hpp"
#include "nav_msgs/msg/odometry.hpp"

using namespace std::chrono_literals;

class SyncTestNode : public rclcpp::Node {
    
    public:

    SyncTestNode();
    
    ~SyncTestNode();
    
    private:

    rclcpp::Publisher<ros2_libcanard_msgs::msg::HexaCmdRaw>::SharedPtr hexa_cmd_raw_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::TimerBase::SharedPtr test_timer_;

    void LoopCallback();

    double t_prev_{0.0};
    double t_curr_{0.0};

};

#endif // SYNC_TEST_NODE_HPP