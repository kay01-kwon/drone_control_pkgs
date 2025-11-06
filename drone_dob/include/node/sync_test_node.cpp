#include "sync_test_node.hpp"

SyncTestNode::SyncTestNode()
: Node("sync_test_node")
{
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));

    hexa_cmd_raw_publisher_ = this->create_publisher<ros2_libcanard_msgs::msg::HexaCmdRaw>(
        "/uav/cmd_raw", 1);
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/S550/ground_truth/odom", rclcpp::SensorDataQoS());
    test_timer_ = this->create_wall_timer(
        10ms, std::bind(&SyncTestNode::LoopCallback, this));
}

SyncTestNode::~SyncTestNode()
{
}

void SyncTestNode::LoopCallback() {

    auto hexa_cmd_raw_msg = ros2_libcanard_msgs::msg::HexaCmdRaw();
    hexa_cmd_raw_msg.header.stamp = this->now();
    hexa_cmd_raw_msg.cmd_raw = {1500, 1500, 1500, 1500, 1500, 1500};
    hexa_cmd_raw_publisher_->publish(hexa_cmd_raw_msg);

    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";
    odom_msg.pose.pose.position.x = 0.0;
    odom_msg.pose.pose.position.y = 0.0;
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.pose.pose.orientation.w = 1.0;
    odom_msg.pose.pose.orientation.x = 0.0;
    odom_msg.pose.pose.orientation.y = 0.0;
    odom_msg.pose.pose.orientation.z = 0.0;
    odom_msg.twist.twist.linear.x = 0.0;
    odom_msg.twist.twist.linear.y = 0.0;
    odom_msg.twist.twist.linear.z = 0.0;
    odom_msg.twist.twist.angular.x = 0.0;
    odom_msg.twist.twist.angular.y = 0.0;
    odom_msg.twist.twist.angular.z = 0.0;
    odom_publisher_->publish(odom_msg);
}