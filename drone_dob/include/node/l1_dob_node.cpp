#include "l1_dob_node.hpp"

L1DobNode::L1DobNode()
: Node("l1_dob_node")
{
    configure_parameters();

    odom_buffer_.reserve(20);
    hexa_rpm_buffer_.reserve(20);

    // Subscribers
    odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/S550/ground_truth/odom", rclcpp::SensorDataQoS(),
        std::bind(&L1DobNode::odomCallback, this, std::placeholders::_1));

    hexa_rpm_subscriber_ = this->create_subscription<ros2_libcanard_msgs::msg::HexaActualRpm>(
        "/uav/actual_rpm", rclcpp::SensorDataQoS(),
        std::bind(&L1DobNode::hexaActualRpmCallback, this, std::placeholders::_1));

    // Publishers
    filtered_odom_publisher_ = this->create_publisher<Odometry>("/filtered_odom", rclcpp::SensorDataQoS());
    dob_publisher_ = this->create_publisher<WrenchStamped>("/l1_dob/wrench", rclcpp::SensorDataQoS());

    // Control loop timer
    control_loop_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(dob_looptime_),
        std::bind(&L1DobNode::dobEstimateLoopCallback, this));
}

L1DobNode::~L1DobNode(){
    delete l1_dob_model_;
    delete rpm_to_cmd_converter_;

    for(int i = 0; i < 3; ++i){
        delete angular_velocity_lpf_[i];
        delete linear_velocity_lpf_[i];
    }
}
