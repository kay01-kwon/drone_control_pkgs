#include "l1_adaptive_node.hpp"

L1AdaptiveNode::L1AdaptiveNode()
: Node("l1_dob_node")
{
    configure_parameters();

    odom_buffer_.reserve(20);
    hexa_rpm_buffer_.reserve(20);

    // 1. Subscribers
    // 1.1. Odom subscriber
    this->declare_parameter("topic_names.odom_in", "/S550/ground_truth/odom");
    std::string odom_topic_name = this->get_parameter("topic_names.odom_in").as_string();
    odom_subscriber_ = this->create_subscription<Odometry>(
        odom_topic_name, rclcpp::SensorDataQoS(),
        std::bind(&L1AdaptiveNode::odomCallback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to odom topic: %s", odom_topic_name.c_str());
    
    // 1.2. Hexa Actual RPM subscriber
    this->declare_parameter("topic_names.actual_rpm", "/uav/actual_rpm");
    std::string hexa_actual_rpm_topic_name = this->get_parameter("topic_names.actual_rpm").as_string();
    hexa_rpm_subscriber_ = this->create_subscription<HexaActualRpm>(
        hexa_actual_rpm_topic_name, rclcpp::SensorDataQoS(),
        std::bind(&L1AdaptiveNode::hexaActualRpmCallback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to hexa actual rpm topic: %s", hexa_actual_rpm_topic_name.c_str());

    // 2. Publishers
    // 2.1. Filtered odom publisher
    this->declare_parameter("topic_names.filtered_odom", "/l1_adaptive/filtered_odom");
    std::string filtered_odom_topic_name = this->get_parameter("topic_names.filtered_odom").as_string();
    filtered_odom_publisher_ = this->create_publisher<Odometry>(filtered_odom_topic_name, rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Publishing filtered odom to topic: %s", filtered_odom_topic_name.c_str());

    // 2.2. DOB wrench publisher
    this->declare_parameter("topic_names.dob_wrench", "/l1_adaptive/wrench");
    std::string dob_wrench_topic_name = this->get_parameter("topic_names.dob_wrench").as_string();
    dob_publisher_ = this->create_publisher<WrenchStamped>(dob_wrench_topic_name, rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Publishing DOB wrench to topic: %s", dob_wrench_topic_name.c_str());
    
    // Control loop timer
    control_loop_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(dob_looptime_),
        std::bind(&L1AdaptiveNode::dobEstimateLoopCallback, this));
}

L1AdaptiveNode::~L1AdaptiveNode(){
    delete l1_adaptive_model_;
    delete rpm_to_cmd_converter_;

    for(int i = 0; i < 3; ++i){
        delete angular_velocity_lpf_[i];
        delete linear_velocity_lpf_[i];
    }
}

void L1AdaptiveNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    OdomData odom_data;
    odom_data.timestamp = msg->header.stamp.sec + 
                        msg->header.stamp.nanosec * 1e-9;

    // Fill odom_data (4 parts)

    // 1. Position
    odom_data.position <<
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z;

    // 2. Linear velocities
    odom_data.linear_velocity << 
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z;

    // 3. Orientation (quaternion)
    odom_data.orientation.w()
    = msg->pose.pose.orientation.w;

    odom_data.orientation.x()
    = msg->pose.pose.orientation.x;

    odom_data.orientation.y()
    = msg->pose.pose.orientation.y;

    odom_data.orientation.z()
    = msg->pose.pose.orientation.z;

    // 4. Angular velocities
    odom_data.angular_velocity <<
        msg->twist.twist.angular.x,
        msg->twist.twist.angular.y,
        msg->twist.twist.angular.z;

    if(odom_buffer_.is_full())
    {
        odom_buffer_.pop();
    }

    odom_buffer_.push_back(odom_data);

    if(odom_buffer_.size() >= 2)
        odom_filter();

    if(!odom_buffer_.is_empty() && !hexa_rpm_buffer_.is_empty())
    {
        if(odom_buffer_.size() >= 2 && hexa_rpm_buffer_.size() >= 2)
        {
            dob_estimate();
        }
    }
}

void L1AdaptiveNode::hexaActualRpmCallback(const ros2_libcanard_msgs::msg::HexaActualRpm::SharedPtr msg)
{
    RpmData rpm_data;
    rpm_data.timestamp = msg->header.stamp.sec + 
                        msg->header.stamp.nanosec * 1e-9;
    rpm_data.rpm << msg->rpm[0], 
                    msg->rpm[1],
                    msg->rpm[2],
                    msg->rpm[3],
                    msg->rpm[4],
                    msg->rpm[5];

    if(hexa_rpm_buffer_.is_full())
    {
        hexa_rpm_buffer_.pop();
    }

    hexa_rpm_buffer_.push_back(rpm_data);
}

void L1AdaptiveNode::dobEstimateLoopCallback()
{
    // Implementation of the DOB estimate loop
    // This function will use the data from odom_buffer_ and hexa_rpm_buffer_
    // to compute the disturbance observer estimates and publish the results.

    filtered_odom_publisher_->publish(filtered_odom_msg_);
    dob_publisher_->publish(dob_msg_);
}

void L1AdaptiveNode::odom_filter()
{
    // Implement odometry filtering if needed

    OdomData odom_recent;
    OdomData odom_prev;
    odom_recent = odom_buffer_.get_latest().value();
    odom_prev = odom_buffer_.at(odom_buffer_.size()-2).value();

    double dt = odom_recent.timestamp - odom_prev.timestamp;

    // Apply LPF to linear and angular velocity
    for (int i = 0; i < 3; ++i)
    {
        lin_vel_filtered_(i) = 
        linear_velocity_lpf_[i]->update(odom_recent.linear_velocity(i), 
        dt);

        ang_vel_filtered_(i) =
        angular_velocity_lpf_[i]->update(odom_recent.angular_velocity(i),
        dt);
    }

    // Publish filtered odometry
    filtered_odom_msg_.header.stamp = this->now();
    filtered_odom_msg_.header.frame_id = "odom";
    filtered_odom_msg_.child_frame_id = "base_link";
    filtered_odom_msg_.pose.pose.position.x = odom_recent.position(0);
    filtered_odom_msg_.pose.pose.position.y = odom_recent.position(1);
    filtered_odom_msg_.pose.pose.position.z = odom_recent.position(2);
    filtered_odom_msg_.pose.pose.orientation.w = odom_recent.orientation.w();
    filtered_odom_msg_.pose.pose.orientation.x = odom_recent.orientation.x();
    filtered_odom_msg_.pose.pose.orientation.y = odom_recent.orientation.y();
    filtered_odom_msg_.pose.pose.orientation.z = odom_recent.orientation.z();
    filtered_odom_msg_.twist.twist.linear.x = lin_vel_filtered_(0);
    filtered_odom_msg_.twist.twist.linear.y = lin_vel_filtered_(1);
    filtered_odom_msg_.twist.twist.linear.z = lin_vel_filtered_(2);
    filtered_odom_msg_.twist.twist.angular.x = ang_vel_filtered_(0);
    filtered_odom_msg_.twist.twist.angular.y = ang_vel_filtered_(1);
    filtered_odom_msg_.twist.twist.angular.z = ang_vel_filtered_(2);
}

void L1AdaptiveNode::dob_estimate()
{
    OdomData odom_recent;
    odom_recent = odom_buffer_.get_latest().value();

    size_t odom_recent_index = odom_buffer_.size() - 1;

    OdomData odom_prev = odom_buffer_.at(odom_recent_index-1).value();
    
    Vector6int16 rpm_recent = get_rpm_near_odom(odom_recent.timestamp);
    Vector6int16 rpm_prev = get_rpm_near_odom(odom_prev.timestamp);

    Vector4d u_recent = rpm_to_cmd_converter_->convert(rpm_recent);
    Vector4d u_prev = rpm_to_cmd_converter_->convert(rpm_prev);

    Vector4d u_med = 0.5 * (u_prev + u_recent);

    StateData state_meas;
    state_meas.p = odom_recent.position;
    state_meas.v = lin_vel_filtered_;
    state_meas.q = odom_recent.orientation;
    state_meas.w = ang_vel_filtered_;
    
    // Update L1 adaptive model
    l1_adaptive_model_->update(odom_prev.timestamp,
                               odom_recent.timestamp,
                               state_meas,
                               u_med);

    disturbance_estimate_.tail<4>() = l1_adaptive_model_->get_u_L1();

    // Publish DOB estimate
    dob_msg_.header.stamp = this->now();
    dob_msg_.header.frame_id = "base_link";
    dob_msg_.wrench.force.x = 0.0;
    dob_msg_.wrench.force.y = 0.0;
    dob_msg_.wrench.force.z = disturbance_estimate_(2);
    dob_msg_.wrench.torque.x = disturbance_estimate_(3);
    dob_msg_.wrench.torque.y = disturbance_estimate_(4);
    dob_msg_.wrench.torque.z = disturbance_estimate_(5);
}

Vector6int16 L1AdaptiveNode::get_rpm_near_odom(const double &odom_time_stamp)
{
    size_t rpm_recent_index = hexa_rpm_buffer_.size() - 1;

    double rpm_recent_time_stamp = get_rpm_time_stamp(hexa_rpm_buffer_, rpm_recent_index);

    if(odom_time_stamp < rpm_recent_time_stamp)
    {
        // Search for the closest rpm data before the odom timestamp
        for(int i = rpm_recent_index - 1; i >= 0; --i)
        {
            double rpm_time_stamp = get_rpm_time_stamp(hexa_rpm_buffer_, i);
            if(rpm_time_stamp <= odom_time_stamp)
            {
                Vector6int16 rpm_before = hexa_rpm_buffer_.at(i).value().rpm;
                Vector6int16 rpm_after = hexa_rpm_buffer_.at(i+1).value().rpm;

                double time_before = rpm_time_stamp;
                double time_after = get_rpm_time_stamp(hexa_rpm_buffer_, i+1);
                return rpm_linear_interpolation(odom_time_stamp,
                                                rpm_before,
                                                time_before,
                                                rpm_after,
                                                time_after);
            }
        }
    }
    // If no interpolation is needed, return the most recent rpm data
    return hexa_rpm_buffer_.at(rpm_recent_index).value().rpm;
}

Vector6int16 L1AdaptiveNode::rpm_linear_interpolation(const double &odom_time_stamp,
                                        const Vector6int16 &rpm_before,
                                        const double &time_before,
                                        const Vector6int16 &rpm_after,
                                        const double &time_after)
{
    Vector6int16 rpm_interpolated;
    if(time_after - time_before < 1e-6)
    {
        rpm_interpolated = rpm_before;
        return rpm_interpolated;
    }

    double alpha = (odom_time_stamp - time_before) / (time_after - time_before);
    for(int i = 0; i < 6; ++i)
    {
        rpm_interpolated(i) = static_cast<int16_t>(
            rpm_before(i) + alpha * (rpm_after(i) - rpm_before(i))
        );
    }
    return rpm_interpolated;
}

double L1AdaptiveNode::get_rpm_time_stamp(CircularBuffer<RpmData>& buffer, size_t index)
{
    if(index >= buffer.size())
    {
        throw std::out_of_range("Index out of range in get_rpm_time_stamp");
    }
    return buffer.at(index).value().timestamp;
}

void L1AdaptiveNode::configure_parameters()
{

    // 1. Pass Drone parameters
    this->declare_parameter<double>("drone.m", 3.0);
    double m = this->get_parameter("drone.m").get_value<double>();

    this->declare_parameter<double>("drone.l", 0.265);
    double l = this->get_parameter("drone.l").get_value<double>();

    this->declare_parameter<std::vector<double>>("drone.MoiArray", {0.03, 0.03, 0.05});
    std::vector<double> MoiArray = this->get_parameter("drone.MoiArray").get_value<std::vector<double>>();

    this->declare_parameter<double>("drone.motor_const", 1.465e-07);
    double motor_const = this->get_parameter("drone.motor_const").get_value<double>();

    this->declare_parameter<double>("drone.moment_const", 0.01569);
    double moment_const = this->get_parameter("drone.moment_const").get_value<double>();

    DroneParam drone_param;
    drone_param.m = m;
    drone_param.l = l;
    drone_param.J = Matrix3x3d::Zero();
    for(size_t i = 0; i < 3; ++i){
        drone_param.J(i, i) = MoiArray[i];
    }
    drone_param.C_T = motor_const;
    drone_param.k_m = moment_const;

    rpm_to_cmd_converter_ = new HexaRotorRpmToCmd(drone_param);

    // 2. Pass L1 DOB parameters
    this->declare_parameter<double>("l1_adaptive.dob_looptime", 0.01);
    dob_looptime_ = this->get_parameter("l1_adaptive.dob_looptime").get_value<double>();

    L1AdaptiveParam l1_adaptive_param;

    this->declare_parameter<std::vector<double>>("l1_adaptive.As_array",
    {-5.0, -5.0, -5.0, -5.0, -5.0, -5.0});
    std::vector<double> As_array = this->get_parameter("l1_adaptive.As_array").get_value<std::vector<double>>();
    
    l1_adaptive_param.As = Matrix6x6d::Zero();
    for(size_t i = 0; i < 6; ++i){
        l1_adaptive_param.As(i, i) = As_array[i];
    }

    this->declare_parameter("l1_adaptive.freq_cutoff_trans", 20.0);
    l1_adaptive_param.freq_cutoff_trans = this->get_parameter("l1_adaptive.freq_cutoff_trans").get_value<double>();

    this->declare_parameter("l1_adaptive.freq_cutoff_rot", 60.0);
    l1_adaptive_param.freq_cutoff_rot = this->get_parameter("l1_adaptive.freq_cutoff_rot").get_value<double>();
    
    l1_adaptive_model_ = new L1AdaptationModel(drone_param, l1_adaptive_param);

    // 3. Other parameters
    this->declare_parameter("lpf.lin_vel_cutoff", 20.0);
    double lin_cutoff_freq = this->get_parameter("lpf.lin_vel_cutoff").get_value<double>();

    this->declare_parameter("lpf.ang_vel_cutoff", 60.0);
    double ang_cutoff_freq = this->get_parameter("lpf.ang_vel_cutoff").get_value<double>();

    for(int i = 0; i < 3; ++i){
        linear_velocity_lpf_[i] = new LowPassFilter(lin_cutoff_freq);
        angular_velocity_lpf_[i] = new LowPassFilter(ang_cutoff_freq);
    }

    print_parameters(drone_param, l1_adaptive_param, lin_cutoff_freq, ang_cutoff_freq);

}

void L1AdaptiveNode::print_parameters(const DroneParam &drone_param,
                                const L1AdaptiveParam &l1_adaptive_param,
                                const double &lin_cutoff_freq,
                                const double &ang_cutoff_freq)
{
    RCLCPP_INFO(this->get_logger(), "===== L1 DOB Node Parameters =====");
    RCLCPP_INFO(this->get_logger(), "Drone Parameters:");
    RCLCPP_INFO(this->get_logger(), "Mass (m): %.3f kg", drone_param.m);
    RCLCPP_INFO(this->get_logger(), "Arm length (l): %.3f m", drone_param.l);
    RCLCPP_INFO(this->get_logger(), "Moment of Inertia (J):");
    RCLCPP_INFO(this->get_logger(), "\n%.5f %.5f %.5f\n%.5f %.5f %.5f\n%.5f %.5f %.5f",
                drone_param.J(0,0), drone_param.J(0,1), drone_param.J(0,2),
                drone_param.J(1,0), drone_param.J(1,1), drone_param.J(1,2),
                drone_param.J(2,0), drone_param.J(2,1), drone_param.J(2,2));
    RCLCPP_INFO(this->get_logger(), "Thrust Coefficient (C_T): %.8e", drone_param.C_T);
    RCLCPP_INFO(this->get_logger(), "Moment Coefficient (k_m): %.8e", drone_param.k_m);

    RCLCPP_INFO(this->get_logger(), "L1 Adaptive Parameters:");
    RCLCPP_INFO(this->get_logger(), "DOB Looptime: %.3f ms", dob_looptime_);
    RCLCPP_INFO(this->get_logger(), "Hurwitz Matrix (As):");
    RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f %.3f %.3f\n",
                l1_adaptive_param.As(0,0), l1_adaptive_param.As(1,1), l1_adaptive_param.As(2,2),
                l1_adaptive_param.As(3,3), l1_adaptive_param.As(4,4), l1_adaptive_param.As(5,5));

    RCLCPP_INFO(this->get_logger(), "Cutoff Frequencies for Uncertainty LPF:");
    RCLCPP_INFO(this->get_logger(), "Translational Uncertainty LPF Cutoff Frequency: %.3f Hz", 
    l1_adaptive_param.freq_cutoff_trans);

    RCLCPP_INFO(this->get_logger(), "Rotational Uncertainty LPF Cutoff Frequency: %.3f Hz",
    l1_adaptive_param.freq_cutoff_rot);

    RCLCPP_INFO(this->get_logger(), "Low Pass Filter Cutoff Frequencies:");
    RCLCPP_INFO(this->get_logger(), "Linear Velocity LPF Cutoff Frequency: %.3f Hz", lin_cutoff_freq);
    RCLCPP_INFO(this->get_logger(), "Angular Velocity LPF Cutoff Frequency: %.3f Hz", ang_cutoff_freq);
    RCLCPP_INFO(this->get_logger(), "==========================");
}