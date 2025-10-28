#include "hgdo_node.hpp"

HgdoNode::HgdoNode()
: Node("hgdo_node")
{
    configure_parameters();

    odom_buffer_.reserve(20);
    hexa_rpm_buffer_.reserve(20);

    // Subscribers
    odom_subscriber_ = this->create_subscription<Odometry>(
        "/S550/ground_truth/odom", rclcpp::SensorDataQoS(),
        std::bind(&HgdoNode::odomCallback, this, std::placeholders::_1));

    hexa_rpm_subscriber_ = this->create_subscription<HexaActualRpm>(
        "/uav/actual_rpm", rclcpp::SensorDataQoS(),
        std::bind(&HgdoNode::hexaRpmCallback, this, std::placeholders::_1));

    // Publishers
    filtered_odom_publisher_ = this->create_publisher<Odometry>("/filtered_odom", rclcpp::SensorDataQoS());
    dob_publisher_ = this->create_publisher<WrenchStamped>("/hgdo/wrench", rclcpp::SensorDataQoS());

    // Control loop timer
    control_loop_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(dob_looptime_),
        std::bind(&HgdoNode::dobEstimateLoopCallback, this));
}

HgdoNode::~HgdoNode(){
    delete hgdo_model_;
    delete rpm_to_cmd_converter_;
    for(int i = 0; i < 3; ++i){
        delete angular_velocity_lpf_[i];
        delete linear_velocity_lpf_[i];
    }
}

void HgdoNode::odomCallback(const Odometry::SharedPtr msg)
{
    OdomData odom_data;
    odom_data.timestamp = this->now().seconds();
    odom_data.position <<
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z;

    odom_data.linear_velocity << 
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z;

    odom_data.orientation.w()
    = msg->pose.pose.orientation.w;

    odom_data.orientation.x()
    = msg->pose.pose.orientation.x;

    odom_data.orientation.y()
    = msg->pose.pose.orientation.y;

    odom_data.orientation.z()
    = msg->pose.pose.orientation.z;

    odom_data.angular_velocity << 
        msg->twist.twist.angular.x,
        msg->twist.twist.angular.y,
        msg->twist.twist.angular.z;
    
    if(odom_buffer_.is_full())
    {
        odom_buffer_.pop();
    }

    odom_buffer_.push_back(odom_data);
}

void HgdoNode::hexaRpmCallback(const HexaActualRpm::SharedPtr msg)
{
    RpmData rpm_data;
    rpm_data.timestamp = this->now().seconds();
    rpm_data.rpm << msg->rpm[0], msg->rpm[1], msg->rpm[2],
                    msg->rpm[3], msg->rpm[4], msg->rpm[5];
    
    if(hexa_rpm_buffer_.is_full())
    {
        hexa_rpm_buffer_.pop();
    }

    hexa_rpm_buffer_.push_back(rpm_data);
}

void HgdoNode::dobEstimateLoopCallback()
{
    // Implementation of the DOB estimate loop
    // This function will use the data from odom_buffer_ and hexa_rpm_buffer_
    // to compute the disturbance observer estimates and publish the results.
    
    t_curr_ = this->now().seconds();

    if(odom_buffer_.is_empty() || hexa_rpm_buffer_.is_empty())
    {
        // RCLCPP_INFO(this->get_logger(), "Waiting for sufficient data in buffers...");
        return;
    }


    OdomData odom_recent;
    RpmData rpm_recent;

    odom_recent = odom_buffer_.get_latest().value();
    rpm_recent = hexa_rpm_buffer_.get_latest().value();

    size_t odom_recent_index = odom_buffer_.size() - 1;
    size_t rpm_recent_index = hexa_rpm_buffer_.size() - 1;

    Vector3d lin_vel_filtered = Vector3d::Zero();
    Vector3d ang_vel_filtered = Vector3d::Zero();

    if (odom_recent_index >= 2 && rpm_recent_index >= 2)
    {
        OdomData odom_data_prev = odom_buffer_.at(odom_recent_index-1).value();
        RpmData rpm_data_prev = hexa_rpm_buffer_.at(rpm_recent_index-1).value();
        double dt_odom = odom_recent.timestamp - odom_data_prev.timestamp;

        // RCLCPP_INFO(this->get_logger(), "dt_odom: %.6f", dt_odom);
        // RCLCPP_INFO(this->get_logger(), "dt_rpm: %.6f", rpm_recent.timestamp - rpm_data_prev.timestamp);
        // Apply LPF to linear and angular velocity
        for (int i = 0; i < 3; ++i)
        {
            lin_vel_filtered(i) = 
            linear_velocity_lpf_[i]->update(odom_recent.linear_velocity(i), 
            dt_odom);

            ang_vel_filtered(i) =
            angular_velocity_lpf_[i]->update(odom_recent.angular_velocity(i), 
            dt_odom);
        }

        Quaterniond q_body_to_world = odom_recent.orientation.conjugate();

        Vector3d lin_vel_world = q_body_to_world * lin_vel_filtered;

        // Vector4d u_curr = rpm_to_cmd_converter_->convert(rpm_recent.rpm);
        Vector4d u_prev = rpm_to_cmd_converter_->convert(rpm_data_prev.rpm);
        // Vector4d u_med = (u_curr + u_prev) / 2.0;

        hgdo_model_->update(t_prev_, 
                            t_curr_, 
                            lin_vel_world, 
                            ang_vel_filtered,
                            odom_recent.orientation, 
                            u_prev);
        disturbance_estimate_ = hgdo_model_->get_disturbance_estimate();
    }
    else
    {
        lin_vel_filtered = odom_recent.linear_velocity;
        ang_vel_filtered = odom_recent.angular_velocity;
    }

    // Publish filtered odometry
    Odometry filtered_odom_msg;
    filtered_odom_msg.header.stamp = this->now();
    filtered_odom_msg.header.frame_id = "odom";
    filtered_odom_msg.child_frame_id = "base_link";
    filtered_odom_msg.pose.pose.position.x = odom_recent.position(0);
    filtered_odom_msg.pose.pose.position.y = odom_recent.position(1);
    filtered_odom_msg.pose.pose.position.z = odom_recent.position(2);
    filtered_odom_msg.pose.pose.orientation.w = odom_recent.orientation.w();
    filtered_odom_msg.pose.pose.orientation.x = odom_recent.orientation.x();
    filtered_odom_msg.pose.pose.orientation.y = odom_recent.orientation.y();
    filtered_odom_msg.pose.pose.orientation.z = odom_recent.orientation.z();
    filtered_odom_msg.twist.twist.linear.x = lin_vel_filtered(0);
    filtered_odom_msg.twist.twist.linear.y = lin_vel_filtered(1);
    filtered_odom_msg.twist.twist.linear.z = lin_vel_filtered(2);
    filtered_odom_msg.twist.twist.angular.x = ang_vel_filtered(0);
    filtered_odom_msg.twist.twist.angular.y = ang_vel_filtered(1);
    filtered_odom_msg.twist.twist.angular.z = ang_vel_filtered(2);
    filtered_odom_publisher_->publish(filtered_odom_msg);

    // Publish DOB estimate
    WrenchStamped dob_msg;
    dob_msg.header.stamp = this->now();
    dob_msg.header.frame_id = "base_link";
    dob_msg.wrench.force.x = disturbance_estimate_(0);
    dob_msg.wrench.force.y = disturbance_estimate_(1);
    dob_msg.wrench.force.z = disturbance_estimate_(2);
    dob_msg.wrench.torque.x = disturbance_estimate_(3);
    dob_msg.wrench.torque.y = disturbance_estimate_(4);
    dob_msg.wrench.torque.z = disturbance_estimate_(5);
    dob_publisher_->publish(dob_msg);
    
    t_prev_ = t_curr_;
}

Vector3d HgdoNode::linear_interpolation(const Vector3d &start,
                                     const Vector3d &end,
                                     const double &t_start,
                                     const double &t_end,
                                     const double &t_query)
{
    if(t_query <= t_start){
        return start;
    }
    else if(t_query >= t_end){
        return end;
    }
    else{
        double alpha = (t_query - t_start) / (t_end - t_start);
        return (1.0 - alpha) * start + alpha * end;
    }
}

void HgdoNode::configure_parameters()
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

    // 2. Pass Hgdo parameters
    this->declare_parameter<double>("dob.dob_looptime", 100.0);
    dob_looptime_ = this->get_parameter("dob.dob_looptime").get_value<double>();

    this->declare_parameter<double>("dob.eps_f", 0.01);
    double eps_f = this->get_parameter("dob.eps_f").get_value<double>();

    this->declare_parameter<double>("dob.eps_tau", 0.01);
    double eps_tau = this->get_parameter("dob.eps_tau").get_value<double>();

    HgdoParam hgdo_param;
    hgdo_param.eps_f = eps_f;
    hgdo_param.eps_tau = eps_tau;


    // 3. Pass LPF parameters
    this->declare_parameter<double>("lpf.ang_vel_cutoff", 60.0);
    double ang_vel_cutoff = this->get_parameter("lpf.ang_vel_cutoff").as_double();

    this->declare_parameter<double>("lpf.lin_vel_cutoff", 20.0);
    double lin_vel_cutoff = this->get_parameter("lpf.lin_vel_cutoff").as_double();

    // 4. Initialize HGDO model and other utilities

    for(int i = 0; i < 3; ++i){
        angular_velocity_lpf_[i] = new LowPassFilter(ang_vel_cutoff);
        linear_velocity_lpf_[i] = new LowPassFilter(lin_vel_cutoff);
    }

    rpm_to_cmd_converter_ = new HexaRotorRpmToCmd(drone_param);
    hgdo_model_ = new HgdoModel(drone_param, hgdo_param);

    print_parameters(drone_param, hgdo_param, ang_vel_cutoff, lin_vel_cutoff);
}

void HgdoNode::print_parameters(const DroneParam &drone_param,
                                const HgdoParam &hgdo_param,
                                const double &ang_cutoff_freq,
                                const double &lin_cutoff_freq)
{
    RCLCPP_INFO(this->get_logger(), "===== HGDO Node Parameters =====");
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

    RCLCPP_INFO(this->get_logger(), "HGDO Parameters:");
    RCLCPP_INFO(this->get_logger(), "Epsilon for Force Estimation (eps_f): %.5f", hgdo_param.eps_f);
    RCLCPP_INFO(this->get_logger(), "Epsilon for Torque Estimation (eps_tau): %.5f", hgdo_param.eps_tau);

    RCLCPP_INFO(this->get_logger(), "Low-Pass Filter Cutoff Frequencies:");
    RCLCPP_INFO(this->get_logger(), 
    "Angular Velocity LPF Cutoff Frequency: %.2f Hz", ang_cutoff_freq);
    RCLCPP_INFO(this->get_logger(), 
    "Linear Velocity LPF Cutoff Frequency: %.2f Hz", lin_cutoff_freq);
    RCLCPP_INFO(this->get_logger(), "==========================");
}