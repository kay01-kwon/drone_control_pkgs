#include "hgdo_node.hpp"

HgdoNode::HgdoNode()
: Node("hgdo_node")
{
    configure_parameters();

    odom_buffer_.reserve(20);
    hexa_rpm_buffer_.reserve(20);

    // Subscribers
    odom_subscriber_ = this->create_subscription<Odometry>(
        "/mavros/local_position/odom", rclcpp::SensorDataQoS(),
        std::bind(&HgdoNode::odomCallback, this, std::placeholders::_1));

    hexa_rpm_subscriber_ = this->create_subscription<HexaActualRpm>(
        "/uav/actual_rpm", rclcpp::SensorDataQoS(),
        std::bind(&HgdoNode::hexaRpmCallback, this, std::placeholders::_1));

    // Publishers
    filtered_odom_publisher_ = this->create_publisher<Odometry>("/filtered_odometry", rclcpp::SensorDataQoS());
    dob_publisher_ = this->create_publisher<WrenchStamped>("/uav/dob_estimate", rclcpp::SensorDataQoS());

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
    odom_data.linear_velocity << 
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z;
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

void HgdoNode::dobEstimateLoopCallback()
{
    // Implementation of the DOB estimate loop
    // This function will use the data from odom_buffer_ and hexa_rpm_buffer_
    // to compute the disturbance observer estimates and publish the results.
}

void HgdoNode::configure_parameters()
{
    std::string node_name;

    node_name = this->get_name();

    std::string param_prefix = std::string(node_name) + ".";

    // 1. Pass Drone parameters
    this->declare_parameter<double>(param_prefix + "drone.m", 3.0);
    double m = this->get_parameter(param_prefix + "drone.m").as_double();
    
    this->declare_parameter<double>(param_prefix + "drone.l", 0.265);
    double l = this->get_parameter(param_prefix + "drone.l").as_double();

    this->declare_parameter<std::vector<double>>(param_prefix + "drone.MoiArray", {0.03, 0.03, 0.05});
    std::vector<double> MoiArray = this->get_parameter(param_prefix + "drone.MoiArray").as_double_array();

    this->declare_parameter<double>(param_prefix + "drone.motor_const", 1.465e-07);
    double motor_const = this->get_parameter(param_prefix + "drone.motor_const").as_double();

    this->declare_parameter<double>(param_prefix + "drone.moment_const", 0.01569);
    double moment_const = this->get_parameter(param_prefix + "drone.moment_const").as_double();

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
    this->declare_parameter<double>(param_prefix + "dob.control_loop_frequency", 100.0);
    double control_loop_frequency = this->get_parameter(param_prefix + "dob.control_loop_frequency").as_double();
    dob_looptime_ = 1.0 / control_loop_frequency;

    this->declare_parameter<double>(param_prefix + "dob.eps_f", 0.01);
    double eps_f = this->get_parameter(param_prefix + "dob.eps_f").as_double();

    this->declare_parameter<double>(param_prefix + "dob.eps_tau", 0.01);
    double eps_tau = this->get_parameter(param_prefix + "dob.eps_tau").as_double();

    HgdoParam hgdo_param;
    hgdo_param.eps_f = eps_f;
    hgdo_param.eps_tau = eps_tau;


    // 3. Pass LPF parameters
    this->declare_parameter<double>(param_prefix + "lpf.ang_vel_cutoff", 60.0);
    double ang_vel_cutoff = this->get_parameter(param_prefix + "lpf.ang_vel_cutoff").as_double();

    this->declare_parameter<double>(param_prefix + "lpf.lin_vel_cutoff", 20.0);
    double lin_vel_cutoff = this->get_parameter(param_prefix + "lpf.lin_vel_cutoff").as_double();

    // 4. Initialize HGDO model and other utilities

    for(int i = 0; i < 3; ++i){
        angular_velocity_lpf_[i] = new LowPassFilter(ang_vel_cutoff);
        linear_velocity_lpf_[i] = new LowPassFilter(lin_vel_cutoff);
    }

    rpm_to_cmd_converter_ = new HexaRotorRpmToCmd(drone_param);
    hgdo_model_ = new HgdoModel(drone_param, hgdo_param);

    print_parameters(drone_param, hgdo_param);
}

void HgdoNode::print_parameters(const DroneParam &drone_param,
                                const HgdoParam &hgdo_param)
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
    RCLCPP_INFO(this->get_logger(), "================================");
}