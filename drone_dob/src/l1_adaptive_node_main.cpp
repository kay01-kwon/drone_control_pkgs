#include "node/l1_adaptive_node.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<L1AdaptiveNode>());
    rclcpp::shutdown();
    return 0;
}