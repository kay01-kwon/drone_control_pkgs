#include "node/hgdo_node.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HgdoNode>());
    rclcpp::shutdown();
    return 0;
}