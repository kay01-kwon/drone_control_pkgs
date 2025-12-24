#include "node/sync_test_node.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SyncTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}