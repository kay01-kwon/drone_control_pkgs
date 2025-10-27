cmake_minimum_required(VERSION 3.8)
project(drone_dob)

set(CMAKE_CXX_STANDARD 17)

set(dob_src
  include/models/hgdo_model.cpp
  include/node/hgdo_node.cpp
  include/utils/HexaRotorRpmToCmd.cpp
  include/utils/LowPassFilter.cpp
  include/utils/math_tool.cpp
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ros2_libcanard_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)


include_directories(
  include
)

add_library(dob_lib
  ${dob_src}
)

ament_target_dependencies(
  dob_lib
  ros2_libcanard_msgs
  nav_msgs
  geometry_msgs
  rclcpp
)

add_executable(
  hgdo_node
  src/hgdo_node_main.cpp
)

target_link_libraries(
  hgdo_node
  dob_lib
)

install(
  TARGETS hgdo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
