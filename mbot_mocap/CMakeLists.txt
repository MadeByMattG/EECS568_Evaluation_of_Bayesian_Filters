cmake_minimum_required(VERSION 3.16)


project(mbot_mocap)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################
# DEPENDENCIES #
################

find_package(Boost REQUIRED COMPONENTS thread date_time chrono)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


# include_directories(
# 	${Boost_INCLUDE_DIRS}
# 	include
# 	)

##################
# ROS EXECUTABLE #
##################


add_executable(mbot_ground_truth src/mbot_groundtruth_node.cpp)
ament_target_dependencies(mbot_ground_truth
	rclcpp
	tf2
	tf2_ros
	tf2_geometry_msgs
	geometry_msgs
)

install(TARGETS
	mbot_ground_truth
	DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
	launch
	DESTINATION share/${PROJECT_NAME}/
)

ament_package()
