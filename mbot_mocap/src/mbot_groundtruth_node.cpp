#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>

class GroundTruthNode : public rclcpp::Node {
public:
  GroundTruthNode() : Node("ground_truth_node") {
    sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/ground_truth/mbot/mbot/pose", 10,
        std::bind(&GroundTruthNode::cb, this, std::placeholders::_1));

    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/ground_truth/mbot_corrected", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    warmup_target_ = declare_parameter<int>("warmup_samples", 10);
    lever_arm_x_m_ = declare_parameter<double>("mocap_lever_arm_x_m", 0.0);
    lever_arm_y_m_ = declare_parameter<double>("mocap_lever_arm_y_m", 0.0);
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  double p0x_{0.0}, p0y_{0.0};
  double yaw0_sin_{0.0}, yaw0_cos_{0.0};
  double lever_arm_x_m_{0.0}, lever_arm_y_m_{0.0};
  int warmup_{0}, warmup_target_{10};
  bool locked_{false};

  static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }
  static geometry_msgs::msg::Quaternion quatFromYaw(double yaw) {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0; q.y = 0.0;
    q.z = std::sin(yaw * 0.5);
    q.w = std::cos(yaw * 0.5);
    return q;
  }
  static double wrap2pi(double a) {
    const double TWO_PI = 2.0 * M_PI;
    a = std::fmod(a + M_PI, TWO_PI);
    if (a < 0) a += TWO_PI;
    return a - M_PI;
  }

  void cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    const auto& p = msg->pose.position;
    const double yaw = yawFromQuat(msg->pose.orientation);
    const double c_yaw = std::cos(yaw);
    const double s_yaw = std::sin(yaw);
    const double px = p.x - (c_yaw * lever_arm_x_m_ - s_yaw * lever_arm_y_m_);
    const double py = p.y - (s_yaw * lever_arm_x_m_ + c_yaw * lever_arm_y_m_);

    if (!locked_) {
      p0x_ += px; p0y_ += py;
      yaw0_sin_ += std::sin(yaw);
      yaw0_cos_ += std::cos(yaw);
      if (++warmup_ >= warmup_target_) {
        const double n = static_cast<double>(warmup_);
        p0x_ /= n; p0y_ /= n;
        yaw0_sin_ /= n; yaw0_cos_ /= n;
        locked_ = true;
      }
      return;
    }

    const double yaw0 = wrap2pi(0.5 * M_PI + std::atan2(yaw0_sin_, yaw0_cos_));

    const double dx = px - p0x_;
    const double dy = py - p0y_;
    const double c = std::cos(-yaw0), s = std::sin(-yaw0);
    const double x_rel = c * dx - s * dy;
    const double y_rel = s * dx + c * dy;
    const double yaw_rel = wrap2pi(yaw - yaw0 + 0.5*M_PI);

    geometry_msgs::msg::PoseStamped out;
    out.header.stamp = this->now();
    out.header.frame_id = "odom";
    out.pose.position.x = x_rel;
    out.pose.position.y = y_rel;
    out.pose.position.z = 0.0;
    out.pose.orientation = quatFromYaw(yaw_rel);
    pub_pose_->publish(out);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = this->now();
    tf.header.frame_id = "odom";
    tf.child_frame_id = "base_footprint_gt";
    tf.transform.translation.x = x_rel;
    tf.transform.translation.y = y_rel;
    tf.transform.translation.z = 0.0;
    tf.transform.rotation = quatFromYaw(yaw_rel);
    tf_broadcaster_->sendTransform(tf);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GroundTruthNode>());
  rclcpp::shutdown();
  return 0;
}
