#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class OdomPathCompare(Node):
    def __init__(self):
        super().__init__("odom_path_compare")

        self.compare_frame = "compare"

        self.gt_path = Path()
        self.rtab_path = Path()
        self.vo_path = Path()

        self.gt_path.header.frame_id = self.compare_frame
        self.rtab_path.header.frame_id = self.compare_frame
        self.vo_path.header.frame_id = self.compare_frame

        self.gt_origin = None
        self.rtab_origin = None
        self.vo_origin = None

        self.gt_pub = self.create_publisher(
            Path,
            "/compare/ground_truth_path",
            10
        )

        self.rtab_pub = self.create_publisher(
            Path,
            "/compare/rtabmap_path",
            10
        )

        self.vo_pub = self.create_publisher(
            Path,
            "/compare/rgbd_vo_path",
            10
        )

        self.create_subscription(
            Odometry,
            "/ground_truth/odom",
            self.gt_callback,
            10
        )

        self.create_subscription(
            Odometry,
            "/rtabmap/odom",
            self.rtab_callback,
            10
        )

        self.create_subscription(
            Odometry,
            "/rgbd_vo/odom",
            self.vo_callback,
            10
        )

        self.get_logger().info(
            "Comparing /ground_truth/odom, /rtabmap/odom, and /rgbd_vo/odom"
        )

    def make_pose(self, msg, origin):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        if origin is None:
            origin = (x, y, z)

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.compare_frame

        pose.pose.position.x = x - origin[0]
        pose.pose.position.y = y - origin[1]
        # pose.pose.position.z = z - origin[2]
        pose.pose.position.z = 0.0

        pose.pose.orientation = msg.pose.pose.orientation

        return pose, origin

    def gt_callback(self, msg):
        pose, self.gt_origin = self.make_pose(msg, self.gt_origin)

        self.gt_path.header.stamp = msg.header.stamp
        self.gt_path.poses.append(pose)

        self.gt_pub.publish(self.gt_path)

    def rtab_callback(self, msg):
        pose, self.rtab_origin = self.make_pose(msg, self.rtab_origin)

        self.rtab_path.header.stamp = msg.header.stamp
        self.rtab_path.poses.append(pose)

        self.rtab_pub.publish(self.rtab_path)

    def vo_callback(self, msg):
        pose, self.vo_origin = self.make_pose(msg, self.vo_origin)

        self.vo_path.header.stamp = msg.header.stamp
        self.vo_path.poses.append(pose)

        self.vo_pub.publish(self.vo_path)


def main(args=None):
    rclpy.init(args=args)

    node = OdomPathCompare()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()