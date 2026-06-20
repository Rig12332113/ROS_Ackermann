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
        self.raw_vo_path = Path()
        self.opt_vo_path = Path()

        self.gt_path.header.frame_id = self.compare_frame
        self.raw_vo_path.header.frame_id = self.compare_frame
        self.opt_vo_path.header.frame_id = self.compare_frame

        self.gt_origin = None
        self.raw_vo_origin = None
        self.opt_vo_origin = None

        self.gt_pub = self.create_publisher(
            Path,
            "/compare/ground_truth_path",
            10,
        )

        self.raw_vo_pub = self.create_publisher(
            Path,
            "/compare/raw_rgbd_vo_path",
            10,
        )

        self.opt_vo_pub = self.create_publisher(
            Path,
            "/compare/optimized_rgbd_vo_path",
            10,
        )

        self.create_subscription(
            Odometry,
            "/ground_truth/odom",
            self.gt_callback,
            10,
        )

        self.create_subscription(
            Path,
            "/rgbd_vo/path",
            self.raw_vo_path_callback,
            10,
        )

        self.create_subscription(
            Path,
            "/rgbd_slam/optimized_path",
            self.opt_vo_path_callback,
            10,
        )

        self.get_logger().info(
            "Comparing /ground_truth/odom, /rgbd_vo/path, and /rgbd_slam/optimized_path"
        )

    def make_pose_from_odom(self, msg, origin):
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
        pose.pose.position.z = 0.0

        pose.pose.orientation = msg.pose.pose.orientation

        return pose, origin

    def normalize_path(self, path_msg, origin):
        out_path = Path()
        out_path.header.stamp = path_msg.header.stamp
        out_path.header.frame_id = self.compare_frame

        if len(path_msg.poses) == 0:
            return out_path, origin

        if origin is None:
            p0 = path_msg.poses[0].pose.position
            origin = (p0.x, p0.y, p0.z)

        for pose_in in path_msg.poses:
            p = pose_in.pose.position

            pose_out = PoseStamped()
            pose_out.header.stamp = pose_in.header.stamp
            pose_out.header.frame_id = self.compare_frame

            pose_out.pose.position.x = p.x - origin[0]
            pose_out.pose.position.y = p.y - origin[1]
            pose_out.pose.position.z = 0.0

            pose_out.pose.orientation = pose_in.pose.orientation

            out_path.poses.append(pose_out)

        return out_path, origin

    def gt_callback(self, msg):
        pose, self.gt_origin = self.make_pose_from_odom(msg, self.gt_origin)

        self.gt_path.header.stamp = msg.header.stamp
        self.gt_path.poses.append(pose)

        self.gt_pub.publish(self.gt_path)

    def raw_vo_path_callback(self, msg):
        self.raw_vo_path, self.raw_vo_origin = self.normalize_path(
            msg,
            self.raw_vo_origin,
        )

        self.raw_vo_pub.publish(self.raw_vo_path)

    def opt_vo_path_callback(self, msg):
        self.opt_vo_path, self.opt_vo_origin = self.normalize_path(
            msg,
            self.opt_vo_origin,
        )

        self.opt_vo_pub.publish(self.opt_vo_path)


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
