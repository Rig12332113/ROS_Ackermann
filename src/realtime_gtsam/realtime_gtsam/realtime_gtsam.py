import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import tf_transformations
from realtime_gtsam.keyframe_manager import KeyframeManager
from realtime_gtsam.pose_graph import RealtimePoseGraph


class RGBD_Node(Node):
    def __init__(self):
        super().__init__("rgbd_subscriber")
        self.get_logger().info("rgbd_node create")

        self.rgb_sub = Subscriber(
            self,
            Image,
            "/depth_camera/image"
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            "/depth_camera/depth_image"
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.03
        )

        self.ts.registerCallback(self.rgbdCallBack)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/depth_camera/camera_info",
            self.cameraCallBack,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            "/rgbd_vo/path",
            10
        )

        self.opt_path_pub = self.create_publisher(
            Path,
            "/rgbd_slam/optimized_path",
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/rgbd_vo/odom",
            10
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.opt_path_msg = Path()
        self.opt_path_msg.header.frame_id = "map"

        self.raw_camera_pose_history = []

        self.has_camera_info = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.intrinsic = None

        self.num_match = 125
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
        self.prev_rgb = None

        self.T_base_camera = np.array([
            [0.0,  0.0,  1.0, 0.310],
            [-1.0, 0.0,  0.0, 0.000],
            [0.0, -1.0,  0.0, 0.250],
            [0.0,  0.0,  0.0, 1.000],
        ], dtype=np.float64)

        self.T_camera_base = np.linalg.inv(self.T_base_camera)

        self.T_world_camera = self.T_base_camera.copy()
        self.T_map_odom = np.eye(4)

        self.kf_manager = KeyframeManager()
        self.pose_graph = RealtimePoseGraph()

    def rgbdCallBack(self, rgb_msg, depth_msg):
        if not self.has_camera_info:
            print("Waiting for camera_info...")
            return

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        valid_keypoints = []
        valid_descriptors = []

        if descriptors is not None:
            for i, kp in enumerate(keypoints):
                u = int(kp.pt[0])
                v = int(kp.pt[1])

                if u < 0 or u >= depth_image.shape[1] or v < 0 or v >= depth_image.shape[0]:
                    continue

                z = depth_image[v, u]

                if not np.isfinite(z) or (z <= 0.1 or z >= 10.0):
                    continue

                valid_keypoints.append(kp)
                valid_descriptors.append(descriptors[i])

        if len(valid_descriptors) > 0:
            valid_descriptors = np.array(valid_descriptors, dtype=np.uint8)
        else:
            valid_descriptors = None

        if self.prev_descriptors is not None and valid_descriptors is not None:
            inliers, T_prev_curr = self.estimateRelativePose(
                prev_keypoints=self.prev_keypoints,
                prev_descriptors=self.prev_descriptors,
                prev_depth=self.prev_depth,
                curr_keypoints=valid_keypoints,
                curr_descriptors=valid_descriptors
            )

            if T_prev_curr is not None:
                self.T_world_camera = self.T_world_camera @ T_prev_curr
                self.raw_camera_pose_history.append(self.T_world_camera.copy())

                stamp = self.get_clock().now().to_msg()

                # -------------------------
                # Raw VO path
                # -------------------------
                T_raw_base = self.T_world_camera @ self.T_camera_base
                raw_position = T_raw_base[:3, 3]
                raw_quat = tf_transformations.quaternion_from_matrix(T_raw_base)

                raw_pose_msg = PoseStamped()
                raw_pose_msg.header.stamp = stamp
                raw_pose_msg.header.frame_id = "map"

                raw_pose_msg.pose.position.x = float(raw_position[0])
                raw_pose_msg.pose.position.y = float(raw_position[1])
                raw_pose_msg.pose.position.z = float(raw_position[2])

                raw_pose_msg.pose.orientation.x = float(raw_quat[0])
                raw_pose_msg.pose.orientation.y = float(raw_quat[1])
                raw_pose_msg.pose.orientation.z = float(raw_quat[2])
                raw_pose_msg.pose.orientation.w = float(raw_quat[3])

                self.path_msg.header.stamp = stamp
                self.path_msg.poses.append(raw_pose_msg)
                self.path_pub.publish(self.path_msg)

                # -------------------------
                # Optimized full corrected path
                # -------------------------
                self.opt_path_msg = Path()
                self.opt_path_msg.header.stamp = stamp
                self.opt_path_msg.header.frame_id = "map"

                for T_world_camera_raw in self.raw_camera_pose_history:
                    T_map_camera = self.T_map_odom @ T_world_camera_raw
                    T_map_base = T_map_camera @ self.T_camera_base

                    opt_position = T_map_base[:3, 3]
                    opt_quat = tf_transformations.quaternion_from_matrix(T_map_base)

                    opt_pose_msg = PoseStamped()
                    opt_pose_msg.header.stamp = stamp
                    opt_pose_msg.header.frame_id = "map"

                    opt_pose_msg.pose.position.x = float(opt_position[0])
                    opt_pose_msg.pose.position.y = float(opt_position[1])
                    opt_pose_msg.pose.position.z = float(opt_position[2])

                    opt_pose_msg.pose.orientation.x = float(opt_quat[0])
                    opt_pose_msg.pose.orientation.y = float(opt_quat[1])
                    opt_pose_msg.pose.orientation.z = float(opt_quat[2])
                    opt_pose_msg.pose.orientation.w = float(opt_quat[3])

                    self.opt_path_msg.poses.append(opt_pose_msg)

                self.opt_path_pub.publish(self.opt_path_msg)

                # -------------------------
                # Corrected current odom
                # -------------------------
                T_map_camera = self.T_map_odom @ self.T_world_camera
                T_opt_base = T_map_camera @ self.T_camera_base

                opt_position = T_opt_base[:3, 3]
                opt_quat = tf_transformations.quaternion_from_matrix(T_opt_base)

                odom_msg = Odometry()
                odom_msg.header.stamp = stamp
                odom_msg.header.frame_id = "map"
                odom_msg.child_frame_id = "base_link"

                odom_msg.pose.pose.position.x = float(opt_position[0])
                odom_msg.pose.pose.position.y = float(opt_position[1])
                odom_msg.pose.pose.position.z = float(opt_position[2])

                odom_msg.pose.pose.orientation.x = float(opt_quat[0])
                odom_msg.pose.pose.orientation.y = float(opt_quat[1])
                odom_msg.pose.pose.orientation.z = float(opt_quat[2])
                odom_msg.pose.pose.orientation.w = float(opt_quat[3])

                self.odom_pub.publish(odom_msg)

        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        succ = self.kf_manager.add_keyframe(
            timestamp,
            rgb_image,
            depth_image,
            self.T_world_camera,
            valid_keypoints,
            valid_descriptors,
        )

        if succ:
            keyframes = self.kf_manager.get_keyframes()
            new_kf = keyframes[-1]

            if new_kf.id == 0:
                self.pose_graph.add_first_keyframe(
                    keyframe_id=new_kf.id,
                    T_world_keyframe=new_kf.pose,
                )

            else:
                prev_kf = keyframes[-2]

                T_prev_curr = np.linalg.inv(prev_kf.pose) @ new_kf.pose

                self.pose_graph.add_keyframe(
                    keyframe_id=new_kf.id,
                    prev_keyframe_id=prev_kf.id,
                    T_prev_curr=T_prev_curr,
                )

                for kf in keyframes:
                    if kf.id >= new_kf.id - 15:          # TODO: tune skip close keyframe
                        continue

                    inliers, T_old_new = self.estimateRelativePose(
                        kf.keypoints,
                        kf.descriptors,
                        kf.depth,
                        new_kf.keypoints,
                        new_kf.descriptors,
                    )

                    if T_old_new is None:
                        continue

                    if inliers >= 100:                   # TODO: tune inlier number
                        self.pose_graph.add_loop_factor(
                            old_keyframe_id=kf.id,
                            new_keyframe_id=new_kf.id,
                            T_old_new=T_old_new,
                        )

                        self.get_logger().info(
                            f"Loop closure added: KF {kf.id} -> KF {new_kf.id}, inliers={inliers}"
                        )

                        break

            optimized_poses = self.pose_graph.get_optimized_poses()

            if new_kf.id in optimized_poses:
                T_map_kf_opt = optimized_poses[new_kf.id]
                T_odom_kf = new_kf.pose

                self.T_map_odom = T_map_kf_opt @ np.linalg.inv(T_odom_kf)

                corr_t = np.linalg.norm(self.T_map_odom[:3, 3])
                corr_r = np.arccos(
                    np.clip(
                        (np.trace(self.T_map_odom[:3, :3]) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )

                self.get_logger().info(
                    f"T_map_odom correction: trans={corr_t:.4f}, rot={corr_r:.4f}"
                )

        self.prev_keypoints = valid_keypoints
        self.prev_descriptors = valid_descriptors
        self.prev_depth = depth_image
        self.prev_rgb = rgb_image

    def estimateRelativePose(
        self,
        prev_keypoints,
        prev_descriptors,
        prev_depth,
        curr_keypoints,
        curr_descriptors
    ):
        if prev_descriptors is None or curr_descriptors is None:
            return 0, None

        matches = self.matcher.match(prev_descriptors, curr_descriptors)
        matches = sorted(matches, key=lambda m: m.distance)

        good_matches = matches[:self.num_match]

        object_points = []
        image_points = []

        for match in good_matches:
            prev_kp = prev_keypoints[match.queryIdx]
            u_prev, v_prev = prev_kp.pt
            u_prev = int(u_prev)
            v_prev = int(v_prev)

            if (
                u_prev < 0 or u_prev >= prev_depth.shape[1]
                or v_prev < 0 or v_prev >= prev_depth.shape[0]
            ):
                continue

            Z = prev_depth[v_prev, u_prev]

            if not np.isfinite(Z) or Z <= 0.1 or Z >= 10.0:
                continue

            curr_kp = curr_keypoints[match.trainIdx]
            u_curr, v_curr = curr_kp.pt

            X = (u_prev - self.cx) * Z / self.fx
            Y = (v_prev - self.cy) * Z / self.fy

            object_points.append([X, Y, Z])
            image_points.append([u_curr, v_curr])

        if len(object_points) < 10:
            return 0, None

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, t, inlier = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.intrinsic,
            np.zeros((5, 1)),
        )

        if not success or inlier is None:
            return 0, None

        R, _ = cv2.Rodrigues(rvec)

        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = t.ravel()

        T_prev_curr = np.linalg.inv(T_curr_prev)

        inlier_count = len(inlier)

        return inlier_count, T_prev_curr

    def cameraCallBack(self, msg):
        if not self.has_camera_info:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsic = msg.k.reshape((3, 3))
            self.has_camera_info = True
            print("get camera info")
        else:
            return


def main(args=None):
    rclpy.init(args=args)
    node = RGBD_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()