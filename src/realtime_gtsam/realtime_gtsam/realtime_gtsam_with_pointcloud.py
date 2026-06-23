import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2

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

        self.rgb_sub = Subscriber(self, Image, "/depth_camera/image")
        self.depth_sub = Subscriber(self, Image, "/depth_camera/depth_image")

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.03,
        )
        self.ts.registerCallback(self.rgbdCallBack)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/depth_camera/camera_info",
            self.cameraCallBack,
            10,
        )

        self.path_pub = self.create_publisher(Path, "/rgbd_vo/path", 10)
        self.opt_path_pub = self.create_publisher(Path, "/rgbd_slam/optimized_path", 10)
        self.odom_pub = self.create_publisher(Odometry, "/rgbd_vo/odom", 10)

        self.kf_map_pub = self.create_publisher(
            PointCloud2,
            "/rgbd_slam/keyframe_map",
            10,
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

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

        self.min_loop_gap = 15
        self.min_loop_inliers = 60
        self.max_loop_trans_error = 6.0
        self.max_loop_rot_error = 1.2
        self.max_loop_translation = 7.0
        self.loop_search_stride = 1

        self.min_vo_inliers = 30
        self.max_vo_translation = 0.80
        self.max_vo_rotation = 0.80

        self.local_tracking_window = 5
        self.local_tracking_min_inliers = 35
        self.local_tracking_max_translation = 1.20
        self.local_tracking_max_rotation = 1.00

        self.map_sample_step = 8
        self.map_depth_min = 0.2
        self.map_depth_max = 8.0
        self.max_points_per_keyframe = 3000
        self.max_total_map_points = 250000

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

                if u < 0 or u >= depth_image.shape[1]:
                    continue
                if v < 0 or v >= depth_image.shape[0]:
                    continue

                z = depth_image[v, u]

                if not np.isfinite(z) or z <= 0.1 or z >= 15.0:
                    continue

                valid_keypoints.append(kp)
                valid_descriptors.append(descriptors[i])

        if len(valid_descriptors) > 0:
            valid_descriptors = np.array(valid_descriptors, dtype=np.uint8)
        else:
            valid_descriptors = None

        vo_success = False

        if valid_descriptors is not None:
            success, T_world_camera_new, track_info = self.track_against_local_keyframes(
                curr_keypoints=valid_keypoints,
                curr_descriptors=valid_descriptors,
            )

            if success:
                self.T_world_camera = T_world_camera_new
                self.raw_camera_pose_history.append(self.T_world_camera.copy())

                stamp = self.get_clock().now().to_msg()
                self.publish_raw_path(stamp)
                self.publish_corrected_odom(stamp)

                vo_success = True

                self.get_logger().info(
                    f"Local track: anchor={track_info['anchor_id']}, "
                    f"inliers={track_info['inliers']}, "
                    f"trans={track_info['trans']:.3f}, "
                    f"rot={track_info['rot']:.3f}"
                )

        if not vo_success and self.prev_descriptors is not None and valid_descriptors is not None:
            inliers, T_prev_curr = self.estimateRelativePose(
                prev_keypoints=self.prev_keypoints,
                prev_descriptors=self.prev_descriptors,
                prev_depth=self.prev_depth,
                curr_keypoints=valid_keypoints,
                curr_descriptors=valid_descriptors,
            )

            if self.is_good_vo_motion(T_prev_curr, inliers):
                self.T_world_camera = self.T_world_camera @ T_prev_curr
                self.raw_camera_pose_history.append(self.T_world_camera.copy())

                stamp = self.get_clock().now().to_msg()
                self.publish_raw_path(stamp)
                self.publish_corrected_odom(stamp)

                vo_success = True

            else:
                if T_prev_curr is not None:
                    trans = np.linalg.norm(T_prev_curr[:3, 3])
                    rot = self.rotation_angle(T_prev_curr[:3, :3])
                else:
                    trans = -1.0
                    rot = -1.0

                self.get_logger().info(
                    f"Reject bad VO: inliers={inliers}, "
                    f"trans={trans:.3f}, rot={rot:.3f}"
                )

        if self.prev_descriptors is not None and not vo_success:
            self.prev_keypoints = valid_keypoints
            self.prev_descriptors = valid_descriptors
            self.prev_depth = depth_image
            self.prev_rgb = rgb_image
            return

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

                self.try_add_loop_closure(
                    keyframes=keyframes,
                    new_kf=new_kf,
                )

            self.update_keyframe_poses_from_gtsam()
            self.publish_optimized_keyframe_path()
            self.publish_keyframe_map()

            optimized_poses = self.pose_graph.get_optimized_poses()

            if new_kf.id in optimized_poses:
                T_map_kf_opt = optimized_poses[new_kf.id]
                T_odom_kf = new_kf.pose

                T_target = T_map_kf_opt @ np.linalg.inv(T_odom_kf)
                self.T_map_odom = T_target.copy()

                raw_p = new_kf.pose[:3, 3]
                opt_p = T_map_kf_opt[:3, 3]
                diff = np.linalg.norm(opt_p - raw_p)

                self.get_logger().info(
                    f"KF {new_kf.id}: raw={raw_p}, opt={opt_p}, diff={diff:.3f}"
                )

        self.prev_keypoints = valid_keypoints
        self.prev_descriptors = valid_descriptors
        self.prev_depth = depth_image
        self.prev_rgb = rgb_image

    def depth_to_camera_points(self, depth_image):
        h, w = depth_image.shape

        points = []

        for v in range(0, h, self.map_sample_step):
            for u in range(0, w, self.map_sample_step):
                z = depth_image[v, u]

                if not np.isfinite(z):
                    continue

                if z < self.map_depth_min or z > self.map_depth_max:
                    continue

                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy

                points.append([x, y, z, 1.0])

                if len(points) >= self.max_points_per_keyframe:
                    break

            if len(points) >= self.max_points_per_keyframe:
                break

        if len(points) == 0:
            return np.empty((0, 4), dtype=np.float64)

        return np.array(points, dtype=np.float64)

    def publish_keyframe_map(self):
        keyframes = self.kf_manager.get_keyframes()

        if len(keyframes) == 0:
            return

        map_points = []

        for kf in keyframes:
            T_map_camera = kf.optimized_pose

            points_camera = self.depth_to_camera_points(kf.depth)

            if points_camera.shape[0] == 0:
                continue

            points_map_h = (T_map_camera @ points_camera.T).T
            points_map = points_map_h[:, :3]

            for p in points_map:
                map_points.append([
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                ])

                if len(map_points) >= self.max_total_map_points:
                    break

            if len(map_points) >= self.max_total_map_points:
                break

        if len(map_points) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            map_points,
        )

        self.kf_map_pub.publish(cloud_msg)

    def track_against_local_keyframes(self, curr_keypoints, curr_descriptors):
        keyframes = self.kf_manager.get_keyframes()

        if len(keyframes) == 0:
            return False, None, None

        candidates = keyframes[-self.local_tracking_window:]

        best = None

        for kf in reversed(candidates):
            if kf.descriptors is None:
                continue

            inliers, T_kf_curr = self.estimateRelativePose(
                prev_keypoints=kf.keypoints,
                prev_descriptors=kf.descriptors,
                prev_depth=kf.depth,
                curr_keypoints=curr_keypoints,
                curr_descriptors=curr_descriptors,
            )

            if T_kf_curr is None:
                continue

            if not np.all(np.isfinite(T_kf_curr)):
                continue

            trans = np.linalg.norm(T_kf_curr[:3, 3])
            rot = self.rotation_angle(T_kf_curr[:3, :3])

            if inliers < self.local_tracking_min_inliers:
                continue
            if trans > self.local_tracking_max_translation:
                continue
            if rot > self.local_tracking_max_rotation:
                continue

            score = inliers - 20.0 * trans - 10.0 * rot

            if best is None or score > best["score"]:
                best = {
                    "anchor_id": kf.id,
                    "anchor_pose": kf.pose.copy(),
                    "T_anchor_curr": T_kf_curr.copy(),
                    "inliers": inliers,
                    "trans": trans,
                    "rot": rot,
                    "score": score,
                }

        if best is None:
            return False, None, None

        T_world_camera_new = best["anchor_pose"] @ best["T_anchor_curr"]

        return True, T_world_camera_new, best

    def is_good_vo_motion(self, T_prev_curr, inliers):
        if T_prev_curr is None:
            return False

        if not np.all(np.isfinite(T_prev_curr)):
            return False

        if inliers < self.min_vo_inliers:
            return False

        trans = np.linalg.norm(T_prev_curr[:3, 3])
        rot = self.rotation_angle(T_prev_curr[:3, :3])

        if trans > self.max_vo_translation:
            return False

        if rot > self.max_vo_rotation:
            return False

        return True

    def update_keyframe_poses_from_gtsam(self):
        optimized_poses = self.pose_graph.get_optimized_poses()
        keyframes = self.kf_manager.get_keyframes()

        for kf in keyframes:
            if kf.id in optimized_poses:
                kf.optimized_pose = optimized_poses[kf.id].copy()
            else:
                kf.optimized_pose = kf.pose.copy()

    def try_add_loop_closure(self, keyframes, new_kf):
        best_candidate = None

        searchable_keyframes = keyframes[::self.loop_search_stride]

        for kf in searchable_keyframes:
            if kf.id >= new_kf.id - self.min_loop_gap:
                continue

            inliers, T_old_new = self.estimateRelativePose(
                prev_keypoints=kf.keypoints,
                prev_descriptors=kf.descriptors,
                prev_depth=kf.depth,
                curr_keypoints=new_kf.keypoints,
                curr_descriptors=new_kf.descriptors,
            )

            if T_old_new is None:
                continue

            if inliers < self.min_loop_inliers:
                continue

            T_raw_old_new = np.linalg.inv(kf.pose) @ new_kf.pose
            T_error = np.linalg.inv(T_raw_old_new) @ T_old_new

            trans_error = np.linalg.norm(T_error[:3, 3])
            rot_error = self.rotation_angle(T_error[:3, :3])
            loop_translation = np.linalg.norm(T_old_new[:3, 3])

            self.get_logger().info(
                f"Loop candidate KF {kf.id}->{new_kf.id}: "
                f"inliers={inliers}, "
                f"trans_error={trans_error:.3f}, "
                f"rot_error={rot_error:.3f}, "
                f"loop_translation={loop_translation:.3f}"
            )

            if trans_error > self.max_loop_trans_error:
                continue

            if rot_error > self.max_loop_rot_error:
                continue

            if loop_translation > self.max_loop_translation:
                continue

            score = inliers - 50.0 * trans_error - 30.0 * rot_error

            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "old_id": kf.id,
                    "new_id": new_kf.id,
                    "T_old_new": T_old_new,
                    "inliers": inliers,
                    "trans_error": trans_error,
                    "rot_error": rot_error,
                    "score": score,
                }

        if best_candidate is None:
            return

        self.pose_graph.add_loop_factor(
            old_keyframe_id=best_candidate["old_id"],
            new_keyframe_id=best_candidate["new_id"],
            T_old_new=best_candidate["T_old_new"],
        )

        self.get_logger().info(
            f"Loop closure added: KF {best_candidate['old_id']} -> "
            f"KF {best_candidate['new_id']}, "
            f"inliers={best_candidate['inliers']}, "
            f"trans_error={best_candidate['trans_error']:.3f}, "
            f"rot_error={best_candidate['rot_error']:.3f}"
        )

    def estimateRelativePose(
        self,
        prev_keypoints,
        prev_descriptors,
        prev_depth,
        curr_keypoints,
        curr_descriptors,
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

            if u_prev < 0 or u_prev >= prev_depth.shape[1]:
                continue

            if v_prev < 0 or v_prev >= prev_depth.shape[0]:
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
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
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

    def publish_raw_path(self, stamp):
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

    def publish_corrected_odom(self, stamp):
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

    def publish_optimized_keyframe_path(self):
        keyframes = self.kf_manager.get_keyframes()

        if len(keyframes) == 0:
            return

        stamp = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = "map"

        for kf in keyframes:
            T_world_camera = kf.optimized_pose

            T_world_base = T_world_camera @ self.T_camera_base

            position = T_world_base[:3, 3]
            quat = tf_transformations.quaternion_from_matrix(T_world_base)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = "map"

            pose_msg.pose.position.x = float(position[0])
            pose_msg.pose.position.y = float(position[1])
            pose_msg.pose.position.z = float(position[2])

            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            path_msg.poses.append(pose_msg)

        self.opt_path_pub.publish(path_msg)

    def rotation_angle(self, R):
        cos_angle = (np.trace(R) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def cameraCallBack(self, msg):
        if not self.has_camera_info:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsic = np.array(msg.k, dtype=np.float64).reshape((3, 3))
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