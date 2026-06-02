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
            slop=0.03  # maximum allowed timestamp difference
        )

        self.ts.registerCallback(self.rgbdCallBack)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/depth_camera/camera_info", self.cameraCallBack, 10
        )

        self.path_pub = self.create_publisher(
            Path, "/rgbd_vo/path", 10
        )
        self.odom_pub = self.create_publisher(
            Odometry, "/rgbd_vo/odom", 10
        )
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.has_camera_info = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.intrinsic = None

        self.num_match = 100 # change this to param later
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

        # Since GT starts near world == base_link, initialize camera pose from base pose.
        self.T_world_camera = self.T_base_camera.copy()


    def rgbdCallBack(self, rgb_msg, depth_msg):
        if not self.has_camera_info:
            print("Waiting for camera_info...")
            return

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # pick valid feature
        valid_keypoints = []
        valid_descriptors = []

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
            matches = self.matcher.match(self.prev_descriptors, valid_descriptors)

            # smaller distance means better descriptor match
            matches = sorted(matches, key=lambda m: m.distance)

            good_matches = matches[:self.num_match]

            # print(f"Raw matches: {len(matches)}")
            # print(f"Good matches: {len(good_matches)}")

            match_img = cv2.drawMatches(
                self.prev_rgb,
                self.prev_keypoints,
                rgb_image,
                valid_keypoints,
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # cv2.imshow("ORB matches", match_img)
            # cv2.waitKey(1)
        

            # use previous 3D points and current 2D points to save PnP question
            object_points = []
            image_points = []
            for match in good_matches:
                prev_kp = self.prev_keypoints[match.queryIdx]
                u_prev, v_prev = prev_kp.pt
                u_prev = int(u_prev)
                v_prev = int(v_prev)

                curr_kp = valid_keypoints[match.trainIdx]
                u_curr, v_curr = curr_kp.pt

                Z = self.prev_depth[v_prev, u_prev]
                X = (u_prev - self.cx) * Z / self.fx
                Y = (v_prev - self.cy) * Z / self.fy
                object_points.append([X, Y, Z])

                image_points.append([u_curr, v_curr])

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            # run PnP solver 
            success, rvec, t, inlier = cv2.solvePnPRansac(
                object_points, image_points, self.intrinsic, np.zeros((5,1)),
            ) 
            R, _ = cv2.Rodrigues(rvec)

            T_curr_prev = np.eye(4)
            T_curr_prev[:3, :3] = R
            T_curr_prev[:3, 3] = t.ravel()

            T_prev_curr = np.linalg.inv(T_curr_prev)

            # Update accumulated camera pose in world
            self.T_world_camera = self.T_world_camera @ T_prev_curr

            T_world_base = self.T_world_camera @ self.T_camera_base
            position = T_world_base[:3, 3]
            quat = tf_transformations.quaternion_from_matrix(T_world_base)

            # Initialize the message and publish
            pose_msg = PoseStamped()

            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"

            pose_msg.pose.position.x = float(position[0])
            pose_msg.pose.position.y = float(position[1])
            pose_msg.pose.position.z = float(position[2])

            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            self.path_msg.header.stamp = pose_msg.header.stamp
            self.path_msg.poses.append(pose_msg)

            self.path_pub.publish(self.path_msg)

            odom_msg = Odometry()

            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "base_link"

            odom_msg.pose.pose.position.x = float(position[0])
            odom_msg.pose.pose.position.y = float(position[1])
            odom_msg.pose.pose.position.z = float(position[2])

            odom_msg.pose.pose.orientation.x = float(quat[0])
            odom_msg.pose.pose.orientation.y = float(quat[1])
            odom_msg.pose.pose.orientation.z = float(quat[2])
            odom_msg.pose.pose.orientation.w = float(quat[3])

            self.odom_pub.publish(odom_msg)

        # update frame
        self.prev_keypoints = valid_keypoints
        self.prev_descriptors = valid_descriptors
        self.prev_depth = depth_image
        self.prev_rgb = rgb_image

    def cameraCallBack(self, msg):
        if (not self.has_camera_info):
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsic = msg.k.reshape((3,3))
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