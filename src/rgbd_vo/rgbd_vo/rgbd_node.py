import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np

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
            CameraInfo, "depth_camera/camera_info", self.cameraCallBack, 10
        )
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
        self.prev_rgb = None

    def rgbdCallBack(self, rgb_msg, depth_msg):
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

        print(f"Original keypoints: {len(keypoints)}")
        print(f"Valid-depth keypoints: {len(valid_keypoints)}")

        # debug_img = cv2.drawKeypoints(
        #     rgb_image,
        #     valid_keypoints,
        #     None,
        #     flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        # )

        # cv2.imshow("ORB keypoints", debug_img)
        # cv2.waitKey(1)

        if self.prev_descriptors is not None and valid_descriptors is not None:
            matches = self.matcher.match(self.prev_descriptors, valid_descriptors)

            # smaller distance means better descriptor match
            matches = sorted(matches, key=lambda m: m.distance)

            good_matches = matches[:50]

            print(f"Raw matches: {len(matches)}")
            print(f"Good matches: {len(good_matches)}")

            match_img = cv2.drawMatches(
                self.prev_rgb,
                self.prev_keypoints,
                rgb_image,
                valid_keypoints,
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            cv2.imshow("ORB matches", match_img)
            cv2.waitKey(1)
        
        self.prev_keypoints = valid_keypoints
        self.prev_descriptors = valid_descriptors
        self.prev_depth = depth_image
        self.prev_rgb = rgb_image

    def cameraCallBack(self, msg):
        # print("camera_info")
        intrinsic = msg.k
        # print(f"fx = {intrinsic[0]},  \
        #         fy = {intrinsic[4]},  \
        #         cx = {intrinsic[2]},  \
        #         cy = {intrinsic[5]}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBD_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == "__main__":
    main()