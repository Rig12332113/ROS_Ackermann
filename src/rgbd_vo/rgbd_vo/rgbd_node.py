import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2

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
        self.orb = cv2.ORB_create(nfeatures=1000)

    def rgbdCallBack(self, rgb_msg, depth_msg):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        print(f"Number of keypoints: {len(keypoints)}")

        debug_img = cv2.drawKeypoints(
            rgb_image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        cv2.imshow("ORB keypoints", debug_img)
        cv2.waitKey(1)

    def cameraCallBack(self, msg):
        print("camera_info")
        intrinsic = msg.k
        print(f"fx = {intrinsic[0]},  \
                fy = {intrinsic[4]},  \
                cx = {intrinsic[2]},  \
                cy = {intrinsic[5]}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBD_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == "__main__":
    main()