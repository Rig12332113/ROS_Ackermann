import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R

from keyframe_manager import KeyframeManager


IMAGE_TOPIC = "/depth_camera/image"
VO_TOPIC = "/rgbd_vo/odom"


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def odom_to_T(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation

    quat_xyzw = [q.x, q.y, q.z, q.w]
    rot = R.from_quat(quat_xyzw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def open_bag_reader(bag_path: Path):
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id="mcap",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(reader):
    topic_types = reader.get_all_topics_and_types()
    return {topic.name: topic.type for topic in topic_types}


def load_bag_data(bag_path: Path, use_bag_time: bool = True):
    reader = open_bag_reader(bag_path)
    topic_type_map = get_topic_type_map(reader)

    print("Topics in bag:")
    for topic_name, topic_type in topic_type_map.items():
        print(f"  {topic_name}: {topic_type}")

    if IMAGE_TOPIC not in topic_type_map:
        raise RuntimeError(f"Cannot find image topic: {IMAGE_TOPIC}")

    if VO_TOPIC not in topic_type_map:
        raise RuntimeError(f"Cannot find VO topic: {VO_TOPIC}")

    image_msg_type = get_message(topic_type_map[IMAGE_TOPIC])
    odom_msg_type = get_message(topic_type_map[VO_TOPIC])

    bridge = CvBridge()

    images = []
    vo_poses = []

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        bag_t = bag_time_ns * 1e-9

        if topic == IMAGE_TOPIC:
            msg = deserialize_message(data, image_msg_type)

            if use_bag_time:
                t = bag_t
            else:
                t = stamp_to_sec(msg.header.stamp)

            image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            images.append((t, image))

        elif topic == VO_TOPIC:
            msg = deserialize_message(data, odom_msg_type)

            if use_bag_time:
                t = bag_t
            else:
                t = stamp_to_sec(msg.header.stamp)

            T = odom_to_T(msg)
            vo_poses.append((t, T))

    return images, vo_poses


def print_time_debug(images, vo_poses):
    print("")
    print("Timestamp debug")
    print("---------------")

    if len(images) == 0:
        print("No images loaded.")
    else:
        print(f"Image time range: {images[0][0]:.9f} -> {images[-1][0]:.9f}")
        print("First image times:")
        for t, _ in images[:5]:
            print(f"  {t:.9f}")

    if len(vo_poses) == 0:
        print("No VO poses loaded.")
    else:
        print(f"VO time range:    {vo_poses[0][0]:.9f} -> {vo_poses[-1][0]:.9f}")
        print("First VO times:")
        for t, _ in vo_poses[:5]:
            print(f"  {t:.9f}")

    if len(images) > 0 and len(vo_poses) > 0:
        print("")
        print(f"First image - first VO dt: {images[0][0] - vo_poses[0][0]:.9f} s")
        print(f"Last image  - last VO dt:  {images[-1][0] - vo_poses[-1][0]:.9f} s")


def find_nearest_pose(image_time, vo_poses, start_idx, max_dt):
    if len(vo_poses) == 0:
        return None, start_idx

    i = start_idx

    while i + 1 < len(vo_poses):
        curr_dt = abs(vo_poses[i][0] - image_time)
        next_dt = abs(vo_poses[i + 1][0] - image_time)

        if next_dt <= curr_dt:
            i += 1
        else:
            break

    nearest_t, nearest_T = vo_poses[i]
    dt = abs(nearest_t - image_time)

    if dt > max_dt:
        return None, i

    return (nearest_t, nearest_T, dt), i


def save_keyframe_images(keyframes, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for kf in keyframes:
        filename = output_dir / f"kf_{kf.id:04d}_frame_{kf.frame_id:06d}.png"
        cv2.imwrite(str(filename), kf.image)


def plot_trajectory(all_poses, keyframes, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip trajectory plot.")
        return

    if len(all_poses) == 0:
        print("No synchronized poses; skip trajectory plot.")
        return

    traj = np.array([T[:3, 3] for T in all_poses])

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], linewidth=1, label="VO trajectory")

    if len(keyframes) > 0:
        kf_positions = np.array([kf.pose[:3, 3] for kf in keyframes])
        plt.scatter(
            kf_positions[:, 0],
            kf_positions[:, 1],
            s=20,
            label="Keyframes",
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("VO trajectory with selected keyframes")
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_keyframe_table(keyframes, output_path: Path):
    with open(output_path, "w") as f:
        f.write("# kf_id frame_id timestamp tx ty tz num_keypoints\n")

        for kf in keyframes:
            t = kf.timestamp
            p = kf.pose[:3, 3]
            n_kp = len(kf.keypoints) if kf.keypoints is not None else 0

            f.write(
                f"{kf.id} "
                f"{kf.frame_id} "
                f"{t:.9f} "
                f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"{n_kp}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to rosbag folder, the folder containing metadata.yaml",
    )
    parser.add_argument(
        "--out",
        default="keyframe_debug_output",
        help="Output folder for debug images and plots",
    )
    parser.add_argument(
        "--sync-tol",
        type=float,
        default=0.10,
        help="Maximum allowed image-VO timestamp difference in seconds",
    )
    parser.add_argument(
        "--use-header-time",
        action="store_true",
        help="Use msg.header.stamp instead of bag receive time",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.out)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag folder does not exist: {bag_path}")

    if not (bag_path / "metadata.yaml").exists():
        raise FileNotFoundError(
            f"metadata.yaml not found in {bag_path}. "
            "Pass the rosbag folder, not the .mcap file."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    use_bag_time = not args.use_header_time

    print(f"Loading bag: {bag_path}")
    print(f"Using time source: {'bag receive time' if use_bag_time else 'message header stamp'}")
    print(f"Sync tolerance: {args.sync_tol:.3f} s")

    images, vo_poses = load_bag_data(
        bag_path=bag_path,
        use_bag_time=use_bag_time,
    )

    print("")
    print(f"Loaded images:   {len(images)}")
    print(f"Loaded VO poses: {len(vo_poses)}")

    print_time_debug(images, vo_poses)

    manager = KeyframeManager()

    synced_count = 0
    skipped_count = 0
    max_seen_dt = 0.0
    all_synced_poses = []

    vo_idx = 0

    for frame_id, (img_t, image) in enumerate(images):
        nearest, vo_idx = find_nearest_pose(
            image_time=img_t,
            vo_poses=vo_poses,
            start_idx=vo_idx,
            max_dt=args.sync_tol,
        )

        if nearest is None:
            skipped_count += 1
            continue

        vo_t, pose, dt = nearest
        max_seen_dt = max(max_seen_dt, dt)

        synced_count += 1
        all_synced_poses.append(pose)

        before_count = len(manager.get_keyframes())

        manager.add_keyframe(
            frame_id=frame_id,
            pose=pose,
            image=image,
        )

        after_count = len(manager.get_keyframes())

        if after_count > before_count:
            kf = manager.get_keyframes()[-1]

            # Your current KeyframeManager stores timestamp=0.0.
            # Patch it here for debug output.
            kf.timestamp = img_t

            print(
                f"Insert KF {kf.id:04d}: "
                f"frame={kf.frame_id:06d}, "
                f"img_t={img_t:.6f}, "
                f"vo_t={vo_t:.6f}, "
                f"dt={dt:.4f}, "
                f"features={len(kf.keypoints)}"
            )

    keyframes = manager.get_keyframes()

    print("")
    print("Summary")
    print("-------")
    print(f"Synchronized frames: {synced_count}")
    print(f"Skipped images:       {skipped_count}")
    print(f"Max accepted dt:      {max_seen_dt:.6f} s")
    print(f"Selected keyframes:   {len(keyframes)}")

    kf_img_dir = out_dir / "keyframes"
    save_keyframe_images(keyframes, kf_img_dir)

    traj_path = out_dir / "trajectory_keyframes.png"
    plot_trajectory(
        all_poses=all_synced_poses,
        keyframes=keyframes,
        output_path=traj_path,
    )

    table_path = out_dir / "keyframes.txt"
    save_keyframe_table(keyframes, table_path)

    print("")
    print(f"Saved keyframe images to: {kf_img_dir}")
    print(f"Saved trajectory plot to: {traj_path}")
    print(f"Saved keyframe table to: {table_path}")


if __name__ == "__main__":
    main()