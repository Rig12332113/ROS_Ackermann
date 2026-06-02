#!/usr/bin/env python3

import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import gtsam


VO_TOPIC = "/rgbd_vo/odom"
GT_TOPIC = "/ground_truth/odom"

def read_bag_messages(bag_dir: str, target_topics: set[str]):
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_dir,
        storage_id=""   # let ROS infer mcap/sqlite3 from metadata.yaml
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()

    type_map = {}
    for topic_type in topic_types:
        if topic_type.name in target_topics:
            type_map[topic_type.name] = get_message(topic_type.type)

    print("Found target topics:")
    for name, msg_class in type_map.items():
        print(f"  {name}: {msg_class}")

    if VO_TOPIC not in type_map:
        raise RuntimeError(f"Missing topic in bag: {VO_TOPIC}")

    if GT_TOPIC not in type_map:
        raise RuntimeError(f"Missing topic in bag: {GT_TOPIC}")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic not in target_topics:
            continue

        msg_class = type_map[topic]
        msg = deserialize_message(data, msg_class)

        yield topic, timestamp, msg
        
def quat_to_rot3(qx: float, qy: float, qz: float, qw: float) -> gtsam.Rot3:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

    if norm < 1e-12:
        return gtsam.Rot3()

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return gtsam.Rot3.Quaternion(qw, qx, qy, qz)


def odom_msg_to_pose3(msg) -> gtsam.Pose3:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation

    rot = quat_to_rot3(q.x, q.y, q.z, q.w)
    trans = gtsam.Point3(p.x, p.y, p.z)

    return gtsam.Pose3(rot, trans)


def load_poses_from_bag(bag_dir: str):
    rclpy.init(args=None)

    vo_poses = []
    gt_poses = []

    for topic, timestamp, msg in read_bag_messages(
        bag_dir,
        {VO_TOPIC, GT_TOPIC},
    ):
        pose = odom_msg_to_pose3(msg)

        if topic == VO_TOPIC:
            vo_poses.append((timestamp, pose))
        elif topic == GT_TOPIC:
            gt_poses.append((timestamp, pose))

    rclpy.shutdown()

    print()
    print(f"Loaded VO poses: {len(vo_poses)}")
    print(f"Loaded GT poses: {len(gt_poses)}")

    if len(vo_poses) < 2:
        raise RuntimeError("Need at least 2 VO poses.")

    if len(gt_poses) < 2:
        print("Warning: GT has fewer than 2 poses. Plot may be incomplete.")

    return vo_poses, gt_poses


def normalize_pose_sequence(timestamped_poses):
    if len(timestamped_poses) == 0:
        return []

    first_pose = timestamped_poses[0][1]
    first_inv = first_pose.inverse()

    normalized = []

    for timestamp, pose in timestamped_poses:
        normalized_pose = first_inv.compose(pose)
        normalized.append((timestamp, normalized_pose))

    return normalized


def build_gtsam_graph_from_vo(vo_poses):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    vo_norm = normalize_pose_sequence(vo_poses)

    # GTSAM Pose3 noise order is approximately:
    # rotation vector xyz, translation xyz
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    )

    # First simple setting. We will tune this later.
    vo_between_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.20, 0.20, 0.20, 0.30, 0.30, 0.30])
    )

    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
    )

    # Insert initial guesses.
    for i, (_, pose) in enumerate(vo_norm):
        key = gtsam.symbol("x", i)
        initial.insert(key, pose)

    # Fix first pose at identity.
    graph.add(
        gtsam.PriorFactorPose3(
            gtsam.symbol("x", 0),
            gtsam.Pose3(),
            prior_noise,
        )
    )

    # Add between factors from consecutive VO poses.
    for i in range(len(vo_norm) - 1):
        pose_i = vo_norm[i][1]
        pose_j = vo_norm[i + 1][1]

        relative_ij = pose_i.between(pose_j)

        graph.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", i),
                gtsam.symbol("x", i + 1),
                relative_ij,
                vo_between_noise,
            )
        )
    
    last_idx = len(vo_norm) - 1

    graph.add(
        gtsam.BetweenFactorPose3(
            gtsam.symbol("x", 0),
            gtsam.symbol("x", last_idx),
            gtsam.Pose3(),
            loop_noise,
        )
    )
    print()
    print(f"GTSAM variables: {len(vo_norm)}")
    print(f"GTSAM factors: {graph.size()}")

    return graph, initial


def optimize_graph(graph, initial):
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    print()
    print("Optimizing...")
    result = optimizer.optimize()

    print("Done.")
    return result


def result_to_pose_list(result, n):
    poses = []

    for i in range(n):
        key = gtsam.symbol("x", i)
        poses.append(result.atPose3(key))

    return poses


def pose3_xy(pose: gtsam.Pose3):
    t = pose.translation()
    return float(t[0]), float(t[1])


def timestamped_pose_xy_array(timestamped_poses):
    xy = []

    for _, pose in timestamped_poses:
        xy.append(pose3_xy(pose))

    return np.array(xy)


def pose_list_xy_array(poses):
    xy = []

    for pose in poses:
        xy.append(pose3_xy(pose))

    return np.array(xy)


def plot_results(vo_poses, gt_poses, optimized_poses):
    vo_norm = normalize_pose_sequence(vo_poses)
    gt_norm = normalize_pose_sequence(gt_poses)

    raw_vo_xy = timestamped_pose_xy_array(vo_norm)
    gt_xy = timestamped_pose_xy_array(gt_norm)
    opt_xy = pose_list_xy_array(optimized_poses)

    plt.figure()

    if len(gt_xy) > 0:
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground Truth")

    if len(raw_vo_xy) > 0:
        plt.plot(raw_vo_xy[:, 0], raw_vo_xy[:, 1], label="Raw VO")

    if len(opt_xy) > 0:
        plt.plot(opt_xy[:, 0], opt_xy[:, 1], label="GTSAM Optimized")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Ground Truth vs Raw VO vs GTSAM")
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python offline_gtsam_from_bag.py <bag_folder>")
        sys.exit(1)

    bag_dir = sys.argv[1]

    vo_poses, gt_poses = load_poses_from_bag(bag_dir)
    vo_poses = vo_poses[::10]

    graph, initial = build_gtsam_graph_from_vo(vo_poses)
    result = optimize_graph(graph, initial)

    optimized_poses = result_to_pose_list(result, len(vo_poses))

    plot_results(vo_poses, gt_poses, optimized_poses)


if __name__ == "__main__":
    main()
