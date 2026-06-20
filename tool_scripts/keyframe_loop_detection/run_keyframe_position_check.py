import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as SciRot

from keyframe_manager import KeyframeManager


IMAGE_TOPIC = "/depth_camera/image"
DEPTH_TOPIC = "/depth_camera/depth_image"
CAMERA_INFO_TOPIC = "/depth_camera/camera_info"
VO_TOPIC = "/rgbd_vo/odom"
GT_TOPIC = "/ground_truth/odom"


@dataclass
class DenseFrame:
    node_id: int
    image_frame_id: int
    timestamp: float
    image: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    gt_pose: np.ndarray | None = None


def odom_to_T(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    R = SciRot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def camera_info_to_K(msg):
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def make_T_from_R(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


def get_T_vo_optical(convention):
    if convention == "none":
        return np.eye(4)

    if convention == "optical_to_base":
        R = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        return make_T_from_R(R)

    raise ValueError(f"Unknown optical convention: {convention}")


def convert_relative_from_optical_to_vo(T_rel_optical, T_vo_optical):
    return T_vo_optical @ T_rel_optical @ T_inv(T_vo_optical)


def rotation_angle_deg_from_T(T):
    R = T[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def relative_pose_error(T_ref, T_est):
    T_err = T_inv(T_ref) @ T_est
    trans_err = np.linalg.norm(T_err[:3, 3])
    rot_err = rotation_angle_deg_from_T(T_err)
    return trans_err, rot_err


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


def load_bag_data(bag_path: Path):
    reader = open_bag_reader(bag_path)
    topic_type_map = get_topic_type_map(reader)

    for topic in [IMAGE_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, VO_TOPIC]:
        if topic not in topic_type_map:
            raise RuntimeError(f"Cannot find required topic: {topic}")

    image_type = get_message(topic_type_map[IMAGE_TOPIC])
    depth_type = get_message(topic_type_map[DEPTH_TOPIC])
    camera_info_type = get_message(topic_type_map[CAMERA_INFO_TOPIC])
    vo_type = get_message(topic_type_map[VO_TOPIC])
    gt_type = get_message(topic_type_map[GT_TOPIC]) if GT_TOPIC in topic_type_map else None

    bridge = CvBridge()

    images = []
    depths = []
    camera_infos = []
    vo_poses = []
    gt_poses = []

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        bag_t = bag_time_ns * 1e-9

        if topic == IMAGE_TOPIC:
            msg = deserialize_message(data, image_type)
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            images.append((bag_t, image))

        elif topic == DEPTH_TOPIC:
            msg = deserialize_message(data, depth_type)
            depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depths.append((bag_t, depth))

        elif topic == CAMERA_INFO_TOPIC:
            msg = deserialize_message(data, camera_info_type)
            K = camera_info_to_K(msg)
            camera_infos.append((bag_t, K))

        elif topic == VO_TOPIC:
            msg = deserialize_message(data, vo_type)
            vo_poses.append((bag_t, odom_to_T(msg)))

        elif topic == GT_TOPIC and gt_type is not None:
            msg = deserialize_message(data, gt_type)
            gt_poses.append((bag_t, odom_to_T(msg)))

    return images, depths, camera_infos, vo_poses, gt_poses


def find_nearest(query_time, data_list, start_idx, max_dt):
    if len(data_list) == 0:
        return None, start_idx

    i = start_idx

    while i + 1 < len(data_list):
        curr_dt = abs(data_list[i][0] - query_time)
        next_dt = abs(data_list[i + 1][0] - query_time)

        if next_dt <= curr_dt:
            i += 1
        else:
            break

    nearest_t, nearest_data = data_list[i]
    dt = abs(nearest_t - query_time)

    if dt > max_dt:
        return None, i

    return (nearest_t, nearest_data, dt), i


def build_dense_frames(
    images,
    depths,
    camera_infos,
    vo_poses,
    gt_poses,
    sync_tol,
    depth_sync_tol,
    gt_sync_tol,
):
    dense_frames = []

    vo_idx = 0
    depth_idx = 0
    K_idx = 0
    gt_idx = 0

    first_K = camera_infos[0][1] if len(camera_infos) > 0 else None
    skipped = 0

    for image_frame_id, (img_t, image) in enumerate(images):
        nearest_vo, vo_idx = find_nearest(img_t, vo_poses, vo_idx, sync_tol)
        if nearest_vo is None:
            skipped += 1
            continue

        nearest_depth, depth_idx = find_nearest(img_t, depths, depth_idx, depth_sync_tol)
        if nearest_depth is None:
            skipped += 1
            continue

        nearest_K, K_idx = find_nearest(img_t, camera_infos, K_idx, sync_tol)
        if nearest_K is None:
            K = first_K
        else:
            _, K, _ = nearest_K

        if K is None:
            skipped += 1
            continue

        nearest_gt, gt_idx = find_nearest(img_t, gt_poses, gt_idx, gt_sync_tol)
        gt_T = nearest_gt[1] if nearest_gt is not None else None

        _, vo_T, _ = nearest_vo
        _, depth, _ = nearest_depth

        dense_frames.append(
            DenseFrame(
                node_id=len(dense_frames),
                image_frame_id=image_frame_id,
                timestamp=img_t,
                image=image,
                depth=depth,
                K=K,
                pose=vo_T,
                gt_pose=gt_T,
            )
        )

    print(f"Synchronized dense frames: {len(dense_frames)}")
    print(f"Skipped images: {skipped}")

    return dense_frames


def select_keyframes_from_dense_frames(dense_frames):
    manager = KeyframeManager()

    for frame in dense_frames:
        before = len(manager.get_keyframes())

        manager.add_keyframe(
            frame_id=frame.image_frame_id,
            pose=frame.pose,
            image=frame.image,
        )

        after = len(manager.get_keyframes())

        if after > before:
            kf = manager.get_keyframes()[-1]
            kf.timestamp = frame.timestamp
            kf.node_id = frame.node_id
            kf.depth = frame.depth
            kf.K = frame.K
            kf.gt_pose = frame.gt_pose

            print(
                f"Insert KF {kf.id:04d}: "
                f"image_frame={kf.frame_id:06d}, "
                f"node_id={kf.node_id:06d}, "
                f"features={len(kf.keypoints)}"
            )

    keyframes = manager.get_keyframes()
    print(f"Selected keyframes: {len(keyframes)}")
    return keyframes


def depth_to_meters(depth_value, depth_dtype):
    if np.issubdtype(depth_dtype, np.integer):
        return float(depth_value) * 0.001
    return float(depth_value)


def backproject_pixel_to_3d(u, v, z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z], dtype=np.float64)


def solve_rgbd_pnp_raw(
    kf_a,
    kf_b,
    ratio_thresh=0.75,
    min_good_matches=40,
    min_valid_depth_points=25,
    min_pnp_inliers=20,
    min_inlier_ratio=0.25,
    max_depth_m=20.0,
    reproj_error_px=4.0,
    save_match_path=None,
):
    if kf_a.descriptors is None or kf_b.descriptors is None:
        return {"success": False, "reason": "missing_descriptors"}

    if kf_a.depth is None or kf_a.K is None:
        return {"success": False, "reason": "missing_depth_or_K"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(kf_a.descriptors, kf_b.descriptors, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < min_good_matches:
        return {
            "success": False,
            "reason": "not_enough_good_matches",
            "good_matches": len(good_matches),
        }

    object_points = []
    image_points = []
    kept_matches = []

    h, w = kf_a.depth.shape[:2]
    K = kf_a.K

    for m in good_matches:
        u_a, v_a = kf_a.keypoints[m.queryIdx].pt
        u_b, v_b = kf_b.keypoints[m.trainIdx].pt

        u_ai = int(round(u_a))
        v_ai = int(round(v_a))

        if u_ai < 0 or u_ai >= w or v_ai < 0 or v_ai >= h:
            continue

        raw_depth = kf_a.depth[v_ai, u_ai]
        z = depth_to_meters(raw_depth, kf_a.depth.dtype)

        if not np.isfinite(z):
            continue

        if z <= 0.05 or z > max_depth_m:
            continue

        object_points.append(backproject_pixel_to_3d(u_a, v_a, z, K))
        image_points.append([u_b, v_b])
        kept_matches.append(m)

    if len(object_points) < min_valid_depth_points:
        return {
            "success": False,
            "reason": "not_enough_valid_depth_points",
            "good_matches": len(good_matches),
            "valid_depth_points": len(object_points),
        }

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=np.zeros((4, 1), dtype=np.float64),
        iterationsCount=200,
        reprojectionError=reproj_error_px,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None:
        return {
            "success": False,
            "reason": "pnp_failed",
            "good_matches": len(good_matches),
            "valid_depth_points": len(object_points),
        }

    inlier_count = len(inliers)
    inlier_ratio = inlier_count / max(1, len(object_points))

    if inlier_count < min_pnp_inliers or inlier_ratio < min_inlier_ratio:
        return {
            "success": False,
            "reason": "weak_pnp",
            "good_matches": len(good_matches),
            "valid_depth_points": len(object_points),
            "pnp_inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
        }

    R_ba, _ = cv2.Rodrigues(rvec)

    T_b_a = np.eye(4)
    T_b_a[:3, :3] = R_ba
    T_b_a[:3, 3] = tvec.reshape(3)

    if save_match_path is not None:
        inlier_set = set(int(idx[0]) for idx in inliers)
        inlier_matches = [
            kept_matches[i]
            for i in range(len(kept_matches))
            if i in inlier_set
        ]

        vis = cv2.drawMatches(
            kf_a.image,
            kf_a.keypoints,
            kf_b.image,
            kf_b.keypoints,
            inlier_matches[:100],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imwrite(str(save_match_path), vis)

    return {
        "success": True,
        "T_b_a": T_b_a,
        "good_matches": len(good_matches),
        "valid_depth_points": len(object_points),
        "pnp_inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
    }


def calculate_keyframe_positions_from_direct_rgbd(
    keyframes,
    use_pnp_inverse,
    optical_convention,
    max_pairs,
    fallback_to_vo,
    match_dir,
):
    if match_dir is not None:
        match_dir.mkdir(parents=True, exist_ok=True)

    T_vo_optical = get_T_vo_optical(optical_convention)

    direct_poses = [keyframes[0].pose.copy()]
    edge_rows = []

    n_pairs = len(keyframes) - 1
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)

    print(f"PnP convention: {'inverse(T_b_a)' if use_pnp_inverse else 'T_b_a direct'}")
    print(f"Optical conversion: {optical_convention}")

    for i in range(n_pairs):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]

        save_match_path = None
        if match_dir is not None:
            save_match_path = (
                match_dir
                / f"kf_{kf_a.id:04d}_to_kf_{kf_b.id:04d}_"
                  f"X{kf_a.node_id:06d}_X{kf_b.node_id:06d}.png"
            )

        pnp_result = solve_rgbd_pnp_raw(
            kf_a=kf_a,
            kf_b=kf_b,
            save_match_path=save_match_path,
        )

        T_vo_rel_raw = T_inv(kf_a.pose) @ kf_b.pose

        if pnp_result is None or not pnp_result["success"]:
            reason = pnp_result["reason"] if pnp_result is not None and "reason" in pnp_result else "unknown"

            print(f"[FAIL] KF {kf_a.id:04d}->{kf_b.id:04d}, reason={reason}")

            if fallback_to_vo:
                T_rel_used = T_vo_rel_raw
                source = "vo_fallback"
                rel_diff_t = 0.0
                rel_diff_r = 0.0
            else:
                break

        else:
            T_b_a = pnp_result["T_b_a"]
            T_rel_optical = T_inv(T_b_a) if use_pnp_inverse else T_b_a
            T_rel_used = convert_relative_from_optical_to_vo(T_rel_optical, T_vo_optical)
            source = "rgbd_pnp_world_converted"
            rel_diff_t, rel_diff_r = relative_pose_error(T_vo_rel_raw, T_rel_used)

        T_next = direct_poses[-1] @ T_rel_used
        direct_poses.append(T_next)

        raw_pose = kf_b.pose
        gt_pose = getattr(kf_b, "gt_pose", None)

        raw_vs_direct = np.linalg.norm(raw_pose[:3, 3] - T_next[:3, 3])

        if gt_pose is not None:
            gt_vs_direct = np.linalg.norm(gt_pose[:3, 3] - T_next[:3, 3])
            gt_vs_raw = np.linalg.norm(gt_pose[:3, 3] - raw_pose[:3, 3])
        else:
            gt_vs_direct = np.nan
            gt_vs_raw = np.nan

        print(
            f"[{source}] KF {kf_a.id:04d}->{kf_b.id:04d} "
            f"rel_diff_t={rel_diff_t:.3f}, "
            f"rel_diff_r={rel_diff_r:.2f}, "
            f"raw_pos_diff={raw_vs_direct:.3f}"
        )

        edge_rows.append({
            "kf_a": kf_a.id,
            "kf_b": kf_b.id,
            "node_a": kf_a.node_id,
            "node_b": kf_b.node_id,
            "source": source,
            "rel_diff_t": rel_diff_t,
            "rel_diff_r_deg": rel_diff_r,
            "raw_vs_direct_t": raw_vs_direct,
            "gt_vs_direct_t": gt_vs_direct,
            "gt_vs_raw_t": gt_vs_raw,
            "direct_x": T_next[0, 3],
            "direct_y": T_next[1, 3],
            "direct_z": T_next[2, 3],
        })

    return direct_poses, edge_rows


def align_to_first(positions):
    if positions is None or len(positions) == 0:
        return positions
    return positions - positions[0]


def positions_from_dense_frames(dense_frames):
    return np.array([frame.pose[:3, 3] for frame in dense_frames])


def gt_positions_from_dense_frames(dense_frames):
    if any(frame.gt_pose is None for frame in dense_frames):
        return None
    return np.array([frame.gt_pose[:3, 3] for frame in dense_frames])


def positions_from_keyframes_raw(keyframes):
    return np.array([kf.pose[:3, 3] for kf in keyframes])


def positions_from_keyframes_gt(keyframes):
    if any(getattr(kf, "gt_pose", None) is None for kf in keyframes):
        return None
    return np.array([kf.gt_pose[:3, 3] for kf in keyframes])


def positions_from_T_list(T_list):
    return np.array([T[:3, 3] for T in T_list])


def plot_keyframe_position_check(dense_frames, keyframes, direct_poses, output_path):
    raw_dense = positions_from_dense_frames(dense_frames)
    gt_dense = gt_positions_from_dense_frames(dense_frames)

    raw_kf = positions_from_keyframes_raw(keyframes)
    gt_kf = positions_from_keyframes_gt(keyframes)

    direct_kf = positions_from_T_list(direct_poses)

    raw_kf_for_direct = raw_kf[:len(direct_kf)]
    gt_kf_for_direct = gt_kf[:len(direct_kf)] if gt_kf is not None else None

    raw_dense_plot = align_to_first(raw_dense)
    raw_kf_plot = align_to_first(raw_kf)
    direct_kf_plot = align_to_first(direct_kf)

    plt.figure(figsize=(9, 9))

    plt.plot(raw_dense_plot[:, 0], raw_dense_plot[:, 1], label="All raw VO frames", linewidth=1.5)
    plt.scatter(raw_kf_plot[:, 0], raw_kf_plot[:, 1], s=20, label="Raw VO keyframes")
    plt.plot(
        direct_kf_plot[:, 0],
        direct_kf_plot[:, 1],
        marker="x",
        linewidth=2,
        label="KF positions accumulated from direct RGB-D PnP",
    )

    if gt_dense is not None:
        gt_dense_plot = align_to_first(gt_dense)
        plt.plot(gt_dense_plot[:, 0], gt_dense_plot[:, 1], label="All GT frames", linewidth=1.5)

    if gt_kf_for_direct is not None:
        gt_kf_for_direct_plot = align_to_first(gt_kf_for_direct)
        plt.scatter(
            gt_kf_for_direct_plot[:, 0],
            gt_kf_for_direct_plot[:, 1],
            s=25,
            marker="^",
            label="GT at direct-RGBD keyframes",
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Keyframe position check: raw VO vs direct RGB-D accumulation")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved position check plot to: {output_path}")


def save_keyframe_position_check(path, keyframes, direct_poses, edge_rows):
    raw_kf = positions_from_keyframes_raw(keyframes)
    gt_kf = positions_from_keyframes_gt(keyframes)
    direct_kf = positions_from_T_list(direct_poses)

    n = len(direct_kf)

    with open(path, "w") as f:
        if gt_kf is None:
            f.write(
                "# kf_id image_frame_id node_id "
                "raw_x raw_y raw_z "
                "direct_x direct_y direct_z "
                "raw_direct_error\n"
            )
        else:
            f.write(
                "# kf_id image_frame_id node_id "
                "raw_x raw_y raw_z "
                "direct_x direct_y direct_z "
                "gt_x gt_y gt_z "
                "raw_direct_error gt_direct_error gt_raw_error\n"
            )

        for i in range(n):
            kf = keyframes[i]
            raw = raw_kf[i]
            direct = direct_kf[i]
            raw_direct_error = np.linalg.norm(raw - direct)

            if gt_kf is None:
                f.write(
                    f"{kf.id} {kf.frame_id} {kf.node_id} "
                    f"{raw[0]:.9f} {raw[1]:.9f} {raw[2]:.9f} "
                    f"{direct[0]:.9f} {direct[1]:.9f} {direct[2]:.9f} "
                    f"{raw_direct_error:.9f}\n"
                )
            else:
                gt = gt_kf[i]
                gt_direct_error = np.linalg.norm(gt - direct)
                gt_raw_error = np.linalg.norm(gt - raw)

                f.write(
                    f"{kf.id} {kf.frame_id} {kf.node_id} "
                    f"{raw[0]:.9f} {raw[1]:.9f} {raw[2]:.9f} "
                    f"{direct[0]:.9f} {direct[1]:.9f} {direct[2]:.9f} "
                    f"{gt[0]:.9f} {gt[1]:.9f} {gt[2]:.9f} "
                    f"{raw_direct_error:.9f} "
                    f"{gt_direct_error:.9f} "
                    f"{gt_raw_error:.9f}\n"
                )

    edge_path = path.with_name(path.stem + "_edges.txt")

    with open(edge_path, "w") as f:
        f.write(
            "# kf_a kf_b node_a node_b source "
            "rel_diff_t rel_diff_r_deg "
            "raw_vs_direct_t gt_vs_direct_t gt_vs_raw_t "
            "direct_x direct_y direct_z\n"
        )

        for row in edge_rows:
            f.write(
                f"{row['kf_a']} {row['kf_b']} "
                f"{row['node_a']} {row['node_b']} "
                f"{row['source']} "
                f"{row['rel_diff_t']:.9f} "
                f"{row['rel_diff_r_deg']:.9f} "
                f"{row['raw_vs_direct_t']:.9f} "
                f"{row['gt_vs_direct_t']:.9f} "
                f"{row['gt_vs_raw_t']:.9f} "
                f"{row['direct_x']:.9f} "
                f"{row['direct_y']:.9f} "
                f"{row['direct_z']:.9f}\n"
            )

    print(f"Saved keyframe position table to: {path}")
    print(f"Saved edge table to: {edge_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", default="keyframe_position_check")

    parser.add_argument("--sync-tol", type=float, default=0.10)
    parser.add_argument("--depth-sync-tol", type=float, default=0.05)
    parser.add_argument("--gt-sync-tol", type=float, default=0.10)

    parser.add_argument("--use-pnp-direct", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--fallback-to-vo", action="store_true")
    parser.add_argument(
        "--optical-convention",
        choices=["optical_to_base", "none"],
        default="optical_to_base",
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

    images, depths, camera_infos, vo_poses, gt_poses = load_bag_data(bag_path)

    print(f"Loaded images: {len(images)}")
    print(f"Loaded depths: {len(depths)}")
    print(f"Loaded camera infos: {len(camera_infos)}")
    print(f"Loaded VO poses: {len(vo_poses)}")
    print(f"Loaded GT poses: {len(gt_poses)}")

    dense_frames = build_dense_frames(
        images=images,
        depths=depths,
        camera_infos=camera_infos,
        vo_poses=vo_poses,
        gt_poses=gt_poses,
        sync_tol=args.sync_tol,
        depth_sync_tol=args.depth_sync_tol,
        gt_sync_tol=args.gt_sync_tol,
    )

    if len(dense_frames) < 2:
        print("Not enough dense frames.")
        return

    keyframes = select_keyframes_from_dense_frames(dense_frames)

    if len(keyframes) < 2:
        print("Not enough keyframes.")
        return

    direct_poses, edge_rows = calculate_keyframe_positions_from_direct_rgbd(
        keyframes=keyframes,
        use_pnp_inverse=not args.use_pnp_direct,
        optical_convention=args.optical_convention,
        max_pairs=args.max_pairs,
        fallback_to_vo=args.fallback_to_vo,
        match_dir=out_dir / "matches",
    )

    plot_path = out_dir / "keyframe_position_check.png"
    plot_keyframe_position_check(
        dense_frames=dense_frames,
        keyframes=keyframes,
        direct_poses=direct_poses,
        output_path=plot_path,
    )

    table_path = out_dir / "keyframe_position_check.txt"
    save_keyframe_position_check(
        path=table_path,
        keyframes=keyframes,
        direct_poses=direct_poses,
        edge_rows=edge_rows,
    )

    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()