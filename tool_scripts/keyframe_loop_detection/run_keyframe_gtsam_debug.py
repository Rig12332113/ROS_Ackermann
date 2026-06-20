import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from gtsam.symbol_shorthand import X
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


def T_to_pose3(T):
    return gtsam.Pose3(
        gtsam.Rot3(T[:3, :3]),
        gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
    )


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
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


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
        K = first_K if nearest_K is None else nearest_K[1]

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

    return np.array([
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
    ], dtype=np.float64)


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


def get_rgbd_relative_measurement(
    kf_a,
    kf_b,
    use_pnp_inverse,
    T_vo_optical,
    save_match_path=None,
):
    result = solve_rgbd_pnp_raw(
        kf_a=kf_a,
        kf_b=kf_b,
        save_match_path=save_match_path,
    )

    if result is None or not result["success"]:
        return result

    T_b_a = result["T_b_a"]
    T_rel_optical = T_inv(T_b_a) if use_pnp_inverse else T_b_a
    T_rel_vo = convert_relative_from_optical_to_vo(T_rel_optical, T_vo_optical)

    result["T_measurement"] = T_rel_vo
    return result


def should_accept_rgbd_factor(
    kf_a,
    kf_b,
    T_measurement,
    max_translation_disagreement,
    max_rotation_disagreement_deg,
):
    T_vo = T_inv(kf_a.pose) @ kf_b.pose
    trans_diff, rot_diff = relative_pose_error(T_vo, T_measurement)

    accepted = (
        trans_diff <= max_translation_disagreement
        and rot_diff <= max_rotation_disagreement_deg
    )

    stats = {
        "vo_trans": np.linalg.norm(T_vo[:3, 3]),
        "meas_trans": np.linalg.norm(T_measurement[:3, 3]),
        "vo_rot_deg": rotation_angle_deg_from_T(T_vo),
        "meas_rot_deg": rotation_angle_deg_from_T(T_measurement),
        "trans_diff": trans_diff,
        "rot_diff_deg": rot_diff,
    }

    return accepted, stats


def add_head_tail_loop_factor(
    graph,
    keyframes,
    loop_noise,
    same_position_only=True,
):
    if len(keyframes) < 2:
        return 0

    kf_a = keyframes[0]
    kf_b = keyframes[-1]

    if same_position_only:
        loop_measurement = gtsam.Pose3()
    else:
        loop_measurement = T_to_pose3(T_inv(kf_a.pose) @ kf_b.pose)

    graph.add(
        gtsam.BetweenFactorPose3(
            X(kf_a.node_id),
            X(kf_b.node_id),
            loop_measurement,
            loop_noise,
        )
    )

    print(
        f"[ADD LOOP] KF {kf_a.id:04d}->{kf_b.id:04d} "
        f"X{kf_a.node_id}->X{kf_b.node_id}"
    )

    return 1


def build_dense_graph(
    dense_frames,
    keyframes,
    add_rgbd_factors,
    single_kf_factor,
    max_kf_factors,
    use_pnp_inverse,
    optical_convention,
    rgbd_rot_sigma,
    rgbd_trans_sigma,
    max_translation_disagreement,
    max_rotation_disagreement_deg,
    add_head_tail_loop,
    match_dir,
):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    )

    vo_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10])
    )

    rgbd_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([
            rgbd_rot_sigma,
            rgbd_rot_sigma,
            rgbd_rot_sigma,
            rgbd_trans_sigma,
            rgbd_trans_sigma,
            rgbd_trans_sigma,
        ])
    )

    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([10.0, 10.0, 10.0, 0.10, 0.10, 0.10])
    )

    for frame in dense_frames:
        initial.insert(X(frame.node_id), T_to_pose3(frame.pose))

    graph.add(
        gtsam.PriorFactorPose3(
            X(dense_frames[0].node_id),
            T_to_pose3(dense_frames[0].pose),
            prior_noise,
        )
    )

    for i in range(len(dense_frames) - 1):
        a = dense_frames[i]
        b = dense_frames[i + 1]
        T_a_b = T_inv(a.pose) @ b.pose

        graph.add(
            gtsam.BetweenFactorPose3(
                X(a.node_id),
                X(b.node_id),
                T_to_pose3(T_a_b),
                vo_noise,
            )
        )

    added = 0
    rejected = 0
    failed = 0
    loop_count = 0

    if match_dir is not None:
        match_dir.mkdir(parents=True, exist_ok=True)

    T_vo_optical = get_T_vo_optical(optical_convention)

    if add_rgbd_factors:
        if single_kf_factor is not None:
            pair_indices = [single_kf_factor]
        else:
            pair_indices = list(range(len(keyframes) - 1))

        print("Adding RGB-D keyframe factors")
        print(f"PnP convention: {'inverse(T_b_a)' if use_pnp_inverse else 'T_b_a direct'}")
        print(f"Optical conversion: {optical_convention}")

        for pair_i in pair_indices:
            if pair_i < 0 or pair_i >= len(keyframes) - 1:
                print(f"[SKIP] invalid pair index {pair_i}")
                continue

            if max_kf_factors is not None and added >= max_kf_factors:
                break

            kf_a = keyframes[pair_i]
            kf_b = keyframes[pair_i + 1]

            save_match_path = None
            if match_dir is not None:
                save_match_path = (
                    match_dir
                    / f"kf_{kf_a.id:04d}_to_kf_{kf_b.id:04d}_"
                      f"X{kf_a.node_id:06d}_X{kf_b.node_id:06d}.png"
                )

            result = get_rgbd_relative_measurement(
                kf_a=kf_a,
                kf_b=kf_b,
                use_pnp_inverse=use_pnp_inverse,
                T_vo_optical=T_vo_optical,
                save_match_path=save_match_path,
            )

            if result is None or not result["success"]:
                failed += 1
                reason = result["reason"] if result is not None and "reason" in result else "unknown"
                print(
                    f"[RGBD FAIL] KF {kf_a.id:04d}->{kf_b.id:04d} "
                    f"X{kf_a.node_id}->X{kf_b.node_id}, reason={reason}"
                )
                continue

            T_measurement = result["T_measurement"]

            accepted, stats = should_accept_rgbd_factor(
                kf_a=kf_a,
                kf_b=kf_b,
                T_measurement=T_measurement,
                max_translation_disagreement=max_translation_disagreement,
                max_rotation_disagreement_deg=max_rotation_disagreement_deg,
            )

            print(
                f"[RGBD TEST] KF {kf_a.id:04d}->{kf_b.id:04d} "
                f"X{kf_a.node_id}->X{kf_b.node_id} | "
                f"inliers={result['pnp_inliers']}, "
                f"ratio={result['inlier_ratio']:.3f}, "
                f"vo_t={stats['vo_trans']:.3f}, "
                f"meas_t={stats['meas_trans']:.3f}, "
                f"diff_t={stats['trans_diff']:.3f}, "
                f"diff_r={stats['rot_diff_deg']:.2f}, "
                f"accepted={accepted}"
            )

            if not accepted:
                rejected += 1
                continue

            graph.add(
                gtsam.BetweenFactorPose3(
                    X(kf_a.node_id),
                    X(kf_b.node_id),
                    T_to_pose3(T_measurement),
                    rgbd_noise,
                )
            )

            added += 1

    if add_head_tail_loop:
        loop_count = add_head_tail_loop_factor(
            graph=graph,
            keyframes=keyframes,
            loop_noise=loop_noise,
            same_position_only=True,
        )

    return graph, initial, added, rejected, failed, loop_count


def optimize_graph(graph, initial):
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(100)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    return optimizer.optimize()


def positions_from_dense_frames(dense_frames):
    return np.array([frame.pose[:3, 3] for frame in dense_frames])


def positions_from_result(dense_frames, result):
    positions = []

    for frame in dense_frames:
        pose = result.atPose3(X(frame.node_id))
        t = pose.translation()
        positions.append([t[0], t[1], t[2]])

    return np.array(positions)


def gt_positions_from_dense_frames(dense_frames):
    if any(frame.gt_pose is None for frame in dense_frames):
        return None
    return np.array([frame.gt_pose[:3, 3] for frame in dense_frames])


def align_to_first(positions):
    if positions is None or len(positions) == 0:
        return positions
    return positions - positions[0]


def compute_errors(pred, gt):
    if pred is None or gt is None or len(pred) != len(gt):
        return None
    return np.linalg.norm(align_to_first(pred) - align_to_first(gt), axis=1)


def print_error_summary(name, errors):
    if errors is None:
        print(f"{name}: no GT comparison available")
        return

    print(f"{name}:")
    print(f"  mean error:   {np.mean(errors):.6f}")
    print(f"  median error: {np.median(errors):.6f}")
    print(f"  max error:    {np.max(errors):.6f}")
    print(f"  final error:  {errors[-1]:.6f}")


def plot_trajectories(dense_frames, keyframes, raw, opt, gt, output_path):
    raw_plot = align_to_first(raw)
    opt_plot = align_to_first(opt)

    plt.figure(figsize=(9, 9))
    plt.plot(raw_plot[:, 0], raw_plot[:, 1], label="Raw dense VO", linewidth=2)
    plt.plot(opt_plot[:, 0], opt_plot[:, 1], label="Optimized dense graph", linewidth=2)

    if gt is not None and len(gt) == len(raw):
        gt_plot = align_to_first(gt)
        plt.plot(gt_plot[:, 0], gt_plot[:, 1], label="Ground truth", linewidth=2)

    kf_nodes = [kf.node_id for kf in keyframes]
    plt.scatter(raw_plot[kf_nodes, 0], raw_plot[kf_nodes, 1], s=18, label="Raw keyframes")
    plt.scatter(opt_plot[kf_nodes, 0], opt_plot[kf_nodes, 1], s=18, marker="x", label="Optimized keyframes")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dense graph optimization with RGB-D keyframe factors")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_dense_positions(path, dense_frames, raw, opt, gt):
    with open(path, "w") as f:
        if gt is None:
            f.write("# node_id image_frame_id timestamp raw_x raw_y raw_z opt_x opt_y opt_z\n")
        else:
            f.write("# node_id image_frame_id timestamp raw_x raw_y raw_z opt_x opt_y opt_z gt_x gt_y gt_z\n")

        for i, frame in enumerate(dense_frames):
            r = raw[i]
            o = opt[i]

            if gt is None:
                f.write(
                    f"{frame.node_id} {frame.image_frame_id} {frame.timestamp:.9f} "
                    f"{r[0]:.9f} {r[1]:.9f} {r[2]:.9f} "
                    f"{o[0]:.9f} {o[1]:.9f} {o[2]:.9f}\n"
                )
            else:
                g = gt[i]
                f.write(
                    f"{frame.node_id} {frame.image_frame_id} {frame.timestamp:.9f} "
                    f"{r[0]:.9f} {r[1]:.9f} {r[2]:.9f} "
                    f"{o[0]:.9f} {o[1]:.9f} {o[2]:.9f} "
                    f"{g[0]:.9f} {g[1]:.9f} {g[2]:.9f}\n"
                )


def save_keyframe_table(path, keyframes):
    with open(path, "w") as f:
        f.write("# kf_id image_frame_id dense_node_id timestamp num_keypoints\n")
        for kf in keyframes:
            f.write(
                f"{kf.id} {kf.frame_id} {kf.node_id} {kf.timestamp:.9f} "
                f"{len(kf.keypoints)}\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", default="dense_rgbd_optimization")

    parser.add_argument("--sync-tol", type=float, default=0.10)
    parser.add_argument("--depth-sync-tol", type=float, default=0.05)
    parser.add_argument("--gt-sync-tol", type=float, default=0.10)

    parser.add_argument("--no-rgbd-factors", action="store_true")
    parser.add_argument("--single-kf-factor", type=int, default=None)
    parser.add_argument("--max-kf-factors", type=int, default=None)

    parser.add_argument("--use-pnp-direct", action="store_true")
    parser.add_argument(
        "--optical-convention",
        choices=["optical_to_base", "none"],
        default="optical_to_base",
    )

    parser.add_argument("--rgbd-rot-sigma", type=float, default=0.30)
    parser.add_argument("--rgbd-trans-sigma", type=float, default=0.50)
    parser.add_argument("--max-translation-disagreement", type=float, default=0.35)
    parser.add_argument("--max-rotation-disagreement-deg", type=float, default=20.0)

    parser.add_argument("--head-tail-loop", action="store_true")

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

    graph, initial, added, rejected, failed, loop_count = build_dense_graph(
        dense_frames=dense_frames,
        keyframes=keyframes,
        add_rgbd_factors=not args.no_rgbd_factors,
        single_kf_factor=args.single_kf_factor,
        max_kf_factors=args.max_kf_factors,
        use_pnp_inverse=not args.use_pnp_direct,
        optical_convention=args.optical_convention,
        rgbd_rot_sigma=args.rgbd_rot_sigma,
        rgbd_trans_sigma=args.rgbd_trans_sigma,
        max_translation_disagreement=args.max_translation_disagreement,
        max_rotation_disagreement_deg=args.max_rotation_disagreement_deg,
        add_head_tail_loop=args.head_tail_loop,
        match_dir=out_dir / "matches",
    )

    print("")
    print("GTSAM graph")
    print("-----------")
    print(f"Dense variables:        {initial.size()}")
    print(f"Total factors:          {graph.size()}")
    print(f"Dense VO factors:       {len(dense_frames) - 1}")
    print(f"RGB-D factors added:    {added}")
    print(f"RGB-D factors rejected: {rejected}")
    print(f"RGB-D factors failed:   {failed}")
    print(f"Head-tail loop factors: {loop_count}")
    print(f"PnP convention:         {'inverse(T_b_a)' if not args.use_pnp_direct else 'T_b_a direct'}")
    print(f"Optical convention:     {args.optical_convention}")
    print(f"RGB-D noise:            rot={args.rgbd_rot_sigma}, trans={args.rgbd_trans_sigma}")

    result = optimize_graph(graph, initial)

    raw_positions = positions_from_dense_frames(dense_frames)
    opt_positions = positions_from_result(dense_frames, result)
    gt_positions = gt_positions_from_dense_frames(dense_frames)

    raw_errors = compute_errors(raw_positions, gt_positions)
    opt_errors = compute_errors(opt_positions, gt_positions)

    print("")
    print("GT comparison, after first-pose alignment")
    print("-----------------------------------------")
    print_error_summary("Raw dense VO", raw_errors)
    print_error_summary("Optimized dense graph", opt_errors)

    plot_path = out_dir / "dense_raw_optimized_gt.png"
    plot_trajectories(
        dense_frames=dense_frames,
        keyframes=keyframes,
        raw=raw_positions,
        opt=opt_positions,
        gt=gt_positions,
        output_path=plot_path,
    )

    positions_path = out_dir / "dense_pose_comparison.txt"
    save_dense_positions(positions_path, dense_frames, raw_positions, opt_positions, gt_positions)

    keyframe_path = out_dir / "keyframes.txt"
    save_keyframe_table(keyframe_path, keyframes)

    print("")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved dense poses to: {positions_path}")
    print(f"Saved keyframes to: {keyframe_path}")
    print(f"Saved matches to: {out_dir / 'matches'}")


if __name__ == "__main__":
    main()