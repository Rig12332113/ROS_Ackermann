import numpy as np
import gtsam
from gtsam import symbol


def np_to_pose3(T):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
    )
    return gtsam.Pose3(R, t)


def pose3_to_np(pose):
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = np.array([
        pose.x(),
        pose.y(),
        pose.z(),
    ])
    return T


class RealtimePoseGraph:
    def __init__(self):
        self.isam = gtsam.ISAM2()

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = gtsam.Values()

        self.keyframe_ids = []
        self.current_estimates = {}

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
        )

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10])
        )

    def add_first_keyframe(self, keyframe_id, T_world_keyframe):
        key = symbol("x", keyframe_id)
        pose = np_to_pose3(T_world_keyframe)

        self.graph.add(
            gtsam.PriorFactorPose3(
                key,
                pose,
                self.prior_noise,
            )
        )

        self.initial.insert(key, pose)

        self.isam.update(self.graph, self.initial)
        self.result = self.isam.calculateEstimate()

        self.graph.resize(0)
        self.initial.clear()

        self.keyframe_ids.append(keyframe_id)
        self.current_estimates[keyframe_id] = T_world_keyframe.copy()

    def add_keyframe(self, keyframe_id, prev_keyframe_id, T_prev_curr):
        prev_key = symbol("x", prev_keyframe_id)
        curr_key = symbol("x", keyframe_id)

        relative_pose = np_to_pose3(T_prev_curr)

        self.graph.add(
            gtsam.BetweenFactorPose3(
                prev_key,
                curr_key,
                relative_pose,
                self.odom_noise,
            )
        )

        if prev_keyframe_id in self.current_estimates:
            T_world_prev = self.current_estimates[prev_keyframe_id]
        else:
            T_world_prev = pose3_to_np(self.result.atPose3(prev_key))

        T_world_curr_init = T_world_prev @ T_prev_curr
        curr_pose_init = np_to_pose3(T_world_curr_init)

        self.initial.insert(curr_key, curr_pose_init)

        self.isam.update(self.graph, self.initial)
        self.result = self.isam.calculateEstimate()

        self.graph.resize(0)
        self.initial.clear()

        self.keyframe_ids.append(keyframe_id)

        for kid in self.keyframe_ids:
            k = symbol("x", kid)
            if self.result.exists(k):
                self.current_estimates[kid] = pose3_to_np(self.result.atPose3(k))

    def add_loop_factor(self, old_keyframe_id, new_keyframe_id, T_old_new):
        old_key = symbol("x", old_keyframe_id)
        new_key = symbol("x", new_keyframe_id)

        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.40, 0.40, 0.40, 0.50, 0.50, 0.50])
        )

        self.graph.add(
            gtsam.BetweenFactorPose3(
                old_key,
                new_key,
                np_to_pose3(T_old_new),
                loop_noise,
            )
        )

        self.isam.update(self.graph, self.initial)
        self.result = self.isam.calculateEstimate()

        self.graph.resize(0)
        self.initial.clear()

        for kid in self.keyframe_ids:
            k = symbol("x", kid)
            if self.result.exists(k):
                self.current_estimates[kid] = pose3_to_np(self.result.atPose3(k))

    def get_optimized_poses(self):
        poses = {}

        for kid in self.keyframe_ids:
            key = symbol("x", kid)

            if self.result.exists(key):
                poses[kid] = pose3_to_np(self.result.atPose3(key))

        return poses