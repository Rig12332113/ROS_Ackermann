import numpy as np
from dataclasses import dataclass
import cv2

@dataclass
class Keyframe:
    id: int
    timestamp: float
    image: np.ndarray
    depth: np.ndarray
    pose: np.ndarray 
    optimized_pose: np.ndarray
    keypoints: list = None
    descriptors: np.ndarray = None

class KeyframeManager:
    def __init__(self):
        self.keyframes = []
        self.time_threshold = 0.3  # threshold for time gap
        self.keypoints_threshold = 100  # threshold for keyframe selection
        self.translation_threshold = 0.25  # threshold for translation
        self.rotation_threshold = 0.1  # threshold for rotation in radians
        self.overlap_threshold = 0.7  # threshold for descriptor overlap
        self.min_good_matches = 20  # minimum number of good matches for keyframe selection
        self.ratio_threshold = 0.75  # ratio test threshold for descriptor matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def compute_descriptor_overlap(self, desc_ref, desc_curr):
        if desc_ref is None or desc_curr is None:
            return 0.0, 0

        if len(desc_ref) == 0 or len(desc_curr) == 0:
            return 0.0, 0

        knn_matches = self.bf.knnMatch(desc_ref, desc_curr, k=2)

        good_matches = []

        for pair in knn_matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        denom = min(len(desc_ref), len(desc_curr))
        overlap = len(good_matches) / denom if denom > 0 else 0.0

        return overlap, len(good_matches)
    
    def is_keyframe(self, timestamp, pose, keypoints, descriptors):
        if keypoints is None or descriptors is None:
            return False

        if len(keypoints) < self.keypoints_threshold:
            return False

        if len(self.keyframes) == 0:
            return True

        last_kf = self.keyframes[-1]

        if timestamp - last_kf.timestamp < self.time_threshold:
            return False

        T_pre = last_kf.pose
        T_curr = pose
        T_rel = np.linalg.inv(T_pre) @ T_curr

        translation = np.linalg.norm(T_rel[:3, 3])

        cos_angle = (np.trace(T_rel[:3, :3]) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rotation = np.arccos(cos_angle)

        if not (
            translation > self.translation_threshold
            or rotation > self.rotation_threshold
        ):
            return False

        overlap, good_match_count = self.compute_descriptor_overlap(
            last_kf.descriptors,
            descriptors,
        )

        if good_match_count < self.min_good_matches:
            return False

        if overlap > self.overlap_threshold:
            return False

        return True

    def add_keyframe(self, timestamp, image, depth, pose, keypoints, descriptors):
        if self.is_keyframe(timestamp, pose, keypoints, descriptors):
            self.keyframes.append(Keyframe(
                    id=len(self.keyframes),
                    timestamp=timestamp,
                    image=image.copy(),
                    depth=depth.copy(),
                    pose=pose.copy(),
                    optimized_pose=pose.copy(),
                    keypoints=keypoints,
                    descriptors=descriptors.copy()
                    ))
            return True
        return False
    
            
    def get_keyframes(self):
        return self.keyframes