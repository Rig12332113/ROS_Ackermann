import numpy as np
from dataclasses import dataclass
import cv2

@dataclass
class Keyframe:
    id: int
    frame_id: int
    timestamp: float
    image: np.ndarray
    pose: np.ndarray 
    keypoints: list = None
    descriptors: np.ndarray = None

class KeyframeManager:
    def __init__(self):
        self.keyframes = []
        self.keypoints_threshold = 100  # threshold for keyframe selection
        self.translation_threshold = 0.25  # threshold for translation
        self.rotation_threshold = 0.1  # threshold for rotation in radians
        self.frame_gap_threshold = 5  # minimum number of frames between keyframes
        self.overlap_threshold = 0.7  # threshold for descriptor overlap
        self.min_good_matches = 20  # minimum number of good matches for keyframe selection
        self.ratio_threshold = 0.75  # ratio test threshold for descriptor matching
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute_keypoints_and_descriptors(self, image):
        # Placeholder for keypoint detection and descriptor extraction
        # You can use ORB, SIFT, or any other feature extractor
        if image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        keypoints, descriptors = self.orb.detectAndCompute(gray_img, None)
        return keypoints, descriptors
    
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
    
    def is_keyframe(self, frame_id, pose, keypoints, descriptors):
        if len(self.keyframes) == 0:
            return True
        if frame_id - self.keyframes[-1].frame_id < self.frame_gap_threshold:
            return False
        if len(keypoints) < self.keypoints_threshold:
            return False
        T_pre = self.keyframes[-1].pose
        T_curr= pose
        T_rel = np.linalg.inv(T_pre) @ T_curr
        translation = np.linalg.norm(T_rel[:3, 3])
        rotation = np.arccos((np.trace(T_rel[:3, :3]) - 1) / 2)
        if not (translation > self.translation_threshold or rotation > self.rotation_threshold):
            return False
        
        overlap, good_match_count = self.compute_descriptor_overlap(
            self.keyframes[-1].descriptors,
            descriptors,
        )   

        if good_match_count < self.min_good_matches:
            return False

        if overlap > self.overlap_threshold:
            return False

        return True
    
    def add_keyframe(self, frame_id, pose, image):
        keypoints, descriptors = self.compute_keypoints_and_descriptors(image)
        if self.is_keyframe(frame_id, pose, keypoints, descriptors):
            self.keyframes.append(Keyframe(
                    id=len(self.keyframes),
                    frame_id=frame_id,
                    timestamp=0.0,
                    image=image,
                    pose=pose,
                    keypoints=keypoints,
                    descriptors=descriptors
                ))

    def get_keyframes(self):
        return self.keyframes