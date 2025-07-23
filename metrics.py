"""
metrics.py
Contains functions for calculating similarity metrics and object detection comparisons.
"""
import numpy as np
from math import sqrt

# Constants
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
VISIBILITY_THRESHOLD = 0.5  # Minimum confidence for a keypoint to be considered valid in OKS


def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0  # Cannot compare if one is missing

    # Ensure numpy arrays for vectorized operations
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    if vec1.shape != vec2.shape or len(vec1.shape) != 1:
        print(f"[Cosine Sim Warning] Vector shape mismatch or not 1D: {vec1.shape} vs {vec2.shape}")
        return 0.0

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate norms (magnitude)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    if norm_vec1 > 0 and norm_vec2 > 0:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        # Clip similarity to [-1, 1] due to potential floating point inaccuracies
        return np.clip(similarity, -1.0, 1.0)
    else:
        # One or both vectors have zero magnitude
        return 0.0


def calculate_oks(keypoints1_data, keypoints2_data, bbox1_area):
    """Calculates Object Keypoint Similarity (OKS) between two sets of keypoints.

    Args:
        keypoints1_data: Tuple (kpts1_xy, kpts1_conf) for the first person.
                         kpts1_xy is Nx2 numpy array or list of lists.
                         kpts1_conf is N length numpy array or list (optional).
        keypoints2_data: Tuple (kpts2_xy, kpts2_conf) for the second person.
                         Format same as keypoints1_data.
        bbox1_area: The area (w*h) of the bounding box for the first person.
                    Used for scaling the distance penalty.

    Returns:
        float: The OKS score (0.0 to 1.0).
               Returns 0.0 if keypoints are missing or formats are incompatible.
    """
    if keypoints1_data is None or keypoints2_data is None:
        return 0.0

    # Check for sequence type and length before unpacking
    if not isinstance(keypoints1_data, (list, tuple)) or len(keypoints1_data) != 2:
        print(f"[calculate_oks Warning] Invalid format for keypoints1_data (type: {type(keypoints1_data)}, len: {len(keypoints1_data) if isinstance(keypoints1_data, (list, tuple)) else 'N/A'}). Returning 0.")
        return 0.0
    if not isinstance(keypoints2_data, (list, tuple)) or len(keypoints2_data) != 2:
        print(f"[calculate_oks Warning] Invalid format for keypoints2_data (type: {type(keypoints2_data)}, len: {len(keypoints2_data) if isinstance(keypoints2_data, (list, tuple)) else 'N/A'}). Returning 0.")
        return 0.0

    # Unpack after validation
    kpts1_xy, kpts1_conf = keypoints1_data
    kpts2_xy, kpts2_conf = keypoints2_data

    # Re-check for None after unpacking
    if kpts1_xy is None or kpts2_xy is None:
        return 0.0

    # Convert to numpy arrays if they are lists
    kpts1_xy = np.array(kpts1_xy)
    kpts2_xy = np.array(kpts2_xy)
    kpts1_conf = np.array(kpts1_conf) if kpts1_conf is not None else None
    
    if kpts1_xy.shape != kpts2_xy.shape or kpts1_xy.shape[1] != 2:
        print(f"[calculate_oks Warning] Keypoint shape mismatch: {kpts1_xy.shape} vs {kpts2_xy.shape}")
        return 0.0  # Shape mismatch
        
    num_keypoints = kpts1_xy.shape[0]
    if num_keypoints == 0:
        return 0.0

    # Handle mismatched keypoint counts with COCO standard
    if num_keypoints != len(OKS_SIGMAS):
        sigmas = np.full(num_keypoints, 0.5)  # Default falloff if not COCO standard
    else:
        sigmas = OKS_SIGMAS
        
    # Calculate squared distances between corresponding keypoints
    sq_distances = np.sum((kpts1_xy - kpts2_xy)**2, axis=1)

    # Scale factor based on bounding box area
    scale_factor = max(bbox1_area, 1.0)  # Avoid division by zero
    
    # Visibility flags (use confidence from the first set)
    visible = np.ones(num_keypoints)  # Default to visible
    if kpts1_conf is not None:
        # Consider keypoints with confidence > 0.1 as somewhat visible
        visibility_threshold = 0.1 
        visible = (kpts1_conf > visibility_threshold).astype(float)
        
    # Calculate OKS per keypoint
    variance = (2 * sigmas)**2
    oks_per_keypoint = np.exp(-sq_distances / (scale_factor * variance + np.finfo(float).eps))

    # Final OKS is the average over visible keypoints
    total_visible = np.sum(visible)
    if total_visible == 0:
        return 0.0  # No visible keypoints to compare
        
    oks_score = np.sum(oks_per_keypoint * visible) / total_visible
    
    return oks_score


def calculate_reid_similarity(kps1, kps2, bbox1, bbox2, feat1, feat2, state1, state2, 
                             time_gap, det_confidence, config):
    """Calculates a combined similarity score for Re-Identification.

    Args:
        kps1: Keypoints data (tuple: xy, conf) for track 1 (inactive track).
        kps2: Keypoints data (tuple: xy, conf) for track 2 (new detection).
        bbox1: Bounding box [x1, y1, x2, y2] for track 1.
        bbox2: Bounding box [x1, y1, x2, y2] for track 2.
        feat1: Feature vector for track 1.
        feat2: Feature vector for track 2.
        state1: State string for track 1.
        state2: State string for track 2.
        time_gap: Time difference (seconds) between last seen inactive and current detection.
        det_confidence: Confidence score of the new detection.
        config: Dictionary with Re-ID configuration parameters.

    Returns:
        tuple: (float combined_score, dict debug_scores)
    """
    oks_score = 0.0
    spatial_score = 0.0
    appearance_score = 0.0
    state_score = 0.0
    final_score = 0.0

    # Extract config parameters
    reid_score_weight_oks = config.get('weight_oks', 0.35)
    reid_score_weight_spatial = config.get('weight_spatial', 0.25)
    reid_score_weight_appearance = config.get('weight_appearance', 0.30)
    reid_score_weight_state = config.get('weight_state', 0.10)
    reid_max_spatial_distance = config.get('max_spatial_distance', 256)
    reid_appearance_threshold = config.get('appearance_threshold', 0.5)
    reid_confidence_factor = config.get('confidence_factor', 0.5)

    # 1. OKS Score
    if kps1 is not None and kps2 is not None and bbox1 is not None:
        try:
            # Ensure bbox1 is array-like with 4 elements for area calculation
            if hasattr(bbox1, '__len__') and len(bbox1) == 4:
                bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                # Ensure area is positive
                if bbox1_area > 0:
                    oks_score = calculate_oks(kps1, kps2, bbox1_area)
                else: 
                    oks_score = 0.0  # Invalid bbox area
            else: 
                oks_score = 0.0  # Invalid bbox format
        except Exception as e_oks:
            print(f"[Re-ID Warning] OKS calculation failed: {e_oks}")
            oks_score = 0.0

    # 2. Spatial Score
    if bbox1 is not None and bbox2 is not None:
        try:
            # Ensure bboxes are array-like with 4 elements
            if hasattr(bbox1, '__len__') and len(bbox1) == 4 and hasattr(bbox2, '__len__') and len(bbox2) == 4:
                center1_x = (bbox1[0] + bbox1[2]) / 2
                center1_y = (bbox1[1] + bbox1[3]) / 2
                center2_x = (bbox2[0] + bbox2[2]) / 2
                center2_y = (bbox2[1] + bbox2[3]) / 2
                distance = sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                # Normalize distance: closer = higher score (max 1.0)
                spatial_score = max(0.0, 1.0 - (distance / reid_max_spatial_distance))
            else: 
                spatial_score = 0.0  # Invalid bbox format
        except Exception as e_spatial:
            print(f"[Re-ID Warning] Spatial score calculation failed: {e_spatial}")
            spatial_score = 0.0

    # 3. Appearance Score
    if feat1 is not None and feat2 is not None:
        try:
            # Ensure features are not empty lists/arrays before calculating
            feat1_arr = np.asarray(feat1)
            feat2_arr = np.asarray(feat2)
            if feat1_arr.size > 0 and feat2_arr.size > 0:
                cosine_sim = calculate_cosine_similarity(feat1_arr, feat2_arr)
                # Only contribute positively if above threshold, scale to 0-1 range
                # Avoid division by zero if threshold is 1.0
                denom = (1.0 - reid_appearance_threshold)
                appearance_score = (cosine_sim - reid_appearance_threshold) / denom if denom > 0 else 1.0
            else: 
                appearance_score = 0.0  # Empty features
        except Exception as e_appear:
            print(f"[Re-ID Warning] Appearance score calculation failed: {e_appear}")
            appearance_score = 0.0

    # 4. State Consistency Score
    from track import STATE_UNKNOWN
    if state1 is not None and state2 is not None:
        if state1 == state2 and state1 != STATE_UNKNOWN:
            state_score = 1.0  # Perfect match (and not Unknown)
        elif state1 != STATE_UNKNOWN and state2 != STATE_UNKNOWN:
            state_score = 0.2  # Mismatch, small penalty/score
        else:  # One or both are Unknown
            state_score = 0.5  # Neutral score if state is unknown
    else:
        state_score = 0.5  # Neutral if states are missing

    # Combine Scores Weighted
    final_score = (
        oks_score * reid_score_weight_oks +
        spatial_score * reid_score_weight_spatial +
        appearance_score * reid_score_weight_appearance +
        state_score * reid_score_weight_state
    )
    
    # Factor in Detection Confidence
    # Confidence factor: Higher confidence slightly boosts score
    conf_factor = 1.0 + (reid_confidence_factor * (det_confidence - 0.5))
    
    # Apply factors
    final_score *= conf_factor

    # Clamp final score to [0, 1]
    final_score = max(0.0, min(1.0, final_score))
    
    debug_scores = {
        "oks": round(oks_score, 3),
        "spatial": round(spatial_score, 3),
        "appearance": round(appearance_score, 3),
        "state": round(state_score, 3),
        "conf_factor": round(conf_factor, 3),
        "combined": round(final_score, 3)
    }

    return final_score, debug_scores 