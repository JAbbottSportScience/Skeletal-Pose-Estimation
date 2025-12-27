"""
================================================================================
BIOMECHANICS MODULE
================================================================================
Joint angle calculations and lead/trail leg identification for sprint analysis.
================================================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional

# COCO Keypoint Indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by vectors p1→p2 and p2→p3.
    
    Args:
        p1, p2, p3: Points as (x, y) arrays
        
    Returns:
        Angle in degrees (0-180)
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return np.degrees(np.arccos(cos_angle))


def get_leading_trailing_legs(
    keypoints: np.ndarray,
    running_direction: str = "auto"
) -> Dict:
    """
    Identify leading (front) and trailing (back) leg based on ankle position.
    
    More reliable than left/right for side-view sprint analysis since
    pose models often confuse L/R when limbs overlap.
    
    Args:
        keypoints: COCO keypoints array (17, 2)
        running_direction: "left" (←), "right" (→), or "auto" (detect)
        
    Returns:
        Dict with 'lead' and 'trail' leg data
    """
    l_ankle, r_ankle = keypoints[L_ANKLE], keypoints[R_ANKLE]
    l_knee, r_knee = keypoints[L_KNEE], keypoints[R_KNEE]
    l_hip, r_hip = keypoints[L_HIP], keypoints[R_HIP]
    
    # Auto-detect running direction from hip movement
    # (In a sequence, you'd track this over time)
    if running_direction == "auto":
        # Assume running left-to-right (positive x direction)
        # Leading leg = ankle with higher x value
        running_direction = "right"
    
    if running_direction == "right":
        # Running →, leading leg has higher x
        left_is_leading = l_ankle[0] > r_ankle[0]
    else:
        # Running ←, leading leg has lower x
        left_is_leading = l_ankle[0] < r_ankle[0]
    
    if left_is_leading:
        return {
            'lead': {
                'side': 'left',
                'ankle': l_ankle,
                'knee': l_knee,
                'hip': l_hip,
                'ankle_idx': L_ANKLE,
                'knee_idx': L_KNEE,
                'hip_idx': L_HIP,
            },
            'trail': {
                'side': 'right',
                'ankle': r_ankle,
                'knee': r_knee,
                'hip': r_hip,
                'ankle_idx': R_ANKLE,
                'knee_idx': R_KNEE,
                'hip_idx': R_HIP,
            },
        }
    else:
        return {
            'lead': {
                'side': 'right',
                'ankle': r_ankle,
                'knee': r_knee,
                'hip': r_hip,
                'ankle_idx': R_ANKLE,
                'knee_idx': R_KNEE,
                'hip_idx': R_HIP,
            },
            'trail': {
                'side': 'left',
                'ankle': l_ankle,
                'knee': l_knee,
                'hip': l_hip,
                'ankle_idx': L_ANKLE,
                'knee_idx': L_KNEE,
                'hip_idx': L_HIP,
            },
        }


def calculate_joint_angles(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate hip and knee flexion for lead and trail legs.
    
    Hip Flexion/Extension: Angle of thigh relative to VERTICAL
        - 0° = thigh pointing straight down
        - +ve = thigh forward of vertical (flexion)
        - -ve = thigh behind vertical (extension)
        
        Example values during sprint:
            Lead leg at max knee drive: +50° to +70°
            Mid-stance: ~0° to +10°
            Trail leg at toe-off: -20° to -30°
        
    Knee Flexion: Angle at knee (hip-knee-ankle)
        - 0° = straight leg (full extension)
        - +ve = bent knee (flexion)
    
    Args:
        keypoints: COCO keypoints array (17, 2)
        
    Returns:
        Dict with lead/trail hip and knee angles
    """
    legs = get_leading_trailing_legs(keypoints)
    
    # Vertical reference (pointing DOWN in image coordinates)
    vertical = np.array([0, 1])
    
    def signed_angle_from_vertical(hip: np.ndarray, knee: np.ndarray) -> float:
        """
        Calculate signed angle of thigh from vertical.
        Positive = forward (flexion), Negative = backward (extension)
        """
        # Thigh vector (hip to knee)
        thigh = knee - hip
        thigh_norm = np.linalg.norm(thigh)
        
        if thigh_norm < 1e-6:
            return 0.0
        
        thigh = thigh / thigh_norm
        
        # Angle from vertical (unsigned)
        cos_angle = np.dot(thigh, vertical)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Sign: positive if knee is in front of hip (forward flexion)
        # In image coords, "forward" depends on running direction
        # Use x-component: if knee.x > hip.x, thigh is angled forward-right
        # We'll use the x-component of thigh vector for sign
        if thigh[0] > 0:  # Knee is to the right of hip
            sign = 1  # Forward flexion (assuming running right)
        else:
            sign = -1  # Backward extension
        
        return sign * angle
    
    # Lead leg angles
    lead_hip_angle = signed_angle_from_vertical(
        legs['lead']['hip'], 
        legs['lead']['knee']
    )
    lead_knee_angle = calculate_angle(
        legs['lead']['hip'],
        legs['lead']['knee'],
        legs['lead']['ankle']
    )
    
    # Trail leg angles  
    trail_hip_angle = signed_angle_from_vertical(
        legs['trail']['hip'],
        legs['trail']['knee']
    )
    trail_knee_angle = calculate_angle(
        legs['trail']['hip'],
        legs['trail']['knee'],
        legs['trail']['ankle']
    )
    
    # Knee flexion = 180 - angle (so 0° = straight, higher = more bent)
    return {
        'lead_hip_flexion': lead_hip_angle,
        'trail_hip_flexion': trail_hip_angle,
        'lead_knee_flexion': 180 - lead_knee_angle,
        'trail_knee_flexion': 180 - trail_knee_angle,
        'lead_side': legs['lead']['side'],
        'trail_side': legs['trail']['side'],
    }


def calculate_stride_metrics(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate additional stride metrics.
    
    Returns:
        Dict with stride metrics
    """
    legs = get_leading_trailing_legs(keypoints)
    
    # Ankle separation (stride width in x)
    ankle_separation = abs(
        legs['lead']['ankle'][0] - legs['trail']['ankle'][0]
    )
    
    # Knee height relative to hip (for knee drive)
    hip_mid_y = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2
    lead_knee_height = hip_mid_y - legs['lead']['knee'][1]  # Positive = above hip
    trail_knee_height = hip_mid_y - legs['trail']['knee'][1]
    
    # Ankle height (for ground contact detection)
    # Lower ankle y = closer to ground (in image coords, y increases downward)
    max_ankle_y = max(legs['lead']['ankle'][1], legs['trail']['ankle'][1])
    lead_ankle_height = max_ankle_y - legs['lead']['ankle'][1]
    trail_ankle_height = max_ankle_y - legs['trail']['ankle'][1]
    
    return {
        'ankle_separation': ankle_separation,
        'lead_knee_height': lead_knee_height,
        'trail_knee_height': trail_knee_height,
        'lead_ankle_height': lead_ankle_height,
        'trail_ankle_height': trail_ankle_height,
    }


def detect_gait_phase(keypoints: np.ndarray) -> str:
    """
    Detect current gait phase based on leg positions.
    
    Phases:
        - "initial_contact": Lead foot about to strike
        - "mid_stance": Lead leg under body, supporting
        - "toe_off": Trail leg pushing off
        - "swing": Both feet off ground (flight)
        - "max_knee_drive": Lead knee at peak height
        
    Returns:
        Phase name string
    """
    legs = get_leading_trailing_legs(keypoints)
    metrics = calculate_stride_metrics(keypoints)
    angles = calculate_joint_angles(keypoints)
    
    lead_ankle_y = legs['lead']['ankle'][1]
    trail_ankle_y = legs['trail']['ankle'][1]
    
    # Simple heuristics (tune thresholds for your setup)
    
    # Max knee drive: Lead knee high, trail leg extended
    if metrics['lead_knee_height'] > 50 and angles['trail_knee_flexion'] < 30:
        return "max_knee_drive"
    
    # Initial contact: Lead ankle low, leg extending
    if angles['lead_knee_flexion'] < 30 and lead_ankle_y > trail_ankle_y:
        return "initial_contact"
    
    # Toe off: Trail ankle high, trail knee flexing
    if metrics['trail_ankle_height'] > 30 and angles['trail_knee_flexion'] > 40:
        return "toe_off"
    
    # Mid stance: Ankles close together
    if metrics['ankle_separation'] < 100:
        return "mid_stance"
    
    return "swing"


# ==============================================================================
# ANALYSIS HELPERS
# ==============================================================================

def analyze_frame(keypoints: np.ndarray, scores: np.ndarray, min_conf: float = 0.3) -> Optional[Dict]:
    """
    Complete biomechanical analysis for a single frame.
    
    Args:
        keypoints: COCO keypoints (17, 2)
        scores: Confidence scores (17,)
        min_conf: Minimum confidence threshold
        
    Returns:
        Dict with all metrics, or None if insufficient confidence
    """
    # Check key joint confidence
    key_joints = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if any(scores[j] < min_conf for j in key_joints):
        return None
    
    angles = calculate_joint_angles(keypoints)
    metrics = calculate_stride_metrics(keypoints)
    phase = detect_gait_phase(keypoints)
    
    return {
        **angles,
        **metrics,
        'gait_phase': phase,
    }