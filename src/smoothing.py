"""
Temporal smoothing for keypoint stabilization.
"""
import numpy as np
from collections import deque
from typing import Optional

class KeypointSmoother:
    """
    Applies exponential moving average to reduce keypoint jitter.
    
    Usage:
        smoother = KeypointSmoother(alpha=0.5)
        for result in results:
            smoothed_kps = smoother.smooth(result.keypoints)
    """
    
    def __init__(self, alpha: float = 0.5, min_confidence: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother
                   0.3 = very smooth, 0.7 = responsive
            min_confidence: Ignore keypoints below this confidence
        """
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_scores: Optional[np.ndarray] = None
    
    def smooth(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Apply exponential moving average smoothing.
        
        Args:
            keypoints: Current frame keypoints (N, 17, 2)
            scores: Current frame confidence scores (N, 17)
        
        Returns:
            Smoothed keypoints
        """
        if len(keypoints) == 0:
            return keypoints
        
        # Only smooth primary person (index 0)
        kps = keypoints[0].copy()
        conf = scores[0]
        
        if self.prev_keypoints is None:
            # First frame, no smoothing possible
            self.prev_keypoints = kps.copy()
            self.prev_scores = conf.copy()
            return keypoints
        
        # Apply EMA: smoothed = alpha * current + (1 - alpha) * previous
        for i in range(17):
            if conf[i] >= self.min_confidence and self.prev_scores[i] >= self.min_confidence:
                # Both current and previous are confident, blend them
                kps[i] = self.alpha * kps[i] + (1 - self.alpha) * self.prev_keypoints[i]
            elif conf[i] < self.min_confidence and self.prev_scores[i] >= self.min_confidence:
                # Current is uncertain, use previous
                kps[i] = self.prev_keypoints[i]
        
        # Update state
        self.prev_keypoints = kps.copy()
        self.prev_scores = conf.copy()
        
        # Put smoothed keypoints back
        smoothed = keypoints.copy()
        smoothed[0] = kps
        return smoothed
    
    def reset(self):
        """Reset smoother state for new video."""
        self.prev_keypoints = None
        self.prev_scores = None


class OneEuroFilter:
    """
    One Euro Filter - adaptive smoothing that's smooth on slow movements
    and responsive on fast movements. Better for sports!
    
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (lower = smoother)
            beta: Speed coefficient (higher = more responsive to fast motion)
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_filtered: Optional[np.ndarray] = None
        self.prev_velocity: Optional[np.ndarray] = None
    
    def _smoothing_factor(self, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau)
    
    def smooth(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Apply One Euro filtering."""
        if len(keypoints) == 0:
            return keypoints
        
        kps = keypoints[0].copy()
        
        if self.prev_keypoints is None:
            self.prev_keypoints = kps.copy()
            self.prev_filtered = kps.copy()
            self.prev_velocity = np.zeros_like(kps)
            return keypoints
        
        # Compute velocity
        velocity = kps - self.prev_keypoints
        
        # Filter velocity
        alpha_d = self._smoothing_factor(self.d_cutoff)
        filtered_velocity = alpha_d * velocity + (1 - alpha_d) * self.prev_velocity
        
        # Compute adaptive cutoff based on speed
        speed = np.linalg.norm(filtered_velocity, axis=1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        
        # Filter position
        alpha = self._smoothing_factor(cutoff.mean())
        filtered_kps = alpha * kps + (1 - alpha) * self.prev_filtered
        
        # Update state
        self.prev_keypoints = kps.copy()
        self.prev_filtered = filtered_kps.copy()
        self.prev_velocity = filtered_velocity.copy()
        
        smoothed = keypoints.copy()
        smoothed[0] = filtered_kps
        return smoothed
    
    def reset(self):
        self.prev_keypoints = None
        self.prev_filtered = None
        self.prev_velocity = None
