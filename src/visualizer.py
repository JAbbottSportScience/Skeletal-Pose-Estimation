"""
================================================================================
VISUALIZER AND KINOGRAM GENERATOR MODULE
================================================================================
Creates visualizations of pose estimation results including:
    - Skeleton overlay on frames
    - Keypoint annotation with confidence coloring
    - Kinogram composite images for gait analysis
    - Sprint phase detection

Note: YOLO has built-in visualization via result.plot(), but this module
provides more control for biomechanical analysis and kinogram generation.

Author: [Your Name]
================================================================================
"""

import sys
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PoseEstimationConfig
from src.pose_estimator import KeypointResult


# ==============================================================================
# SKELETON VISUALIZER
# ==============================================================================

class SkeletonVisualizer:
    """
    Draws skeleton visualizations on frames.
    
    Example:
        >>> viz = SkeletonVisualizer()
        >>> annotated = viz.draw(frame, keypoints, scores)
        >>> cv2.imwrite("annotated.jpg", annotated)
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
        self.viz_config = self.config.visualization
        self.kp_config = self.config.keypoints
    
    def draw(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Draw skeleton visualization on a frame.
        
        Args:
            frame: BGR image (H, W, 3)
            keypoints: Keypoint coordinates (17, 2)
            scores: Keypoint confidence scores (17,)
            color: Override color (None = use config)
            draw_skeleton: Draw connecting lines
            draw_keypoints: Draw keypoint circles
            copy: Work on a copy of the frame
        
        Returns:
            Annotated frame
        """
        if copy:
            frame = frame.copy()
        
        threshold = self.config.model.confidence_threshold
        
        if draw_skeleton:
            frame = self._draw_skeleton_lines(frame, keypoints, scores, threshold, color)
        
        if draw_keypoints:
            frame = self._draw_keypoint_circles(frame, keypoints, scores, threshold, color)
        
        return frame
    
    def _draw_skeleton_lines(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw lines connecting keypoints to form skeleton."""
        skeleton = self.kp_config.skeleton
        line_color = color or self.viz_config.skeleton_color
        thickness = self.viz_config.skeleton_thickness
        
        for start_idx, end_idx in skeleton:
            # Only draw if both keypoints are confident
            if scores[start_idx] < threshold or scores[end_idx] < threshold:
                continue
            
            start_pos = tuple(keypoints[start_idx].astype(int))
            end_pos = tuple(keypoints[end_idx].astype(int))
            
            cv2.line(frame, start_pos, end_pos, line_color, thickness, cv2.LINE_AA)
        
        return frame
    
    def _draw_keypoint_circles(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw circles at each keypoint position."""
        radius = self.viz_config.keypoint_radius
        
        for idx, (kp, score) in enumerate(zip(keypoints, scores)):
            x, y = int(kp[0]), int(kp[1])
            
            if x < 0 or y < 0:
                continue
            
            # Determine color
            if color is not None:
                kp_color = color
            elif self.viz_config.confidence_color_map:
                kp_color = self._get_confidence_color(score)
            else:
                kp_color = self.viz_config.keypoint_color
            
            # Smaller radius for low-confidence
            actual_radius = max(2, radius // 2) if score < threshold else radius
            
            cv2.circle(frame, (x, y), actual_radius, kp_color, -1, cv2.LINE_AA)
        
        return frame
    
    def _get_confidence_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on confidence score."""
        if score >= 0.8:
            return (0, 255, 0)      # Green
        elif score >= 0.5:
            return (0, 255, 255)    # Yellow
        else:
            return (0, 0, 255)      # Red
    
    def draw_multi_person(
        self,
        frame: np.ndarray,
        result: KeypointResult,
        copy: bool = True,
    ) -> np.ndarray:
        """Draw skeletons for all detected people with distinct colors."""
        if copy:
            frame = frame.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]
        
        for i in range(result.num_people):
            color = colors[i % len(colors)]
            frame = self.draw(
                frame, result.keypoints[i], result.scores[i],
                color=color, copy=False
            )
        
        return frame
    
    def draw_yolo_native(
        self,
        result,  # YOLO Results object
    ) -> np.ndarray:
        """
        Use YOLO's built-in visualization.
        
        This is a shortcut using YOLO's native plotting.
        
        Args:
            result: Ultralytics Results object
        
        Returns:
            Annotated frame
        """
        return result.plot()


# ==============================================================================
# SPRINT PHASE DETECTOR
# ==============================================================================

class SprintPhaseDetector:
    """
    Detects key phases in sprint gait cycle from keypoint data.
    
    Phases detected:
        - Toe-off (propulsion end)
        - Maximum knee drive
        - Ground contact
        - Mid-stance
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
    
    def detect_phases(
        self,
        results: List[KeypointResult],
        num_phases: Optional[int] = None,
    ) -> List[int]:
        """
        Detect key phase frame indices from pose results.
        
        Args:
            results: List of KeypointResult from video processing
            num_phases: Number of phases to detect (uses config if None)
        
        Returns:
            List of frame indices representing key phases
        """
        num_phases = num_phases or self.config.kinogram.num_phases
        
        if len(results) == 0:
            return []
        
        if len(results) <= num_phases:
            return [r.frame_idx for r in results]
        
        # Calculate biomechanical metrics
        metrics = self._calculate_metrics(results)
        
        if metrics is None:
            return self._uniform_phases(results, num_phases)
        
        # Find peaks in knee height (max knee drive)
        knee_peaks = self._find_peaks(metrics["knee_height"])
        
        # Find valleys in ankle height (ground contact)
        ankle_valleys = self._find_valleys(metrics["ankle_height"])
        
        # Combine and select best phases
        all_phases = set(knee_peaks + ankle_valleys)
        
        if len(all_phases) >= num_phases:
            phase_list = sorted(all_phases)
            step = len(phase_list) / num_phases
            selected = [phase_list[int(i * step)] for i in range(num_phases)]
        else:
            selected = list(all_phases)
            uniform = self._uniform_phases(results, num_phases - len(selected) + 2)[1:-1]
            selected = sorted(set(selected + uniform))[:num_phases]
        
        return [results[i].frame_idx for i in sorted(selected)]
    
    def _calculate_metrics(
        self,
        results: List[KeypointResult],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Calculate biomechanical metrics for phase detection."""
        # COCO keypoint indices
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16
        
        knee_heights = []
        ankle_heights = []
        
        for result in results:
            person = result.get_primary_person()
            
            if person is None:
                knee_heights.append(np.nan)
                ankle_heights.append(np.nan)
                continue
            
            keypoints, scores = person
            
            # Knee height relative to hip
            hip_y = (keypoints[L_HIP, 1] + keypoints[R_HIP, 1]) / 2
            knee_y_max = min(keypoints[L_KNEE, 1], keypoints[R_KNEE, 1])
            knee_heights.append(hip_y - knee_y_max)
            
            # Ankle height (lower = ground contact)
            ankle_y_max = max(keypoints[L_ANKLE, 1], keypoints[R_ANKLE, 1])
            ankle_heights.append(ankle_y_max)
        
        return {
            "knee_height": np.array(knee_heights),
            "ankle_height": np.array(ankle_heights),
        }
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find local maxima in data array."""
        peaks = []
        for i in range(1, len(data) - 1):
            if np.isnan(data[i]):
                continue
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return peaks
    
    def _find_valleys(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find local minima."""
        return self._find_peaks(-data, min_distance)
    
    def _uniform_phases(self, results: List[KeypointResult], num_phases: int) -> List[int]:
        """Get uniformly spaced frame indices."""
        n = len(results)
        step = n / (num_phases + 1)
        return [int((i + 1) * step) for i in range(num_phases)]


# ==============================================================================
# KINOGRAM GENERATOR
# ==============================================================================

class KinogramGenerator:
    """
    Generates kinogram composite images from video frames and pose data.
    
    Example:
        >>> generator = KinogramGenerator()
        >>> kinogram = generator.generate(frames, results)
        >>> cv2.imwrite("kinogram.png", kinogram)
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
        self.visualizer = SkeletonVisualizer(config)
        self.phase_detector = SprintPhaseDetector(config)
    
    def generate(
        self,
        frames: List[np.ndarray],
        results: List[KeypointResult],
        phase_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Generate a kinogram from video frames and pose results.
        
        Args:
            frames: List of video frames (BGR images)
            results: List of KeypointResult matching frames
            phase_indices: Specific frame indices (auto-detect if None)
        
        Returns:
            Kinogram image as numpy array (BGR)
        """
        if len(frames) != len(results):
            raise ValueError("Frame count must match result count")
        
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        # Build frame index mapping
        frame_indices = [r.frame_idx for r in results]
        idx_to_local = {idx: i for i, idx in enumerate(frame_indices)}
        
        # Detect phases if not provided
        if phase_indices is None:
            phase_indices = self.phase_detector.detect_phases(
                results, self.config.kinogram.num_phases
            )
        
        valid_phases = [idx for idx in phase_indices if idx in idx_to_local]
        
        if not valid_phases:
            raise ValueError("No valid phase indices found")
        
        print(f"Generating kinogram with {len(valid_phases)} phases: {valid_phases}")
        
        # Create background
        background = self._create_background(frames, idx_to_local, valid_phases)
        kinogram = background.copy()
        
        # Draw each phase
        phase_colors = self.config.visualization.phase_colors
        
        for i, phase_idx in enumerate(valid_phases):
            local_idx = idx_to_local[phase_idx]
            result = results[local_idx]
            color = phase_colors[i % len(phase_colors)]
            
            person = result.get_primary_person()
            if person is not None:
                keypoints, scores = person
                kinogram = self.visualizer.draw(
                    kinogram, keypoints, scores, color=color, copy=False
                )
        
        # Add legend
        kinogram = self._add_legend(kinogram, valid_phases, phase_colors)
        
        return kinogram
    
    def _create_background(
        self,
        frames: List[np.ndarray],
        idx_map: Dict[int, int],
        phase_indices: List[int],
    ) -> np.ndarray:
        """Create background image for kinogram."""
        mode = self.config.kinogram.background_mode
        
        if mode == "first":
            return frames[idx_map[phase_indices[0]]].copy()
        elif mode == "average":
            phase_frames = [frames[idx_map[idx]] for idx in phase_indices]
            return np.mean(phase_frames, axis=0).astype(np.uint8)
        elif mode == "black":
            h, w = frames[0].shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        elif mode == "white":
            h, w = frames[0].shape[:2]
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        else:
            return frames[idx_map[phase_indices[0]]].copy()
    
    def _add_legend(
        self,
        image: np.ndarray,
        phase_indices: List[int],
        colors: Tuple[Tuple[int, int, int], ...],
    ) -> np.ndarray:
        """Add phase legend to kinogram."""
        margin, box_size, text_offset, line_height = 10, 20, 25, 30
        legend_width, legend_height = 150, len(phase_indices) * line_height + 10
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (margin - 5, margin - 5),
            (margin + legend_width, margin + legend_height),
            (0, 0, 0), -1
        )
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        
        # Draw entries
        for i, phase_idx in enumerate(phase_indices):
            y = margin + i * line_height + 15
            color = colors[i % len(colors)]
            
            cv2.rectangle(
                image,
                (margin, y - box_size // 2),
                (margin + box_size, y + box_size // 2),
                color, -1
            )
            cv2.putText(
                image, f"Frame {phase_idx}",
                (margin + text_offset, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return image


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def save_annotated_video(
    frames: List[np.ndarray],
    results: List[KeypointResult],
    output_path: Union[str, Path],
    fps: float,
    config: Optional[PoseEstimationConfig] = None,
) -> None:
    """Save video with skeleton annotations."""
    from src.video_processor import VideoWriter
    
    config = config or PoseEstimationConfig()
    visualizer = SkeletonVisualizer(config)
    
    h, w = frames[0].shape[:2]
    
    with VideoWriter(output_path, fps, (w, h)) as writer:
        for frame, result in zip(frames, results):
            annotated = visualizer.draw_multi_person(frame, result)
            writer.write(annotated)


# ==============================================================================
# MODULE TEST
# ==============================================================================

if __name__ == "__main__":
    print("Testing Visualizer module...\n")
    
    # Create dummy data
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    
    # Rough human shape keypoints
    keypoints = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 150], [360, 150], [250, 220], [390, 220],
        [230, 290], [410, 290], [290, 280], [350, 280],
        [280, 360], [360, 360], [275, 440], [365, 440],
    ], dtype=np.float32)
    
    scores = np.ones(17) * 0.9
    
    # Test visualizer
    viz = SkeletonVisualizer()
    annotated = viz.draw(frame, keypoints, scores)
    
    output_dir = Path("/home/claude/yolo_pose_estimation/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / "test_skeleton.jpg"), annotated)
    print(f"Saved test image to {output_dir / 'test_skeleton.jpg'}")
    
    print("\n✓ Module test complete")
