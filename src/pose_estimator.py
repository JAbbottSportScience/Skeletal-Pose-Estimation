"""
================================================================================
YOLO-POSE ESTIMATOR MODULE (WITH SMOOTHING)
================================================================================
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ Ultralytics not found. Install with: pip install ultralytics")

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PoseEstimationConfig
from src.smoothing import OneEuroFilter


@dataclass
class KeypointResult:
    """Container for keypoint estimation results from a single frame."""
    frame_idx: int
    timestamp: Optional[float]
    keypoints: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    bbox_scores: np.ndarray
    track_ids: Optional[np.ndarray] = None
    
    @property
    def num_people(self) -> int:
        return len(self.bboxes)
    
    def get_primary_person(self, by: str = "confidence") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.num_people == 0:
            return None
        
        if by == "confidence":
            primary_idx = np.argmax(self.bbox_scores)
        elif by == "area":
            areas = (self.bboxes[:, 2] - self.bboxes[:, 0]) * \
                    (self.bboxes[:, 3] - self.bboxes[:, 1])
            primary_idx = np.argmax(areas)
        else:
            primary_idx = 0
        
        return self.keypoints[primary_idx], self.scores[primary_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "num_people": self.num_people,
            "keypoints": self.keypoints.tolist(),
            "scores": self.scores.tolist(),
            "bboxes": self.bboxes.tolist(),
            "bbox_scores": self.bbox_scores.tolist(),
        }
        if self.track_ids is not None:
            result["track_ids"] = self.track_ids.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeypointResult":
        return cls(
            frame_idx=data["frame_idx"],
            timestamp=data.get("timestamp"),
            keypoints=np.array(data["keypoints"]),
            scores=np.array(data["scores"]),
            bboxes=np.array(data["bboxes"]),
            bbox_scores=np.array(data["bbox_scores"]),
            track_ids=np.array(data["track_ids"]) if "track_ids" in data else None,
        )


class PoseEstimator:
    """
    YOLO-Pose estimator with temporal smoothing.
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
        self.device: Optional[str] = None
        self.model: Optional[YOLO] = None
        self.smoother: Optional[OneEuroFilter] = None
        self.initialized = False
        self._inference_times: List[float] = []
    
    def initialize(self) -> None:
        """Load YOLO model and initialize smoother."""
        print("\n" + "=" * 60)
        print("INITIALIZING YOLO-POSE ESTIMATOR")
        print("=" * 60)
        
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
        
        # Select device
        print("\n[Step 1/3] Selecting compute device...")
        self.device = self.config.device.get_device()
        
        # Load model
        print(f"\n[Step 2/3] Loading YOLO-Pose model...")
        print(f"  Model: {self.config.model.model_name}")
        self.model = YOLO(self.config.model.model_name)
        print(f"  ✓ Model loaded successfully")
        
        # Initialize smoother
        print(f"\n[Step 3/3] Initializing smoother...")
        if hasattr(self.config, 'smoothing') and self.config.smoothing.enabled:
            self.smoother = OneEuroFilter(
                min_cutoff=self.config.smoothing.min_cutoff,
                beta=self.config.smoothing.beta,
                d_cutoff=self.config.smoothing.d_cutoff,
            )
            print(f"  ✓ Smoothing enabled (min_cutoff={self.config.smoothing.min_cutoff}, beta={self.config.smoothing.beta})")
        else:
            # Default smoother if config not updated
            self.smoother = OneEuroFilter(min_cutoff=0.3, beta=0.1)
            print(f"  ✓ Smoothing enabled (default settings)")
        
        # Warmup
        self._warmup()
        
        self.initialized = True
        print("\n✓ YOLO-Pose Estimator initialized!")
        print("=" * 60 + "\n")
    
    def _warmup(self) -> None:
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self.model(dummy_img, device=self.device, verbose=False)
            print("  ✓ Warmup complete")
        except Exception as e:
            print(f"  ⚠ Warmup failed: {e}")
    
    def estimate_single(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: Optional[float] = None,
        apply_smoothing: bool = True,
    ) -> KeypointResult:
        """Estimate poses for a single frame with optional smoothing."""
        if not self.initialized:
            raise RuntimeError("Estimator not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        results: List[Results] = self.model(
            frame,
            device=self.device,
            conf=self.config.model.confidence_threshold,
            iou=self.config.model.iou_threshold,
            verbose=False,
        )
        
        result = results[0]
        
        if result.keypoints is None or len(result.keypoints) == 0:
            return KeypointResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                keypoints=np.empty((0, 17, 2)),
                scores=np.empty((0, 17)),
                bboxes=np.empty((0, 4)),
                bbox_scores=np.empty((0,)),
            )
        
        keypoints = result.keypoints.xy.cpu().numpy()
        kp_scores = result.keypoints.conf.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()
        bbox_scores = result.boxes.conf.cpu().numpy()
        
        # Apply smoothing
        if apply_smoothing and self.smoother is not None and len(keypoints) > 0:
            keypoints = self.smoother.smooth(keypoints, kp_scores)
        
        elapsed = time.time() - start_time
        self._inference_times.append(elapsed)
        
        return KeypointResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=keypoints,
            scores=kp_scores,
            bboxes=bboxes,
            bbox_scores=bbox_scores,
        )
    
    def estimate_video(
        self,
        video_path: Union[str, Path],
        show_progress: bool = True,
        apply_smoothing: bool = True,
    ) -> Generator[KeypointResult, None, None]:
        """Process video with tracking and smoothing."""
        if not self.initialized:
            raise RuntimeError("Estimator not initialized. Call initialize() first.")
        
        video_path = Path(video_path)
        print(f"\nProcessing video: {video_path.name}")
        
        # Reset smoother for new video
        if self.smoother is not None:
            self.smoother.reset()
        
        # Use tracking
        if self.config.model.enable_tracking:
            results_generator = self.model.track(
                source=str(video_path),
                device=self.device,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                stream=True,
                persist=True,
                tracker=self.config.model.tracker,
                verbose=False,
            )
        else:
            results_generator = self.model(
                source=str(video_path),
                device=self.device,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                stream=True,
                verbose=False,
            )
        
        frame_idx = 0
        
        for result in results_generator:
            if result.keypoints is None or len(result.keypoints) == 0:
                yield KeypointResult(
                    frame_idx=frame_idx,
                    timestamp=None,
                    keypoints=np.empty((0, 17, 2)),
                    scores=np.empty((0, 17)),
                    bboxes=np.empty((0, 4)),
                    bbox_scores=np.empty((0,)),
                    track_ids=None,
                )
            else:
                keypoints = result.keypoints.xy.cpu().numpy()
                kp_scores = result.keypoints.conf.cpu().numpy()
                bboxes = result.boxes.xyxy.cpu().numpy()
                bbox_scores = result.boxes.conf.cpu().numpy()
                
                # Apply smoothing
                if apply_smoothing and self.smoother is not None:
                    keypoints = self.smoother.smooth(keypoints, kp_scores)
                
                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                yield KeypointResult(
                    frame_idx=frame_idx,
                    timestamp=None,
                    keypoints=keypoints,
                    scores=kp_scores,
                    bboxes=bboxes,
                    bbox_scores=bbox_scores,
                    track_ids=track_ids,
                )
            
            if show_progress and frame_idx > 0 and frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")
            
            frame_idx += 1
        
        print(f"  ✓ Processed {frame_idx} frames total")
    
    def estimate_batch(
        self,
        frames: List[np.ndarray],
        start_idx: int = 0,
        show_progress: bool = True,
    ) -> List[KeypointResult]:
        """Estimate poses for multiple frames."""
        if not self.initialized:
            raise RuntimeError("Estimator not initialized.")
        
        # Reset smoother for new batch
        if self.smoother is not None:
            self.smoother.reset()
        
        results = []
        for i, frame in enumerate(frames):
            result = self.estimate_single(frame, frame_idx=start_idx + i)
            results.append(result)
            
            if show_progress and (i + 1) % 50 == 0:
                fps = self.get_average_fps()
                print(f"  Processed {i + 1}/{len(frames)} frames ({fps:.1f} FPS)")
        
        return results
    
    def get_average_fps(self) -> float:
        if not self._inference_times:
            return 0.0
        avg_time = sum(self._inference_times) / len(self._inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def reset_timing_stats(self) -> None:
        self._inference_times.clear()


def create_estimator(config: Optional[PoseEstimationConfig] = None) -> PoseEstimator:
    """Create and initialize a pose estimator."""
    estimator = PoseEstimator(config)
    estimator.initialize()
    return estimator


if __name__ == "__main__":
    print("Testing YOLO PoseEstimator with smoothing...\n")
    
    config = PoseEstimationConfig()
    
    if YOLO_AVAILABLE:
        estimator = PoseEstimator(config)
        estimator.initialize()
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_single(dummy_frame)
        
        print(f"Result: {result.num_people} people detected")
        print(f"Smoothing: enabled")
    
    print("\n✓ Module test complete")
