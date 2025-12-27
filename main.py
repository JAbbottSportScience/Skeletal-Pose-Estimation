"""
================================================================================
SPRINT POSE ESTIMATION PIPELINE - MAIN RUNNER (YOLO-Pose)
================================================================================
Main entry point for sprint pose estimation and kinogram generation.

Usage:
    # Process single video
    python main.py --video path/to/sprint.mp4
    
    # Process all videos in directory  
    python main.py --input-dir ./videos/
    
    # Use fast preset
    python main.py --preset fast --video sprint.mp4
    
    # Generate kinogram with specific phases
    python main.py --video sprint.mp4 --phases 10,25,40,55,70

Author: [Your Name]
Target Hardware: Apple Silicon M4 (MPS Backend)
Backend: Ultralytics YOLO-Pose
================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Union, Optional

# ==============================================================================
# Add project root to path
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    PoseEstimationConfig,
    get_fast_config,
    get_accurate_config,
    get_kinogram_config,
)
from src.pose_estimator import PoseEstimator, KeypointResult, create_estimator
from src.video_processor import VideoProcessor, FrameData
from src.visualizer import SkeletonVisualizer, KinogramGenerator, save_annotated_video


# ==============================================================================
# MODEL DOWNLOAD CHECK
# ==============================================================================

def check_model_exists(model_name: str = "yolov8x-pose.pt") -> bool:
    """Check if model exists, print helpful message if not."""
    model_path = Path(model_name)
    
    if not model_path.exists():
        print("\n" + "=" * 60)
        print("MODEL DOWNLOAD REQUIRED")
        print("=" * 60)
        print(f"  Model '{model_name}' not found locally.")
        print("  Ultralytics will now download it automatically.")
        print("  This is a one-time download (~133 MB for yolov8x-pose).")
        print("=" * 60 + "\n")
        return False
    return True


# ==============================================================================
# MAIN PIPELINE CLASS
# ==============================================================================

class SprintAnalysisPipeline:
    """
    Main pipeline for sprint pose estimation and analysis.
    
    Example:
        >>> pipeline = SprintAnalysisPipeline()
        >>> pipeline.process_video("sprint.mp4")
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
        
        # Lazy-loaded components
        self._estimator = None
        self._video_processor = None
        self._visualizer = None
        self._kinogram_generator = None
        
        # Setup output directories
        self.config.output.setup_directories()
    
    @property
    def estimator(self):
        if self._estimator is None:
            # Check if model needs to be downloaded
            check_model_exists(self.config.model.model_name)
            self._estimator = create_estimator(self.config)
        return self._estimator
    
    @property
    def video_processor(self):
        if self._video_processor is None:
            self._video_processor = VideoProcessor(self.config)
        return self._video_processor
    
    @property
    def visualizer(self):
        if self._visualizer is None:
            self._visualizer = SkeletonVisualizer(self.config)
        return self._visualizer
    
    @property
    def kinogram_generator(self):
        if self._kinogram_generator is None:
            self._kinogram_generator = KinogramGenerator(self.config)
        return self._kinogram_generator
    
    def process_video(
        self,
        video_path: Union[str, Path],
        generate_kinogram: bool = True,
        save_video: bool = False,
        phase_indices: Optional[List[int]] = None,
        use_yolo_video: bool = True,
    ) -> dict:
        """
        Process a single video through the full pipeline.
        
        Args:
            video_path: Path to input video
            generate_kinogram: Whether to generate kinogram image
            save_video: Whether to save annotated video
            phase_indices: Specific frames for kinogram (auto-detect if None)
            use_yolo_video: Use YOLO's built-in video processing (faster)
        
        Returns:
            Dictionary with processing results and statistics
        """
        video_path = Path(video_path)
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print(f"PROCESSING: {video_path.name}")
        print("=" * 60)
        
        # -----------------------------------------------------------------------
        # Step 1: Get video metadata
        # -----------------------------------------------------------------------
        print("\n[Step 1/4] Loading video metadata...")
        metadata = self.video_processor.get_metadata(video_path)
        print(metadata)
        
        # -----------------------------------------------------------------------
        # Step 2: Run pose estimation
        # -----------------------------------------------------------------------
        print("\n[Step 2/4] Running pose estimation...")
        
        frames: List[FrameData] = []
        results: List[KeypointResult] = []
        
        if use_yolo_video:
            # Use YOLO's built-in video processing (faster, supports tracking)
            for result in self.estimator.estimate_video(video_path):
                results.append(result)
        else:
            # Use custom frame iteration (more control)
            for frame_data in self.video_processor.iter_frames(video_path):
                frames.append(frame_data)
                result = self.estimator.estimate_single(
                    frame_data.frame,
                    frame_idx=frame_data.frame_idx,
                    timestamp=frame_data.timestamp,
                )
                results.append(result)
        
        print(f"  ✓ Processed {len(results)} frames")
        
        # -----------------------------------------------------------------------
        # Step 3: Save keypoint data
        # -----------------------------------------------------------------------
        print("\n[Step 3/4] Saving keypoint data...")
        
        keypoints_path = (
            self.config.output.output_dir 
            / "keypoints" 
            / f"{video_path.stem}_keypoints.json"
        )
        
        self._save_keypoints(results, keypoints_path)
        print(f"  ✓ Saved: {keypoints_path}")
        
        # -----------------------------------------------------------------------
        # Step 4: Generate kinogram (needs frames)
        # -----------------------------------------------------------------------
        kinogram_path = None
        
        if generate_kinogram:
            print("\n[Step 4/4] Generating kinogram...")
            
            # Need to reload frames for kinogram generation
            if not frames:
                print("  Loading frames for kinogram...")
                for frame_data in self.video_processor.iter_frames(video_path):
                    frames.append(frame_data)
            
            kinogram_path = (
                self.config.output.output_dir 
                / "kinograms" 
                / f"{video_path.stem}_kinogram.{self.config.output.kinogram_format}"
            )
            
            import cv2
            frame_arrays = [f.frame for f in frames]
            
            # Align results to frames
            aligned_results = self._align_results_to_frames(results, frames)
            
            kinogram = self.kinogram_generator.generate(
                frame_arrays, aligned_results, phase_indices
            )
            cv2.imwrite(str(kinogram_path), kinogram)
            print(f"  ✓ Saved: {kinogram_path}")
        else:
            print("\n[Step 4/4] Skipping kinogram generation")
        
        # -----------------------------------------------------------------------
        # Results summary
        # -----------------------------------------------------------------------
        elapsed = time.time() - start_time
        avg_fps = len(results) / elapsed if elapsed > 0 else 0
        
        result_summary = {
            "video": str(video_path),
            "num_frames": len(results),
            "duration_seconds": metadata.duration,
            "processing_time_seconds": elapsed,
            "avg_fps": avg_fps,
            "keypoints_path": str(keypoints_path),
            "kinogram_path": str(kinogram_path) if kinogram_path else None,
        }
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Frames: {len(results)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: {avg_fps:.1f} FPS")
        print("=" * 60 + "\n")
        
        return result_summary
    
    def _save_keypoints(self, results: List[KeypointResult], output_path: Path) -> None:
        """Save keypoint data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "format": "coco",
            "num_keypoints": 17,
            "keypoint_names": list(self.config.keypoints.keypoint_names),
            "model": self.config.model.model_name,
            "frames": [r.to_dict() for r in results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _align_results_to_frames(
        self,
        results: List[KeypointResult],
        frames: List[FrameData],
    ) -> List[KeypointResult]:
        """Align results to frame indices when they might not match."""
        # Build lookup by frame index
        results_by_idx = {r.frame_idx: r for r in results}
        
        aligned = []
        for frame_data in frames:
            if frame_data.frame_idx in results_by_idx:
                aligned.append(results_by_idx[frame_data.frame_idx])
            else:
                # No result for this frame, create empty
                import numpy as np
                aligned.append(KeypointResult(
                    frame_idx=frame_data.frame_idx,
                    timestamp=frame_data.timestamp,
                    keypoints=np.empty((0, 17, 2)),
                    scores=np.empty((0, 17)),
                    bboxes=np.empty((0, 4)),
                    bbox_scores=np.empty((0,)),
                ))
        
        return aligned
    
    def process_all_videos(self, input_dir: Optional[Union[str, Path]] = None, **kwargs) -> List[dict]:
        """Process all videos in a directory."""
        videos = self.video_processor.find_videos(input_dir)
        
        if not videos:
            print("No videos found")
            return []
        
        all_results = []
        for i, video_path in enumerate(videos, 1):
            print(f"\n[Video {i}/{len(videos)}]")
            result = self.process_video(video_path, **kwargs)
            all_results.append(result)
        
        return all_results


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sprint Pose Estimation Pipeline (YOLO-Pose)",
    )
    
    parser.add_argument("--video", "-v", type=str, help="Path to video file")
    parser.add_argument("--input-dir", "-i", type=str, help="Directory with videos")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument(
        "--preset", "-p",
        choices=["default", "fast", "accurate", "kinogram"],
        default="default",
        help="Configuration preset"
    )
    parser.add_argument("--phases", type=str, help="Comma-separated frame indices")
    parser.add_argument("--no-kinogram", action="store_true", help="Skip kinogram")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def get_config_for_preset(preset: str) -> PoseEstimationConfig:
    presets = {
        "default": PoseEstimationConfig,
        "fast": get_fast_config,
        "accurate": get_accurate_config,
        "kinogram": get_kinogram_config,
    }
    return presets[preset]()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SPRINT POSE ESTIMATION PIPELINE (YOLO-Pose)")
    print("=" * 60)
    
    # Load configuration
    config = get_config_for_preset(args.preset)
    
    if args.output_dir:
        config.output.output_dir = Path(args.output_dir)
    
    if args.input_dir:
        config.video.input_dir = Path(args.input_dir)
    
    if args.verbose:
        config.print_summary()
    
    # Create pipeline
    pipeline = SprintAnalysisPipeline(config)
    
    # Parse phase indices
    phase_indices = None
    if args.phases:
        phase_indices = [int(x.strip()) for x in args.phases.split(",")]
    
    # Process
    if args.video:
        pipeline.process_video(
            args.video,
            generate_kinogram=not args.no_kinogram,
            save_video=args.save_video,
            phase_indices=phase_indices,
        )
    else:
        pipeline.process_all_videos(
            generate_kinogram=not args.no_kinogram,
            save_video=args.save_video,
        )
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()