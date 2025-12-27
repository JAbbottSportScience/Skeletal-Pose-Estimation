"""
================================================================================
VIDEO PROCESSOR MODULE
================================================================================
Handles video loading, frame extraction, and preprocessing.

Note: YOLO has built-in video processing via model.track() which is often
preferred. This module is useful when you need custom frame handling,
preprocessing, or want to integrate with other tools.

Author: [Your Name]
================================================================================
"""

from pathlib import Path
from typing import Union,  Iterator, List, Optional, Tuple, Generator
from dataclasses import dataclass

import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PoseEstimationConfig


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class VideoMetadata:
    """Container for video file metadata."""
    filepath: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    
    def __str__(self) -> str:
        return (
            f"Video: {self.filepath.name}\n"
            f"  Resolution: {self.width}x{self.height}\n"
            f"  FPS: {self.fps:.2f}\n"
            f"  Duration: {self.duration:.2f}s ({self.frame_count} frames)\n"
            f"  Codec: {self.codec}"
        )


@dataclass
class FrameData:
    """Container for a single extracted frame."""
    frame: np.ndarray
    frame_idx: int
    timestamp: float
    original_size: Tuple[int, int]  # (width, height)


# ==============================================================================
# VIDEO PROCESSOR CLASS
# ==============================================================================

class VideoProcessor:
    """
    Handles video loading and frame extraction.
    
    Example:
        >>> processor = VideoProcessor()
        >>> 
        >>> for frame_data in processor.iter_frames("sprint.mp4"):
        ...     result = estimator.estimate_single(frame_data.frame)
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
    
    def get_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """Extract metadata from a video file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            
            return VideoMetadata(
                filepath=video_path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
            )
        finally:
            cap.release()
    
    def iter_frames(
        self,
        video_path: Union[str, Path],
        skip_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Generator[FrameData, None, None]:
        """
        Iterate through video frames as a generator.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame (uses config if None)
            max_frames: Maximum frames to yield
            resize: Target (width, height) or None
        
        Yields:
            FrameData objects containing frame and metadata
        """
        video_path = Path(video_path)
        skip_frames = skip_frames or self.config.video.skip_frames
        
        metadata = self.get_metadata(video_path)
        print(f"\nProcessing: {metadata.filepath.name}")
        print(f"  {metadata.width}x{metadata.height} @ {metadata.fps:.1f} FPS")
        
        # Calculate resize if needed
        if resize is None and self.config.video.max_dimension:
            resize = self._calculate_resize(
                metadata.width, metadata.height,
                self.config.video.max_dimension
            )
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            frame_idx = 0
            frames_yielded = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                original_size = (frame.shape[1], frame.shape[0])
                
                # Resize if needed
                if resize and resize != original_size:
                    interp = cv2.INTER_AREA if resize[0] < original_size[0] else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, resize, interpolation=interp)
                
                yield FrameData(
                    frame=frame,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    original_size=original_size,
                )
                
                frames_yielded += 1
                frame_idx += 1
                
                if max_frames and frames_yielded >= max_frames:
                    break
                    
        finally:
            cap.release()
            print(f"  Yielded {frames_yielded} frames")
    
    def get_frame_range(
        self,
        video_path: Union[str, Path],
        start: int,
        end: int,
    ) -> List[FrameData]:
        """Extract a specific range of frames."""
        video_path = Path(video_path)
        metadata = self.get_metadata(video_path)
        
        start = max(0, start)
        end = min(metadata.frame_count, end)
        
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        frames = []
        
        try:
            for frame_idx in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                original_size = (frame.shape[1], frame.shape[0])
                
                frames.append(FrameData(
                    frame=frame,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    original_size=original_size,
                ))
        finally:
            cap.release()
        
        return frames
    
    def _calculate_resize(
        self,
        width: int,
        height: int,
        max_dim: int,
    ) -> Tuple[int, int]:
        """Calculate resize dimensions maintaining aspect ratio."""
        if max(width, height) <= max_dim:
            return (width, height)
        
        scale = max_dim / max(width, height)
        new_width = int(width * scale) - (int(width * scale) % 2)
        new_height = int(height * scale) - (int(height * scale) % 2)
        
        return (new_width, new_height)
    
    def find_videos(self, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """Find all supported video files in a directory."""
        directory = Path(directory or self.config.video.input_dir)
        
        if not directory.exists():
            return []
        
        videos = []
        for ext in self.config.video.supported_formats:
            videos.extend(directory.glob(f"*{ext}"))
            videos.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(set(videos))


# ==============================================================================
# VIDEO WRITER CLASS
# ==============================================================================

class VideoWriter:
    """Writes processed frames to output video."""
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        self._frames_written = 0
    
    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to video."""
        self._writer.write(frame)
        self._frames_written += 1
    
    def close(self) -> None:
        """Finalize and close the video file."""
        self._writer.release()
        print(f"Saved video: {self.output_path} ({self._frames_written} frames)")
    
    def __enter__(self) -> "VideoWriter":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


# ==============================================================================
# MODULE TEST
# ==============================================================================

if __name__ == "__main__":
    print("Testing VideoProcessor module...\n")
    
    processor = VideoProcessor()
    
    # Create a test video
    test_path = Path("/tmp/test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_path), fourcc, 30, (640, 480))
    
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + i * 15, 240), 50, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    
    # Test metadata
    metadata = processor.get_metadata(test_path)
    print(metadata)
    
    # Test frame iteration
    for frame_data in processor.iter_frames(test_path, max_frames=5):
        print(f"  Frame {frame_data.frame_idx}: {frame_data.frame.shape}")
    
    test_path.unlink()
    print("\n✓ Module test complete")
