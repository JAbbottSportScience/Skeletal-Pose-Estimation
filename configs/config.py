"""
================================================================================
YOLO-POSE ESTIMATION CONFIGURATION
================================================================================
This configuration file centralizes all settings for the pose estimation pipeline.
Modify these values to customize behavior without touching core logic.

Author: [Your Name]
Project: Sprint Pose Estimation & Kinogram Generation
Target Hardware: Apple Silicon M4 (MPS Backend)
Backend: Ultralytics YOLO-Pose (single-stage pose estimation)

WHY YOLO-POSE?
    - Single pip install (no dependency hell)
    - Native MPS (Apple Silicon) support
    - Single-stage detection + pose (faster than two-stage)
    - Built-in video processing and tracking
    - Active development and community

CONFIGURATION SECTIONS:
    1. Device Settings - Hardware acceleration options
    2. Model Settings - YOLO-Pose model selection
    3. Video Processing - Frame extraction and handling
    4. Keypoint Settings - COCO format keypoint definitions
    5. Kinogram Settings - Composite image generation
    6. Output Settings - File paths and formats
    7. Visualization Settings - Drawing and display options
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import torch


# ==============================================================================
# SECTION 1: DEVICE SETTINGS
# ==============================================================================
# Apple Silicon M4 uses Metal Performance Shaders (MPS) for GPU acceleration.
# YOLO-Pose has first-class MPS support - just set device='mps'.
#
# Device Priority:
#   1. MPS (Apple Silicon GPU) - Fastest on M4, native support
#   2. CUDA (NVIDIA GPU) - For non-Mac systems
#   3. CPU - Fallback, slowest but always available
# ==============================================================================

@dataclass
class DeviceConfig:
    """
    Hardware device configuration for YOLO model inference.
    
    YOLO-Pose MPS Support:
        Ultralytics has excellent Apple Silicon support. Simply set
        device='mps' and PyTorch handles the rest via Metal Performance
        Shaders. No special configuration needed.
    
    Attributes:
        preferred_device: Target device string ('mps', 'cuda', 'cpu', or device id)
        fallback_to_cpu: If True, falls back to CPU on device errors
        half_precision: Enable FP16 for faster inference (good on MPS)
    
    Example:
        >>> config = DeviceConfig()
        >>> device = config.get_device()
        >>> print(device)  # 'mps' on M4 MacBook
    """
    preferred_device: str = "mps"
    fallback_to_cpu: bool = True
    half_precision: bool = False  # FP16 - works well on MPS, use True for speed
    
    def get_device(self) -> str:
        """
        Determines the best available device for inference.
        
        YOLO Device Selection:
            - 'mps': Apple Silicon GPU (M1/M2/M3/M4)
            - 'cuda' or '0': NVIDIA GPU
            - 'cpu': CPU fallback
        
        Returns:
            str: Device string compatible with Ultralytics YOLO
        
        Raises:
            RuntimeError: If preferred device unavailable and fallback disabled
        """
        # -----------------------------------------------------------------------
        # MPS Availability Check (Apple Silicon)
        # -----------------------------------------------------------------------
        if self.preferred_device == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("✓ MPS (Metal Performance Shaders) backend available")
                print("  → Using Apple Silicon GPU acceleration")
                return "mps"
            
            if self.fallback_to_cpu:
                print("⚠ MPS unavailable, falling back to CPU")
                return "cpu"
            raise RuntimeError("MPS device requested but not available")
        
        # -----------------------------------------------------------------------
        # CUDA Availability Check (NVIDIA)
        # -----------------------------------------------------------------------
        if self.preferred_device in ("cuda", "0"):
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA available: {gpu_name}")
                return "cuda"
            
            if self.fallback_to_cpu:
                print("⚠ CUDA unavailable, falling back to CPU")
                return "cpu"
            raise RuntimeError("CUDA device requested but not available")
        
        # -----------------------------------------------------------------------
        # CPU Fallback
        # -----------------------------------------------------------------------
        print("→ Using CPU for inference")
        return "cpu"


# ==============================================================================
# SECTION 2: MODEL SETTINGS
# ==============================================================================
# YOLO-Pose models come in different sizes with speed/accuracy tradeoffs.
#
# Model Variants (COCO-trained, 17 keypoints):
#   - yolov8n-pose: Nano - fastest, lowest accuracy (~3ms/frame on MPS)
#   - yolov8s-pose: Small - fast, good accuracy (~5ms/frame)
#   - yolov8m-pose: Medium - balanced (~8ms/frame) [RECOMMENDED]
#   - yolov8l-pose: Large - slower, high accuracy (~12ms/frame)
#   - yolov8x-pose: XLarge - slowest, highest accuracy (~18ms/frame)
#
# For sprint analysis, yolov8m-pose offers the best balance.
# ==============================================================================

@dataclass
class ModelConfig:
    """
    YOLO-Pose model configuration.
    
    Unlike two-stage detectors (MMPose), YOLO-Pose performs detection
    and pose estimation in a single forward pass. This makes it:
        - Faster (no detection → crop → pose pipeline)
        - Simpler (single model file)
        - Better at handling motion blur (unified architecture)
    
    Attributes:
        model_name: YOLO-Pose model variant
        confidence_threshold: Minimum detection confidence (0-1)
        iou_threshold: IoU threshold for NMS (non-max suppression)
        max_detections: Maximum people to detect per frame
    
    Model Selection Guide:
        - Quick testing: yolov8n-pose (fastest)
        - Production: yolov8m-pose (balanced)
        - Maximum accuracy: yolov8l-pose or yolov8x-pose
    """
    
    # ---------------------------------------------------------------------------
    # Model Selection
    # ---------------------------------------------------------------------------
    # Model weights are auto-downloaded on first use to ~/.cache/ultralytics/
    # You can also specify a local path: model_name = "./models/yolov8m-pose.pt"
    # ---------------------------------------------------------------------------
    model_name: str = "yolov8x-pose.pt"
    
    # ---------------------------------------------------------------------------
    # Detection Parameters
    # ---------------------------------------------------------------------------
    # confidence_threshold: Minimum confidence to keep a detection
    #   - Lower (0.25): Keep more detections, may include false positives
    #   - Higher (0.7): Only high-confidence, may miss partially visible people
    #   - Recommended for sprint: 0.5 (single athlete, clear view)
    #
    # iou_threshold: Non-Maximum Suppression overlap threshold
    #   - Controls how much bounding box overlap is tolerated
    #   - Lower (0.3): Aggressive suppression, fewer overlapping boxes
    #   - Higher (0.7): Allow more overlap
    #   - For single athlete: 0.45 is fine
    # ---------------------------------------------------------------------------
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.7
    
    # ---------------------------------------------------------------------------
    # Detection Limits
    # ---------------------------------------------------------------------------
    # max_detections: Maximum number of people to detect per frame
    #   - For sprint analysis with single athlete: 1-3 is plenty
    #   - For crowd scenes: increase as needed
    # ---------------------------------------------------------------------------
    max_detections: int = 1
    
    # ---------------------------------------------------------------------------
    # Tracking (for video processing)
    # ---------------------------------------------------------------------------
    # YOLO has built-in tracking (BoT-SORT, ByteTrack)
    # This maintains consistent person IDs across frames
    # ---------------------------------------------------------------------------
    enable_tracking: bool = True
    tracker: str = "botsort.yaml"  # or "bytetrack.yaml"


# ==============================================================================
# SECTION 3: VIDEO PROCESSING SETTINGS
# ==============================================================================
# Configuration for video input, frame extraction, and preprocessing.
#
# Key Considerations for Sprint Analysis:
#   - Frame rate: 60+ fps captures stride phases well
#   - Resolution: 720p-1080p is sufficient for pose estimation
#   - View: Side/sagittal view perpendicular to running direction is ideal
# ==============================================================================

@dataclass
class VideoConfig:
    """
    Video processing configuration.
    
    Attributes:
        input_dir: Directory containing source videos
        supported_formats: Video file extensions to process
        target_fps: Resample videos to this frame rate (None = keep original)
        max_dimension: Resize frames if larger (None = no resize)
        skip_frames: Process every Nth frame (1 = all frames)
    
    Frame Rate Guidance:
        - 30 fps: May miss key phase transitions
        - 60 fps: Good for general analysis
        - 120+ fps: Ideal for detailed biomechanics
    """
    
    input_dir: Path = field(default_factory=lambda: Path("./videos"))
    
    supported_formats: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    
    # ---------------------------------------------------------------------------
    # Frame Rate Handling
    # ---------------------------------------------------------------------------
    # target_fps=None: Keep original frame rate
    # target_fps=60: Resample to 60fps
    # ---------------------------------------------------------------------------
    target_fps: Optional[int] = None
    
    # ---------------------------------------------------------------------------
    # Resolution Handling
    # ---------------------------------------------------------------------------
    # YOLO models handle various resolutions automatically.
    # Internally, they resize to 640x640 for inference.
    # max_dimension helps reduce memory/processing for 4K videos.
    # ---------------------------------------------------------------------------
    max_dimension: Optional[int] = None
    
    # ---------------------------------------------------------------------------
    # Frame Skipping
    # ---------------------------------------------------------------------------
    # skip_frames: Process every Nth frame
    #   - 1: Every frame (full analysis)
    #   - 2: Every other frame (2x faster)
    #   - Use higher values for quick testing
    # ---------------------------------------------------------------------------
    skip_frames: int = 2


# ==============================================================================
# SECTION 4: KEYPOINT DEFINITIONS (COCO FORMAT)
# ==============================================================================
# YOLO-Pose outputs 17 keypoints in COCO format.
# This matches MMPose, MediaPipe (roughly), and most pose models.
#
# COCO 17-Keypoint Layout:
#   0: nose          5: left_shoulder   10: right_wrist   15: left_ankle
#   1: left_eye      6: right_shoulder  11: left_hip      16: right_ankle
#   2: right_eye     7: left_elbow      12: right_hip
#   3: left_ear      8: right_elbow     13: left_knee
#   4: right_ear     9: left_wrist      14: right_knee
# ==============================================================================

@dataclass
class KeypointConfig:
    """
    Keypoint format and skeleton definition.
    
    YOLO-Pose outputs COCO format with 17 keypoints.
    This is standard across most pose estimation models.
    
    For sprint biomechanics, the key joints are:
        - Shoulders (5, 6): Arm swing, trunk rotation
        - Hips (11, 12): Pelvis position, stride mechanics
        - Knees (13, 14): Knee drive, ground contact preparation
        - Ankles (15, 16): Foot strike, toe-off
    """
    
    format: str = "coco"
    num_keypoints: int = 17
    
    # ---------------------------------------------------------------------------
    # Keypoint Names (indexed 0-16)
    # ---------------------------------------------------------------------------
    keypoint_names: Tuple[str, ...] = (
        "nose",           # 0
        "left_eye",       # 1
        "right_eye",      # 2
        "left_ear",       # 3
        "right_ear",      # 4
        "left_shoulder",  # 5  - KEY for biomechanics
        "right_shoulder", # 6  - KEY for biomechanics
        "left_elbow",     # 7
        "right_elbow",    # 8
        "left_wrist",     # 9
        "right_wrist",    # 10
        "left_hip",       # 11 - KEY for biomechanics
        "right_hip",      # 12 - KEY for biomechanics
        "left_knee",      # 13 - KEY for biomechanics
        "right_knee",     # 14 - KEY for biomechanics
        "left_ankle",     # 15 - KEY for biomechanics
        "right_ankle",    # 16 - KEY for biomechanics
    )
    
    # ---------------------------------------------------------------------------
    # Skeleton Connections
    # Each tuple is (start_keypoint_idx, end_keypoint_idx)
    # ---------------------------------------------------------------------------
    skeleton: Tuple[Tuple[int, int], ...] = (
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Left arm
        (5, 7), (7, 9),
        # Right arm
        (6, 8), (8, 10),
        # Left leg
        (11, 13), (13, 15),
        # Right leg
        (12, 14), (14, 16),
    )
    
    # ---------------------------------------------------------------------------
    # Biomechanically Important Indices
    # ---------------------------------------------------------------------------
    biomech_keypoints: Tuple[int, ...] = (5, 6, 11, 12, 13, 14, 15, 16)
    
    def get_keypoint_index(self, name: str) -> int:
        """Get index for a keypoint by name."""
        return self.keypoint_names.index(name)


# ==============================================================================
# SECTION 5: KINOGRAM SETTINGS
# ==============================================================================
# Kinograms show multiple poses overlaid to visualize the full gait cycle.
# ==============================================================================

@dataclass
class KinogramConfig:
    """
    Kinogram generation configuration.
    
    A kinogram overlays multiple phases of motion on a single image.
    Essential for sprint coaching and biomechanical analysis.
    
    Attributes:
        num_phases: Number of poses to include (5-8 typical)
        phase_detection_method: 'auto' (biomechanical) or 'uniform' (even spacing)
        background_mode: 'first', 'average', 'black', 'white'
        overlay_opacity: Transparency of overlaid poses (0-1)
    """
    
    num_phases: int = 35
    phase_detection_method: str = "auto"  # 'auto', 'uniform', 'manual'
    background_mode: str = "first"        # 'first', 'average', 'black', 'white'
    overlay_opacity: float = 0.75
    output_width: int = 1920
    output_height: Optional[int] = None   # None = maintain aspect ratio


# ==============================================================================
# SECTION 6: OUTPUT SETTINGS
# ==============================================================================

@dataclass
class OutputConfig:
    """Output file configuration."""
    
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    save_keypoints: bool = True
    keypoint_format: str = "json"  # 'json', 'csv'
    save_visualizations: bool = True
    save_kinograms: bool = True
    save_video: bool = False
    visualization_format: str = "jpg"
    kinogram_format: str = "png"
    video_codec: str = "mp4v"
    
    def setup_directories(self) -> None:
        """Create output directory structure."""
        for subdir in ["keypoints", "visualizations", "kinograms", "videos"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SECTION 7: VISUALIZATION SETTINGS
# ==============================================================================

@dataclass
class VisualizationConfig:
    """Visualization and drawing configuration."""
    
    keypoint_radius: int = 4
    keypoint_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow BGR
    skeleton_thickness: int = 2
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)    # Green BGR
    confidence_color_map: bool = True
    draw_labels: bool = False
    font_scale: float = 0.5
    
    # Phase colors for kinogram
    phase_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),      # Blue
        (255, 128, 0),    # Cyan
        (0, 255, 0),      # Green
        (0, 255, 255),    # Yellow
        (0, 128, 255),    # Orange
        (0, 0, 255),      # Red
        (255, 0, 255),    # Magenta
        (128, 0, 128),    # Purple
    )


# ==============================================================================
# SECTION 8: SMOOTHING SETTINGS
# ==============================================================================

@dataclass
class SmoothingConfig:
    """
    Temporal smoothing configuration for reducing keypoint jitter.
    
    Uses One Euro Filter - adaptive smoothing that's smooth on slow 
    movements and responsive on fast movements.
    """
    enabled: bool = True
    min_cutoff: float = 0.3   # Lower = smoother (0.1-1.0)
    beta: float = 0.1         # Lower = less reactive to speed (0.05-0.5)
    d_cutoff: float = 1.0     # Derivative cutoff



# ==============================================================================
# MASTER CONFIGURATION CLASS
# ==============================================================================

@dataclass
class PoseEstimationConfig:
    """
    Master configuration class combining all settings.
    
    Usage:
        >>> config = PoseEstimationConfig()
        >>> device = config.device.get_device()
        >>> config.output.setup_directories()
    """
    
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)
    kinogram: KinogramConfig = field(default_factory=KinogramConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.model.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.kinogram.overlay_opacity <= 1:
            raise ValueError("overlay_opacity must be between 0 and 1")
    
    def print_summary(self) -> None:
        """Print a summary of current configuration."""
        print("\n" + "=" * 60)
        print("YOLO-POSE CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"\n[Device]")
        print(f"  Preferred: {self.device.preferred_device}")
        print(f"  Half Precision: {self.device.half_precision}")
        print(f"\n[Model]")
        print(f"  Model: {self.model.model_name}")
        print(f"  Confidence: {self.model.confidence_threshold}")
        print(f"  Tracking: {self.model.enable_tracking}")
        print(f"\n[Video]")
        print(f"  Input Dir: {self.video.input_dir}")
        print(f"  Skip Frames: {self.video.skip_frames}")
        print(f"\n[Kinogram]")
        print(f"  Phases: {self.kinogram.num_phases}")
        print(f"  Detection: {self.kinogram.phase_detection_method}")
        print(f"\n[Output]")
        print(f"  Directory: {self.output.output_dir}")
        print("=" * 60 + "\n")


# ==============================================================================
# PRESET FACTORIES
# ==============================================================================

def get_fast_config() -> PoseEstimationConfig:
    """Speed-optimized configuration."""
    config = PoseEstimationConfig()
    config.model.model_name = "yolov8n-pose.pt"  # Nano model
    config.device.half_precision = True
    config.video.skip_frames = 2
    config.output.save_visualizations = False
    return config


def get_accurate_config() -> PoseEstimationConfig:
    """Accuracy-optimized configuration."""
    config = PoseEstimationConfig()
    config.model.model_name = "yolov8x-pose.pt"  # XLarge model
    config.model.confidence_threshold = 0.3
    config.video.skip_frames = 1
    return config


def get_kinogram_config() -> PoseEstimationConfig:
    """Kinogram-optimized configuration."""
    config = PoseEstimationConfig()
    config.kinogram.num_phases = 6
    config.kinogram.phase_detection_method = "auto"
    config.output.save_kinograms = True
    config.output.kinogram_format = "png"
    return config


# ==============================================================================
# MODULE TEST
# ==============================================================================

if __name__ == "__main__":
    config = PoseEstimationConfig()
    config.print_summary()

