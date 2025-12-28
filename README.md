# Sprint Pose Estimation & Kinogram Generation (YOLO-Pose)

A streamlined Python pipeline for analyzing sprint biomechanics using YOLO-Pose. 
Optimized for Apple Silicon (M4) with native MPS support.

## Why YOLO-Pose?

- **One-line install**: `pip install ultralytics` — no dependency hell
- **Native MPS support**: First-class Apple Silicon GPU acceleration
- **Single-stage detection**: Faster than two-stage pipelines
- **Built-in tracking**: Maintains person IDs across frames
- **Active development**: Regular updates and community support

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JAbbottSportScience/Skeletal-Pose-Estimation.git
cd Skeletal-Pose-Estimation
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

The YOLOv8-pose model weights (~133 MB) are not included in this repository due to GitHub's file size limits.

**Option A: Automatic download** (recommended)  
The model will download automatically on first run:
```bash
python main.py --video path/to/sprint.mp4
```

**Option B: Manual download**  
```bash
python download_model.py
```

**Option C: Download via Python**
```python
from ultralytics import YOLO
model = YOLO("yolov11x-pose.pt")  # Downloads automatically
```

### 5. Verify Installation

```bash
python -c "from ultralytics import YOLO; print('✓ Ready!')"
```

## Quick Start

### Process a Video

```bash
python main.py --video path/to/sprint.mp4
```

### Use Fast Preset

```bash
python main.py --preset fast --video sprint.mp4
```

### Generate Kinogram with Specific Phases

```bash
python main.py --video sprint.mp4 --phases 10,25,40,55,70
```

## Python API

```python
from configs.config import PoseEstimationConfig
from main import SprintAnalysisPipeline

# Create and run pipeline
pipeline = SprintAnalysisPipeline()
result = pipeline.process_video("sprint.mp4")

print(f"Processed {result['num_frames']} frames at {result['avg_fps']:.1f} FPS")
```

### Direct YOLO Usage

```python
from ultralytics import YOLO

# Load model (auto-downloads weights if not present)
model = YOLO('yolov8x-pose.pt')

# Run inference on Apple Silicon
results = model('sprint.mp4', device='mps')

# Access keypoints
for r in results:
    if r.keypoints is not None:
        keypoints = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)
        scores = r.keypoints.conf.cpu().numpy()   # (N, 17)
```

## Model Variants

| Model | Size | Speed (M4) | Accuracy | Use Case |
|-------|------|-----------|----------|----------|
| `yolov11n-pose.pt` | ~6 MB | ~3ms | Good | Quick testing |
| `yolov11s-pose.pt` | ~23 MB | ~5ms | Better | Fast processing |
| `yolov11m-pose.pt` | ~52 MB | ~8ms | High | **Recommended** |
| `yolov11l-pose.pt` | ~84 MB | ~12ms | Higher | Detailed analysis |
| `yolov11x-pose.pt` | ~133 MB | ~18ms | Highest | Maximum accuracy |

To use a different model, update your config or download it directly:
```bash
python download_model.py --model m  # Download medium model
```

## COCO Keypoint Format (17 points)

```
 0: nose           5: left_shoulder   10: right_wrist    15: left_ankle
 1: left_eye       6: right_shoulder  11: left_hip       16: right_ankle
 2: right_eye      7: left_elbow      12: right_hip
 3: left_ear       8: right_elbow     13: left_knee
 4: right_ear      9: left_wrist      14: right_knee
```

**Key joints for sprint biomechanics**: 5-6 (shoulders), 11-12 (hips), 13-14 (knees), 15-16 (ankles)

## Project Structure

```
Skeletal-Pose-Estimation/
├── configs/config.py      # Configuration settings
├── src/
│   ├── pose_estimator.py  # YOLO-Pose wrapper
│   ├── video_processor.py # Video I/O
│   └── visualizer.py      # Skeleton drawing & kinograms
├── output/                # Generated files
├── main.py                # CLI entry point
├── download_model.py      # Model download utility
└── requirements.txt       # Dependencies
```

## Output Files

- `output/keypoints/*.json` - Raw keypoint data
- `output/kinograms/*.png` - Composite kinogram images
- `output/videos/*.mp4` - Annotated videos (optional)

## Configuration Presets

```python
from configs.config import get_fast_config, get_accurate_config

# Fast: nano model, skip frames, no visualizations
config = get_fast_config()

# Accurate: xlarge model, all frames
config = get_accurate_config()
```

## Troubleshooting

### Model not found
If you see `Model not found` errors, run:
```bash
python download_model.py
```

### MPS (Apple Silicon) issues
If GPU acceleration isn't working, verify PyTorch MPS support:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

### Permission errors on macOS
If you get camera/file access errors:
```bash
# Grant terminal access in System Preferences > Privacy & Security
```

## License

MIT
