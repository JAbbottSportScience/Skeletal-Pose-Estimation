#!/usr/bin/env python3
"""
================================================================================
MODEL DOWNLOAD UTILITY
================================================================================
Downloads the YOLOv8-pose model weights from Ultralytics.

The model file (yolov8x-pose.pt, ~133MB) is not included in this repository
due to GitHub's file size limits. This script downloads it automatically.

Usage:
    python download_model.py              # Download default model (yolov8x-pose)
    python download_model.py --model m    # Download medium model (yolov8m-pose)
    python download_model.py --all        # Download all model variants

Note: The model will also auto-download on first run of main.py
================================================================================
"""

import argparse
from pathlib import Path


# Model variants and their approximate sizes
MODELS = {
    "n": ("yolov8n-pose.pt", "~6 MB"),
    "s": ("yolov8s-pose.pt", "~23 MB"),
    "m": ("yolov8m-pose.pt", "~52 MB"),
    "l": ("yolov8l-pose.pt", "~84 MB"),
    "x": ("yolov8x-pose.pt", "~133 MB"),
}

DEFAULT_MODEL = "x"


def download_model(variant: str = DEFAULT_MODEL, force: bool = False) -> Path:
    """
    Download a YOLOv8-pose model variant.
    
    Args:
        variant: Model size variant (n/s/m/l/x)
        force: Re-download even if file exists
        
    Returns:
        Path to downloaded model file
    """
    if variant not in MODELS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(MODELS.keys())}")
    
    model_name, size = MODELS[variant]
    model_path = Path(model_name)
    
    if model_path.exists() and not force:
        print(f"✓ Model already exists: {model_path}")
        return model_path
    
    print(f"Downloading {model_name} ({size}) from Ultralytics...")
    print("  This may take a few minutes depending on your connection.\n")
    
    try:
        from ultralytics import YOLO
        
        # Loading the model triggers automatic download
        model = YOLO(model_name)
        
        print(f"\n✓ Successfully downloaded: {model_path}")
        return model_path
        
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Run: pip install ultralytics")
        raise
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        raise


def download_all(force: bool = False) -> None:
    """Download all model variants."""
    print("Downloading all YOLOv8-pose model variants...\n")
    
    for variant in MODELS:
        try:
            download_model(variant, force=force)
        except Exception as e:
            print(f"  ✗ Failed to download {variant}: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLOv8-pose model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Variants:
  n  - Nano    (~6 MB)   - Fastest, lowest accuracy
  s  - Small   (~23 MB)  - Fast, good accuracy  
  m  - Medium  (~52 MB)  - Balanced (recommended for most uses)
  l  - Large   (~84 MB)  - Slower, higher accuracy
  x  - XLarge  (~133 MB) - Slowest, highest accuracy (default)

Examples:
  python download_model.py              # Download yolov8x-pose.pt
  python download_model.py --model m    # Download yolov8m-pose.pt
  python download_model.py --all        # Download all variants
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model variant to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all model variants"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true", 
        help="Re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8-Pose Model Downloader")
    print("=" * 60 + "\n")
    
    if args.all:
        download_all(force=args.force)
    else:
        download_model(args.model, force=args.force)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
