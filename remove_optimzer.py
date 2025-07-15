"""
Batch optimization script for Ultralytics models

This script batch optimizes all Ultralytics models (.pt files) in a specified directory.
It reduces file size by removing the optimizer and converting the model to half-precision (FP16).

Usage:
    python batch_optimize_models.py --dir /path/to/models/ [--suffix _optimized]

Arguments:
    --dir: Directory containing model files to optimize (required)
    --suffix: Suffix to add to the output filename (optional, default is _optimized)
    --overwrite: Specify this flag to overwrite input files (optional)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import sys
sys.path.insert(0, "ultralytics-timm")

# Patch torch.load to allow loading the model with weights_only=False
import torch.serialization

original_torch_load = torch.load

def patched_torch_load(file, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(file, *args, **kwargs)

torch.load = patched_torch_load

# Add this to allow loading YOLOv10DetectionModel
try:
    from ultralytics.nn.tasks import YOLOv10DetectionModel
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.YOLOv10DetectionModel', 'ultralytics.nn.modules.block.ConvNeXtBlock'])
except ImportError:
    print("Warning: ultralytics package not found or YOLOv10DetectionModel not available")


def strip_optimizer(f, s="", updates=None):
    """
    Function to strip optimizer from a model to reduce its size.

    Args:
        f (str | Path): Path to the model file to optimize.
        s (str, optional): Path to save the optimized model. If not specified, overwrites the input file.
        updates (dict, optional): Updates to add to the checkpoint.

    Returns:
        dict: Updated checkpoint dictionary.
    """
    try:
        # if already has "_optimized" in the filename, just rename and skip 
        if "_optimized" in f.stem:
            epoch = f.stem.split("_")[-1]
            #/home/sersasj/Kaggle/runs_convnext_small_p2/fold0/weights/epoch0_optimized_optimized.pt
            # to
            #/home/sersasj/Kaggle/runs_convnext_small_p2/fold0/weights/epoch0_optimized.pt
            os.rename(f, f.parent / f"{f.stem.split('_')[0]}_{epoch}{f.suffix}")
            return {}
        
        # Ensure safe loading for YOLOv10 models
        try:
            # For newer PyTorch versions that support this feature
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.YOLOv10DetectionModel', 'ultralytics.nn.modules.block.ConvNeXtBlock'])
        except (AttributeError, ImportError):
            # For older PyTorch versions or if the module is not available
            pass
            
        # Load model on CPU
        x = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "Checkpoint is not a Python dictionary"
        assert "model" in x, "Checkpoint does not contain 'model'"
    except Exception as e:
        print(f"Error: {f} is not a valid Ultralytics model: {e}")
        return {}

    # Add metadata
    metadata = {
        "date": datetime.now().isoformat(),
        "optimized_by": "batch_optimize_models.py",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # Replace with EMA model
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # Convert IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # Remove loss criterion
    x["model"].half()  # Convert to FP16
    for p in x["model"].parameters():
        p.requires_grad = False  # Disable gradient calculation

    # Remove unnecessary keys
    for k in ["optimizer", "best_fitness", "ema", "updates"]:
        x[k] = None
    x["epoch"] = -1

    # Save
    output_path = s or f
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, output_path)

    # Calculate file size (in MB)
    mb = os.path.getsize(output_path) / 1e6
    print(f"Optimizer stripped: {f}{f' -> {s}' if s else ''}, {mb:.1f}MB")

    # Delete original file if we're saving to a new location
    if s and os.path.exists(f):
        try:
            os.remove(f)
            print(f"Deleted original file: {f}")
        except Exception as e:
            print(f"Warning: Could not delete original file {f}: {e}")

    return combined


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch optimize Ultralytics models")
    parser.add_argument("--dir", type=str, default="/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/runs/fold0/weights", 
                        help="Directory containing model files to optimize")
    parser.add_argument("--suffix", type=str, default="_optimized", help="Suffix to add to the output filename (default: _optimized)")
    parser.add_argument("--overwrite", action="store_true", help="Specify this flag to overwrite input files")
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' not found")
        return

    # Find .pt files
    pt_files = list(Path(args.dir).rglob("*.pt"))
    if not pt_files:
        print(f"Warning: No .pt files found in directory '{args.dir}'")
        return
    
    print(f"Processing {len(pt_files)} model files...")

    # Optimize each model
    for pt_file in pt_files:
        if args.overwrite:
            # Overwrite input file
            strip_optimizer(pt_file)
        else:
            # Save with new filename
            stem = pt_file.stem
            output_file = pt_file.parent / f"{stem}{args.suffix}{pt_file.suffix}"
            strip_optimizer(pt_file, str(output_file))

    print("Processing complete.")


if __name__ == "__main__":
    main()
