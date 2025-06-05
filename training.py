import os
import sys
import numpy as np
import pandas as pd
import yaml
import json
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add custom ultralytics implementation to Python path
sys.path.insert(0, "ultralytics-timm")

import torch
import torch.serialization
original_torch_load = torch.load

def patched_torch_load(file, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(file, *args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO


import random

# Set up base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


model = YOLO("ultralytics-timm/ultralytics/cfg/models/11/yolo-11-convx-base-head-l_v3_single_scale.yaml")


yaml_path = Path("/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/combined_dataset/dataset.yaml")  


results = model.train(
    data=str(yaml_path),
    epochs=40,
    batch=2,
    imgsz=960,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    dropout=0.1,
    project=str(RUNS_DIR),
    exist_ok=True,
    name=f"v2",
    patience=100,   
    save_period=1,
    val=True,   
    mosaic=0.5,
    close_mosaic=0,
    mixup=0.1,
    flipud=0.5,
    scale=0.25,
    degrees=45,
    seed=42,
    deterministic=True,
    label_smoothing=0.1,
    augment=True,
    device=0,
    #val_period=1,
)