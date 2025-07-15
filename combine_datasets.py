#/home/sersasj/Kaggle/yolo/yolo_dataset_fold0
#/home/sersasj/Kaggle/yolo_dataset

import os
import shutil
from pathlib import Path

source_dir1 = "/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/yolo_dataset_fold0_25d"
source_dir2 = "/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/yolo_dataset_external_25d"

destination_dir = "/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/combined_dataset_25d"

os.makedirs(os.path.join(destination_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "labels", "val"), exist_ok=True)

shutil.copytree(os.path.join(source_dir1, "images", "train"), 
                os.path.join(destination_dir, "images", "train"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir1, "images", "val"), 
                os.path.join(destination_dir, "images", "val"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir1, "labels", "train"), 
                os.path.join(destination_dir, "labels", "train"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir1, "labels", "val"), 
                os.path.join(destination_dir, "labels", "val"), 
                dirs_exist_ok=True)

# Copy images and labels from second dataset
shutil.copytree(os.path.join(source_dir2, "images", "train"), 
                os.path.join(destination_dir, "images", "train"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir2, "images", "val"), 
                os.path.join(destination_dir, "images", "val"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir2, "labels", "train"), 
                os.path.join(destination_dir, "labels", "train"), 
                dirs_exist_ok=True)
shutil.copytree(os.path.join(source_dir2, "labels", "val"), 
                os.path.join(destination_dir, "labels", "val"), 
                dirs_exist_ok=True)

dataset_yaml = f"""names:
  0: motor
path: {destination_dir}
train: {os.path.join(destination_dir, "images", "train")}
val: {os.path.join(destination_dir, "images", "val")}
"""

with open(os.path.join(destination_dir, "dataset.yaml"), "w") as f:
    f.write(dataset_yaml)

print(f"Combined dataset created at: {destination_dir}")
print("Directory structure:")
for root, dirs, files in os.walk(destination_dir):
    level = root.replace(destination_dir, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level == 2:  
        for f in files:
            print(f"{indent}    {f}")

