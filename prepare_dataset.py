import os
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
from tqdm import tqdm 

# Set random seed for reproducibility
np.random.seed(42)

# Define  paths
data_path = "BYU---Locating-Bacterial-Flagellar-Motors-2025"
train_dir = os.path.join(data_path, "train")

yolo_dataset_dir = "yolo_dataset"
if not os.path.exists(yolo_dataset_dir):
    os.makedirs(yolo_dataset_dir)
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 48  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation


def calc_irregular_score(x, bins=256, lb=50, ub=200):
    hist, _ = np.histogram(x, bins=bins)
    return (hist[:lb].sum() + hist[ub:].sum()) / x.size

def correct_tomo(x):
    x = (x.astype(np.uint8) + 127).astype(np.float32)
    return x

def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    q_min, q_max = np.quantile(slice_data, [0.02, 0.98])
    slice_data = np.clip(slice_data, q_min, q_max)
    slice_data = 255 * (slice_data - q_min) / (q_max - q_min)
    
    return np.uint8(slice_data)

def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv("/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/train_labels.csv")
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] >= 1].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
    
    np.random.shuffle(unique_tomos)  
    split_idx = int(len(unique_tomos) * train_split)
    # FOLD_0 is the validation set
    FOLD_0 = ['tomo_02862f', 'tomo_05f919', 'tomo_0da370', 'tomo_0fab19', 'tomo_101279', 'tomo_12f896', 'tomo_1446aa', 'tomo_19a4fd', 'tomo_1af88d', 'tomo_1efc28', 'tomo_225d8f', 'tomo_2483bb', 'tomo_2acf68', 'tomo_2daaee', 'tomo_30b580', 'tomo_331130', 'tomo_372a5c', 'tomo_37dd38', 'tomo_3b83c7', 'tomo_401341', 'tomo_444829', 'tomo_48dc93', 'tomo_4b59a2', 'tomo_507b7a', 'tomo_518a1f', 'tomo_556257', 'tomo_59b470', 'tomo_5dd63d', 'tomo_616f0b', 'tomo_62eea8', 'tomo_651ecd', 'tomo_672101', 'tomo_676744', 'tomo_6c5a26', 'tomo_6e237a', 'tomo_71ece1', 'tomo_7550f4', 'tomo_79d622', 'tomo_806a8f', 'tomo_85fa87', 'tomo_8b6795', 'tomo_8e4f7d', 'tomo_8e90f9', 'tomo_93c0b4', 'tomo_98686a', 'tomo_9997b3', 'tomo_9d3a0e', 'tomo_a020d7', 'tomo_a537dd', 'tomo_a8073d', 'tomo_a9d067', 'tomo_aec312', 'tomo_b2ebbc', 'tomo_b4a1f0', 'tomo_b80310', 'tomo_ba37ec', 'tomo_bde7f3', 'tomo_c00ab5', 'tomo_c46d3c', 'tomo_c8f3ce', 'tomo_cc3fc4', 'tomo_cff77a', 'tomo_d2339b', 'tomo_d662b0', 'tomo_d9a2af', 'tomo_dcb9b4', 'tomo_e2ccab', 'tomo_e63ab4', 'tomo_eb4fd4', 'tomo_f0adfc', 'tomo_f78e91', 'tomo_fc3c39', 'tomo_ff505c']

    # Use FOLD_0 for validation and the rest for training
    val_tomos = FOLD_0
    train_tomos = [tomo for tomo in unique_tomos if tomo not in FOLD_0]
    
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)
            
            for z in range(z_min, z_max + 1):
                slice_filename = f"slice_{z:04d}.jpg"
                
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                img = Image.open(src_path)
                img_array = np.array(img)
                irregular_score = calc_irregular_score(img_array)
                if irregular_score > 0.6:
                    print(f"Irregular score: {irregular_score}, ", "tomo_id: ", tomo_id, "z: ", z)
                    img_array = correct_tomo(img_array)


                normalized_img = normalize_slice(img_array)
                
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                Image.fromarray(normalized_img).save(dest_path, quality=100)
                
                img_width, img_height = img.size
                

                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }

# Run the preprocessing
summary = prepare_yolo_dataset(TRUST)
print(f"\nPreprocessing Complete:")
print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
print(f"- Dataset directory: {summary['dataset_dir']}")
print(f"- YAML configuration: {summary['yaml_path']}")
print(f"\nReady for YOLO training!")