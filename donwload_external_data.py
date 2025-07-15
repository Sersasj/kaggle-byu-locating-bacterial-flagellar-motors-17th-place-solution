import os
import shutil
import numpy as np
import pandas as pd
import cv2
import glob
import traceback
import zarr
from cryoet_data_portal import Client, Dataset, Run
from multiprocessing import Pool, cpu_count
import functools

TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)


labels_df = pd.read_csv("labels_extra.csv")

total_motors = len(labels_df)
print(f"Total number of motors in the dataset: {total_motors}")

unique_tomos = labels_df[labels_df.groupby('tomo_id')['tomo_id'].transform('count') == 1]['tomo_id'].unique()
print(f"Found {len(unique_tomos)} unique tomograms with only one motor")

print("First few tomogram IDs:")
print(unique_tomos[:10])


def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    q_min, q_max = np.quantile(slice_data, [0.02, 0.98])
    slice_data = np.clip(slice_data, q_min, q_max)
    slice_data = 255 * (slice_data - q_min) / (q_max - q_min)
    
    return np.uint8(slice_data)

def calc_irregular_score(x, bins=256, lb=50, ub=200):
    hist, _ = np.histogram(x, bins=bins)
    return (hist[:lb].sum() + hist[ub:].sum()) / x.size

def process_single_tomogram(tomo_id, tmp_dir, out_dir, modified_labels_df):
    """Process a single tomogram - this will be run in parallel"""
    client = Client()
    run = Run.find(client, query_filters=[Run.name == tomo_id])
    
    if len(run) == 0:
        print(f"MISSING: {tomo_id}")
        return None
    else:
        run = run[0]
        
    if os.path.exists(os.path.join(out_dir, str(tomo_id))):
        print(f"Skipping already processed tomogram: {tomo_id}")
        return None
        
    try:
        process_tmp_dir = os.path.join(tmp_dir, f"tmp_{tomo_id}")
        os.makedirs(process_tmp_dir, exist_ok=True)
        
        tomo = run.tomograms[0]
        tomo.download_omezarr(dest_path=process_tmp_dir)
        
        # Load tomogram data
        files = glob.glob(os.path.join(process_tmp_dir, "*"))
        if not files:
            raise FileNotFoundError(f"No files found in temporary directory: {process_tmp_dir}")
        fpath = files[0]
        arr = zarr.open(fpath, mode='r')
        arr = arr[0]
        irregular_score = calc_irregular_score(arr)
        if irregular_score > 0.6:
            arr = arr.astype("uint8").astype("float32")


        collector = CziiCollector(tmp_dir=process_tmp_dir, out_dir=out_dir)
        _, tomo_labels = collector.process_tomogram(arr, tomo_id)
        
        if not tomo_labels.empty:
            tomo_labels.to_csv(
                os.path.join(out_dir, f"{tomo_id}_labels.csv"), 
                index=False
            )
            print(f"Saved labels for tomogram: {tomo_id}")
        
        print(f"Successfully processed: {tomo_id}")
        
        shutil.rmtree(process_tmp_dir)
        
        return tomo_labels
        
    except Exception as e:
        print(traceback.format_exc())
        print(f"FAILED: {tomo_id} - {str(e)}")
        if os.path.exists(process_tmp_dir):
            shutil.rmtree(process_tmp_dir)
        return None

class CziiCollector():
    def __init__(
        self, 
        tmp_dir: str = "/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/tmp", 
        out_dir: str = "/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/external_data", 
    ):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.out_dir = out_dir

        # Out dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Create a copy of labels_df to modify
        self.modified_labels_df = labels_df.copy()
        # Add a new column for scaled coordinates
        self.modified_labels_df['scaled_z'] = 0.0
        self.modified_labels_df['scaled_y'] = 0.0
        self.modified_labels_df['scaled_x'] = 0.0

    def update_labels(self):
        # Read all CSV labels in a directory and return a combined DataFrame
        dfs = []
        print(self.out_dir)
        for file in os.listdir(self.out_dir):
            if file.endswith(".csv") and not file.startswith("updated"):
                df = pd.read_csv(os.path.join(self.out_dir, file))                
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Updated labels: {len(combined_df)}")
        print(combined_df.head())
        combined_df.to_csv("updated_labels.csv", index=False)


    def process_tomogram(self, tomogram_array, tomogram_id):
        """Process a tomogram and extract slices around motor centers"""
        print(f"Original shape: {tomogram_array.shape}")
        z_size, y_size, x_size = tomogram_array.shape
        
        # Original label size is 128x512x512 (z, x, y)
        label_z_size, label_y_size, label_x_size = 128, 512, 512
        
        # Calculate scaling factors
        z_scale = z_size / label_z_size
        y_scale = y_size / label_y_size
        x_scale = x_size / label_x_size
        
        # Get all motors for this tomogram
        tomo_motors = self.modified_labels_df[self.modified_labels_df['tomo_id'] == tomogram_id]
        
        if len(tomo_motors) == 0:
            print(f"No motors found for tomogram {tomogram_id}")
            return tomogram_array, tomo_motors
            
        # Process each motor in this tomogram
        for idx, row in tomo_motors.iterrows():
            # Scale the coordinates
            original_z = row['z']
            original_y = row['y']
            original_x = row['x']
            
            scaled_z = original_z * z_scale
            scaled_y = original_y * y_scale
            scaled_x = original_x * x_scale
            
            # Update the scaled coordinates
            self.modified_labels_df.loc[idx, 'scaled_z'] = scaled_z
            self.modified_labels_df.loc[idx, 'scaled_y'] = scaled_y
            self.modified_labels_df.loc[idx, 'scaled_x'] = scaled_x
            tomo_motors.loc[idx, 'scaled_z'] = scaled_z
            tomo_motors.loc[idx, 'scaled_y'] = scaled_y
            tomo_motors.loc[idx, 'scaled_x'] = scaled_x


            # Calculate slice range around the motor center
            z_min = max(0, int(scaled_z - TRUST))
            z_max = min(z_size, int(scaled_z + TRUST + 1))
            
            for z in range(z_min, z_max):
                slice_data = tomogram_array[z, :, :]
                normalized_slice = normalize_slice(slice_data)
                
                # Save the slice as PNG
                output_dir = os.path.join(self.out_dir, str(tomogram_id))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                outpath = os.path.join(output_dir, f"{tomogram_id}_{z:04d}.png")
                cv2.imwrite(outpath, normalized_slice)
                
           
        
        return tomogram_array, tomo_motors

    def run(self):
        """Run the collection process for all tomograms in parallel"""
        process_func = functools.partial(
            process_single_tomogram,
            tmp_dir=self.tmp_dir,
            out_dir=self.out_dir,
            modified_labels_df=self.modified_labels_df
        )
        
        num_processes = max(1, int(cpu_count() * 0.2))
        print(f"Using {num_processes} processes for parallel processing")
        
        with Pool(num_processes) as pool:
            results = pool.map(process_func, unique_tomos)
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            combined_labels = pd.concat(valid_results, ignore_index=True)
            combined_labels.to_csv(os.path.join(self.out_dir, "updated_labels.csv"), index=False)
            print("Final labels file updated successfully")
        
        return

# Run the collector
if __name__ == '__main__':
    collector = CziiCollector()
    collector.run()
    collector.update_labels()


