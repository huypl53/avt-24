import os
import shutil
import random
from tqdm import tqdm

# Set up paths and parameters
data_folder = "sample/runway"  # Change to your data folder path
output_folder = "sample/runway"
ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}

# Create output directories
for split in ratios.keys():
    os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "annotations"), exist_ok=True)

# Collect all unique basenames (without extensions)
file_basenames = set(
    os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith(".tif")
)

# Shuffle and split the basenames
file_basenames = list(file_basenames)
random.shuffle(file_basenames)
total_files = len(file_basenames)

split_points = {
    "train": int(ratios["train"] * total_files),
    "valid": int((ratios["train"] + ratios["valid"]) * total_files),
}

splits = {
    "train": file_basenames[: split_points["train"]],
    "valid": file_basenames[split_points["train"] : split_points["valid"]],
    "test": file_basenames[split_points["valid"] :],
}

# Copy files to respective folders
for split, basenames in tqdm(splits.items(), leave=False, desc=split):
    for basename in tqdm(basenames, leave=False):
        for ext in [".tif", ".png", ".json"]:
            src_file = os.path.join(data_folder, basename + ext)
            if os.path.exists(src_file):
                dst_folder = (
                    "images"
                    if ext == ".tif"
                    else "labels" if ext == ".png" else "annotations"
                )
                dst_file = os.path.join(
                    output_folder, split, dst_folder, basename + ext
                )
                shutil.copy2(src_file, dst_file)

print("Data split completed successfully.")
