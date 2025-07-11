import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Input and output directories
input_dir = "/Volumes/NINA_Photos/cosgrove_snowpole_data"
output_dir = "/Volumes/NINA_Photos/cosgrove_snowpole_data_resized"
os.makedirs(output_dir, exist_ok=True)

metadata = []

# Collect all image paths first to use with tqdm
image_paths = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_paths.append((root, file))

# Process with progress bar
for root, file in tqdm(image_paths, desc="Resizing images", unit="image"):
    input_path = os.path.join(root, file)

    # Recreate the folder structure in the output directory
    relative_path = os.path.relpath(root, input_dir)
    output_subdir = os.path.join(output_dir, relative_path)
    os.makedirs(output_subdir, exist_ok=True)

    # Resize image
    try:
        img = Image.open(input_path)
        img = img.resize((448, 448), Image.LANCZOS)

        output_path = os.path.join(output_subdir, file)
        img.save(output_path)

        # Extract metadata from filename
        name = Path(file).stem  # remove .jpg
        parts = name.split("_")
        if len(parts) >= 3:
            camera = "_".join(parts[:3])
            datetime_str = parts[3]
            if len(datetime_str) >= 13:
                date = f"{datetime_str[:4]}-{datetime_str[4:6]}-{datetime_str[6:8]}"
                hour = datetime_str[9:11]
                metadata.append({
                    "image_name": file,
                    "camera": camera,
                    "date": date,
                    "hour": hour
                })
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Save metadata CSV
df = pd.DataFrame(metadata)
df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
print("Done. Resized images saved to 'cosgrove_resized/' and metadata.csv created.")
