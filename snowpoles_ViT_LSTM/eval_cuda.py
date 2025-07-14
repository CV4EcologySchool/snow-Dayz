import os
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from model import ViTLSTM  # Make sure this imports your ViTLSTM class

import IPython

import config  # Your config file with EXP_DIR, EXP_NAME, etc.

def load_model(model_path, device):
    model = ViTLSTM(vit_model_name='vit_base_patch16_224', hidden_dim=256, lstm_layers=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict(model, image_path, output_dir, device='cpu', seq_len=5, batch_size=32):
    from torchvision.transforms.functional import to_tensor  # minor speed-up
    output_folder = os.path.join(output_dir, "predictions")
    os.makedirs(output_folder, exist_ok=True)

    # Filter for E9E and E3B only
    files = sorted(glob.glob(f"{image_path}/**/*.JPG", recursive=True))
    files = [f for f in files if "E9E" in f or "E3B" in f]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Prepare all sequences
    all_seq_tensors = []
    seq_file_metadata = []

    print("Preparing image sequences...")
    for i in range(len(files) - seq_len + 1):
        seq_files = files[i:i + seq_len]
        try:
            images_tensor = [transform(Image.open(file).convert("RGB")) for file in seq_files]
            seq_tensor = torch.stack(images_tensor)  # (seq_len, 3, 224, 224)
            all_seq_tensors.append(seq_tensor)
            seq_file_metadata.append(seq_files[-1])  # Track final image in sequence
        except Exception as e:
            print(f"Skipping sequence due to error: {e}")

    # Predict in batches
    print(f"Running model inference in batches of {batch_size}...")
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_seq_tensors), batch_size)):
            batch = all_seq_tensors[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(device)  # (B, seq_len, 3, 224, 224)
            batch_preds = model(batch_tensor).squeeze()

            # Handle scalar vs vector outputs
            if batch_preds.ndim == 0:
                batch_preds = batch_preds.unsqueeze(0)

            predictions.extend(batch_preds.tolist())

    # Extract metadata
    cameras = [Path(f).name.split("_")[0] for f in seq_file_metadata]
    filenames = [Path(f).name for f in seq_file_metadata]

    # Save prediction CSV
    df = pd.DataFrame({
        "Camera": cameras,
        "filename": filenames,
        "predicted_snowdepth": predictions
    })


    out_csv_path = os.path.join(output_folder, "snowdepth_predictions.csv")
    df.to_csv(out_csv_path, index=False)
    print(f"Saved predictions to {out_csv_path}")

    # Load and merge with ground truth labels
    labels_df = pd.read_csv(config.labels)
    merged_df = df.merge(labels_df, left_on='filename', right_on='image_filename', how="inner")

    #IPython.embed()
    # Compute error metrics
    merged_df["abs_error"] = np.abs(merged_df["predicted_snowdepth"] - merged_df["true_snowdepth"])
    merged_df["pct_error"] = merged_df["abs_error"] / (merged_df["true_snowdepth"].replace(0, np.nan)) * 100
    merged_df["diff"] = merged_df["predicted_snowdepth"] - merged_df["true_snowdepth"]

    mae = merged_df["abs_error"].mean()
    mape = merged_df["pct_error"].mean()
    avg_diff = merged_df["diff"].mean()
    avg_std = merged_df["diff"].std()

    results = {
        "MAE": [mae],
        "MAPE": [mape],
        "Avg_Diff": [avg_diff],
        "Avg_Std": [avg_std]
    }

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "results_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results summary to {results_path}")

    return

def main():
    model_path = os.path.join(config.output_path, "model_epoch0.pth")
    output_dir = config.output_path

    model = load_model(model_path, device=config.device)

    predict(model, config.images, output_dir, device=config.device, seq_len=config.sequence_length)

if __name__ == "__main__":
    main()
