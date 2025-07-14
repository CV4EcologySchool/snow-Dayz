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

import matplotlib.pyplot as plt 

import IPython

import config  # Your config file with EXP_DIR, EXP_NAME, etc.

def load_model(model_path, device):
    model = ViTLSTM(vit_model_name='vit_base_patch16_224', hidden_dim=256, lstm_layers=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def vis_predicted_keypoints(output_path, filename, image, pred_snowdepth, true_snowdepth=None):
    plt.imshow(image)
    if true_snowdepth is not None:
        plt.title(f"Pred: {pred_snowdepth:.1f} cm | True: {true_snowdepth:.1f} cm")
    else:
        plt.title(f"Pred: {pred_snowdepth:.1f} cm")
    plt.axis("off")
    plt.savefig(f"{output_path}/predictions/pred_{filename}.png")
    plt.close()

def predict(model, image_path, output_dir, device='cpu', seq_len=5):
    output_folder = os.path.join(output_dir, "predictions")
    os.makedirs(output_folder, exist_ok=True)

    # Sort files to maintain sequence order
    #files = sorted(glob.glob(f"{image_path}/**/*.JPG"))
    files = sorted(glob.glob(f"{image_path}/E9E/*.JPG"))
    #files = sorted(Path(image_path).rglob("*.JPG"))


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cameras, filenames, predictions = [], [], []

    with torch.no_grad():
        for i in tqdm(range(len(files) - seq_len + 1)):
            seq_files = files[i:i + seq_len]
            images_tensor = []

            for file in seq_files:
                img = Image.open(file).convert("RGB")
                img_tensor = transform(img)
                images_tensor.append(img_tensor)

            # Shape: (seq_len, 3, 224, 224)
            images_tensor = torch.stack(images_tensor).unsqueeze(0).to(device)  # (1, seq_len, 3, 224, 224)
            pred = model(images_tensor)  # Output: (1, 1)
            pred = pred.item()

            last_file = seq_files[-1]
            filename = Path(last_file).name
            camera = filename.split("_")[0]

            cameras.append(camera)
            filenames.append(filename)
            predictions.append(pred)

            if i % 50 == 0:
                # Load image and true label for visualization
                img_vis = Image.open(last_file).convert("RGB").resize((224, 224))
                img_vis = np.array(img_vis)

                # Get true snow depth from merged_df (if available)
                labels_df = pd.read_csv(config.labels)
                true_snowdepth = None
                match = labels_df[labels_df["image_filename"] == filename]
                if not match.empty:
                    true_snowdepth = match.iloc[0]["snowdepth_cm"]

                vis_predicted_keypoints(output_dir, filename, img_vis, pred, true_snowdepth)

    df = pd.DataFrame({
        "Camera": cameras,
        "filename": filenames,
        "predicted_snowdepth": predictions
    })

    out_csv_path = os.path.join(output_folder, "snowdepth_predictions.csv")
    df.to_csv(out_csv_path, index=False)
    print(f"Saved predictions to {out_csv_path}")

        # Merge predictions with true labels

    labels_df = pd.read_csv(config.labels)
    merged_df = df.merge(labels_df, left_on= 'filename',right_on="image_filename", how="inner")

    # Compute metrics
    merged_df["abs_error"] = np.abs(merged_df["predicted_snowdepth"] - merged_df["snowdepth_cm"])
    merged_df["pct_error"] = merged_df["abs_error"] / (merged_df["snowdepth_cm"].replace(0, np.nan)) * 100
    merged_df["diff"] = merged_df["predicted_snowdepth"] - merged_df['snowdepth_cm']

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
    model_path = os.path.join(config.output_path, "model.pth")
    output_dir = config.output_path

    model = load_model(model_path, device=config.device)

    predict(model, config.images, output_dir, device=config.device, seq_len=config.sequence_length)

if __name__ == "__main__":
    main()
