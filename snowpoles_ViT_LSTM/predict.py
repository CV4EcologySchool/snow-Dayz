import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import glob
import tqdm
import cv2
from model import ViTLSTM
import numpy as np
import IPython
import pandas as pd
import config

def load_model(model_path, device = 'cpu'):
    # IPython.embed()
    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )
    model = ViTLSTM().to(device)
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
   
def predict(model, images_path, output_path, seq_len=5):
    if not os.path.exists(f"{output_path}/predictions"):
        os.makedirs(f"{output_path}/predictions", exist_ok=True)

    # Load ground-truth labels
    label_csv = config.labels
    labels_df = pd.read_csv(label_csv)
    labels_dict = dict(zip(labels_df['filename'], labels_df['snowdepth']))

    Cameras, filenames, snowdepths, true_snowdepths = [], [], [], []

    snowpolefiles = sorted(glob.glob(f"{images_path}/*.JPG"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(snowpolefiles) - seq_len + 1)):
            sequence_files = snowpolefiles[i:i + seq_len]

            images_tensor = []
            for file in sequence_files:
                image = Image.open(file).convert("RGB")
                image = transform(image)
                images_tensor.append(image)

            images_tensor = torch.stack(images_tensor).unsqueeze(0)

            outputs = model(images_tensor)
            predicted_snowdepth = outputs.item()

            last_file = sequence_files[-1]
            last_image = np.array(Image.open(last_file).resize((224, 224)))
            filename = Path(last_file).name
            Camera = filename.split('_')[0]

            # Get true label
            true_snowdepth = labels_dict.get(filename, None)

            vis_predicted_keypoints(output_path, filename, last_image, predicted_snowdepth, true_snowdepth)

            Cameras.append(Camera)
            filenames.append(filename)
            snowdepths.append(predicted_snowdepth)
            true_snowdepths.append(true_snowdepth)

    results = pd.DataFrame({
        'Camera': Cameras,
        'filename': filenames,
        'predicted_snowdepth': snowdepths,
        'true_snowdepth': true_snowdepths
    })

    results.to_csv(f"{output_path}/predictions/{Camera}_results.csv")
    return results

def main():

    model_path = '/Users/catherinebreen/Dropbox/snowpoles_ViT_outputs/ViTLSTM_bs2_co_wa_ak/model_epoch0.pth'
    model = load_model(model_path, device = config.device)

    images_path = "/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/CHE6"
    output_path = "/Users/catherinebreen/Dropbox/snowpoles_ViT_outputs/ViTLSTM_bs2_co_wa_ak"

    ## returns a set of images of outputs
    outputs = predict(model, images_path, output_path)  

if __name__ == '__main__':
    main()




