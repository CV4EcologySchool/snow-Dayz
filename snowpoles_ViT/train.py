# load data
import config
import glob
import pandas as pd

# testing on cpu
from torch.utils.data import Subset
import random

# training model
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from dataset import SnowDepthDataset
from model import get_model
import IPython 
from torch.utils.data import DataLoader, random_split
import tqdm

## viz predictions
from torchvision.utils import make_grid

# early stopping 
import numpy as np
import os


jpg_paths = glob.glob(f"{config.images}/**/*.jpg", recursive=True)
JPG_paths = glob.glob(f"{config.images}/**/*.JPG", recursive=True)
image_paths = jpg_paths + JPG_paths
metadata = pd.read_csv(config.labels)

#snow_depths = snow_depths['snowdepth_cm']

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

### little but of data cleaning: 
k = set(metadata['image_filename'])
# Filter image paths that have a matching basename in metadata
filenames_from_image_paths =  [p.split('/')[-1] for p in image_paths]
image_paths = [j for (i, j) in zip(filenames_from_image_paths, image_paths) if i in k]

IPython.embed()
# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#dataset = SnowDepthDataset(image_paths, snow_depths, transform=transform) 
# Set seed for reproducibility
random.seed(42)
random.shuffle(image_paths)

# 75 / 15 / 10 split 
if config.split == 'traditional':
    total = len(image_paths) #* 0.1 ## we just want 10% of data for testing
    train_size = int(0.75 * total) 
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]

# split based off of camera # 
snex_cams = [
    # SnowEx sites
    "E6A", "E6B", "E9A", "E9E", "E9F",
    "W1A", "W2A", "W2B", "W5A", "W6A", "W6B", "W6C",
    "W8A", "W8C", "W9A", "W9B", "W9C", "W9D", "W9E", "W9G",

    # Jaeger Mesa cameras
    "jaegermesa_nabesna_C01", "jaegermesa_nabesna_C02", "jaegermesa_nabesna_C03",
    "jaegermesa_nabesna_C04", "jaegermesa_nabesna_C05", "jaegermesa_nabesna_C06",
    "jaegermesa_nabesna_C07", "jaegermesa_nabesna_C08", "jaegermesa_nabesna_C09",
    "jaegermesa_nabesna_C10", "jaegermesa_nabesna_C11", "jaegermesa_nabesna_C12",
    "jaegermesa_nabesna_C13", "jaegermesa_nabesna_C14", "jaegermesa_nabesna_C15",
    "jaegermesa_nabesna_C16", "jaegermesa_nabesna_C17", "jaegermesa_nabesna_C18",
    "jaegermesa_nabesna_C19", "jaegermesa_nabesna_C20", "jaegermesa_nabesna_C21",
    "jaegermesa_nabesna_C22"
]
wa_cams_val = ['TWISP-U-01', 'TWISP-R-01', 'CUB-H-02', 'CUB-L-02', 'CUB-M-02','CEDAR-H-01',
               'CEDAR-L-01', 'CEDAR-M-01','CUB-H-01','CUB-M-01','CUB-U-01', 'BUNKHOUSE-01']
wa_cams_test = ['CEDAR-L-01', 'CEDAR-M-01','CUB-H-01','CUB-M-01','CUB-U-01', 'BUNKHOUSE-01']


train_paths = [i for i in image_paths if i.split('/')[-2] in (snex_cams)]
val_paths = [i for i in image_paths if i.split('/')[-2] in (wa_cams_val)]
test_paths = [i for i in image_paths if i.split('/')[-2] in (wa_cams_val)]

wa_cams_val
print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Create dataset
train_dataset = SnowDepthDataset(train_paths, metadata, transform=transform)
val_dataset = SnowDepthDataset(val_paths, metadata, transform=transform)
test_dataset = SnowDepthDataset(test_paths, metadata, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'This model will train with a {device}')
model = get_model().to(device)

# Loss and optimizer
#criterion = nn.MSELoss()
criterion = nn.L1Loss() ## might be more robust to outliers
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training
def fit(model, dataloader):
    print('TRAINING')
    model.to(device)
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches #
    # num_batches = int(len(dataloader) / dataloader.batch_size)
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        counter+=1
        ## each of these is a batch ## 
        images, labels, filenames = data['image'], data['label'], data['filename']
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

    train_loss = train_running_loss / counter
    return train_loss


def validate(model, dataloader, epoch): 
    print("Validating")
    model.to(device)
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    #num_batches = int(len(data) / dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            images, labels, filenames = data['image'], data['label'], data['filename']
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            counter += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

    #filenames_batch = [f.split("/")[-1] for f in filenames_batch]  # or os.path.basename(f)
    valid_loss = valid_running_loss / counter
    return valid_loss


train_loss = []
val_loss = []

# early stopping inputs # 
best_loss_val = np.inf
best_loss_val_epoch = 0

# Store predictions per epoch for the last 5 epochs
epoch_visualizations = []

for epoch in range(config.num_epochs):
    print(f"Epoch {epoch+1} of {config.num_epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    # saving model every 50 epochs #
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": config.num_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"{config.output_path}/model_epoch{epoch}.pth",
        )

    # Save visualization for the last 5 epochs
    if epoch >= config.num_epochs - 5:
        model.eval()
        #images_batch, labels_batch, filenames_batch = next(iter(test_loader))
        batch = next(iter(test_loader))
        images_batch = batch['image']
        labels_batch = batch['label']
        filenames_batch = batch['filename']
        filenames_batch = [f.split("/")[-1] for f in filenames_batch] 
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device).unsqueeze(1)
        with torch.no_grad():
            preds = model(images_batch)

        images_np = images_batch.cpu()
        labels_np = labels_batch.cpu().squeeze().numpy()
        preds_np = preds.cpu().squeeze().numpy()
        
        # Store for later visualization
        #epoch_visualizations.append((epoch + 1, images_np, labels_np, preds_np))
        epoch_visualizations.append((epoch + 1, images_np, labels_np, preds_np, filenames_batch))


    ####### early stopping #########
    if val_epoch_loss < best_loss_val:
        best_loss_val = val_epoch_loss
        best_loss_val_epoch = epoch
    elif epoch > best_loss_val_epoch + 10:
        break

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.output_path}/loss.png")
plt.close()  # changed from plt.show()
torch.save(
    {
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
    },
    f"{config.output_path}/model.pth",
)  ### the last model
print("DONE TRAINING")

# Save image grid from last 5 epochs
IPython.embed()
fig, axes = plt.subplots(len(epoch_visualizations), config.batch_size, figsize=(config.batch_size * 2, 4 * len(epoch_visualizations)))

for row, (epoch_num, imgs, labels, preds, filenames) in enumerate(epoch_visualizations):
    for col in range(config.batch_size):
        ax = axes[row, col] if len(epoch_visualizations) > 1 else axes[col]
        img = imgs[col].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        title = f"Epoch {epoch_num}\nTrue: {labels[col]:.1f}, Pred: {preds[col]:.1f}\n \
            {filenames[col]}"
        ax.set_title(title, fontsize=8)

plt.tight_layout()
plt.savefig(f"{config.output_path}/predictions_last5epochs.png")
plt.close()