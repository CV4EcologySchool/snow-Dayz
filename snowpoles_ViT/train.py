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


# early stopping 
import numpy as np


image_paths = glob.glob(f"{config.images}/**/*.JPG")
snow_depths = pd.read_csv(config.labels)
snow_depths = snow_depths['Snow Depth (cm)']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#dataset = SnowDepthDataset(image_paths, snow_depths, transform=transform) 
random.seed(42)
dataset = Subset(SnowDepthDataset(image_paths, snow_depths, transform=transform),
                 random.sample(range(len(image_paths)), int(0.1 * len(image_paths))))

# Split lengths
total_size = len(dataset) 
train_size = int(0.7 * total_size) 
val_size = int(0.15 * total_size) 
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Print split sizes
print(f"Total dataset size: {total_size}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
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
    for images, labels in tqdm.tqdm(dataloader):
        counter+=1
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
    # num_batches = int(len(data) / dataloader.batch_size)
    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            counter += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

    valid_loss = valid_running_loss / counter
    return valid_loss


train_loss = []
val_loss = []

# early stopping inputs # 
best_loss_val = np.inf
best_loss_val_epoch = 0

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