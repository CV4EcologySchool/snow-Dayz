from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import IPython 

class SnowDepthDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.metadata = labels
        self.labels = self.metadata['snowdepth_cm']  # list or numpy array of snow depths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = self.image_paths[idx].split('/')[-1]
        match = self.metadata[self.metadata['image_filename'] == filename]
        if not match.empty:
            label = match['snowdepth_cm'].values[0]
        else:
            label = None  # or raise an error


        ## we need to get label by filename NOT index  ## 
        label = torch.tensor(label, dtype=torch.float32)
        return {'image': img, 'label':label, 'filename': filename}