from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import pandas as pd

class SnowDepthSequenceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, sequence_length=5):
        self.transform = transform
        self.metadata = labels.set_index('image_filename')
        self.sequence_length = sequence_length
        self.sequences = self.build_sequences(image_paths)

    def build_sequences(self, image_paths):
        df = pd.DataFrame({'path': image_paths})
        df['filename'] = df['path'].apply(lambda p: os.path.basename(p))
        df['camera'] = df['filename'].apply(lambda f: "_".join(f.split("_")[:-1]))

        sequences = []

        for cam, group in df.groupby('camera'):
            group = group.sort_values('filename')  # Sort by filename, not datetime
            paths = group['path'].tolist()
            filenames = group['filename'].tolist()
            for i in range(len(paths) - self.sequence_length + 1):
                seq_paths = paths[i:i + self.sequence_length]
                seq_filenames = filenames[i:i + self.sequence_length]
                if all(fname in self.metadata.index for fname in seq_filenames):
                    sequences.append((seq_paths, seq_filenames))

        return sequences


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths, filenames = self.sequences[idx]
        imgs = []

        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        stacked_imgs = torch.stack(imgs)  # Shape: (sequence_length, 3, H, W)

        # Predict snow depth of last image in sequence
        label = self.metadata.loc[filenames[-1], 'snowdepth_cm']
        label = torch.tensor(label, dtype=torch.float32)

        return {'images': stacked_imgs, 'label': label, 'filenames': filenames}
