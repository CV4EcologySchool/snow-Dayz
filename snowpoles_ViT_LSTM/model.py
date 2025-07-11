import timm
import torch.nn as nn

## chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1706.03762

import torch
import torch.nn as nn
import timm

class ViTLSTM(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', hidden_dim=256, lstm_layers=1):
        super(ViTLSTM, self).__init__()

        # Load pretrained ViT
        self.vit = timm.create_model(vit_model_name, pretrained=True)

        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Get feature size from ViT head and remove the classification head
        self.feature_dim = self.vit.head.in_features
        self.vit.head = nn.Identity()  # Output feature vector

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        # Final regression head
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 3, H, W)
        B, T, C, H, W = x.size()

        # Flatten batch and time to process each image through ViT
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)  # (B*T, feature_dim)

        # Reshape back to (B, T, feature_dim)
        feats = feats.view(B, T, self.feature_dim)

        # LSTM
        lstm_out, _ = self.lstm(feats)  # (B, T, hidden_dim)

        # Predict only from last time step
        last_output = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Regression
        out = self.regressor(last_output)  # (B, 1)
        return out
