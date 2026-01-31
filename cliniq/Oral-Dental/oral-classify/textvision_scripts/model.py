# model.py
# CRNN model for OCR with CTC.

from __future__ import annotations
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_h: int = 32, num_channels: int = 1, hidden_size: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (32,128)->(16,64)

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16,64)->(8,32)

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1))  # (8,32)->(4,32)
        )

        # After CNN: [B, 256, 4, W]
        self.rnn = nn.LSTM(
            input_size=256*4,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)              # [B, C, H, W]
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)    # [B, W, C, H]
        x = x.reshape(B, W, C*H)     # [B, W, features]
        x, _ = self.rnn(x)           # [B, W, 2*hidden]
        x = self.fc(x)               # [B, W, num_classes]
        return x
