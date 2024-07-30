import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CNNMutConfig:
    mut_dim: int
    kernal_size: int
    out_dim: int
    dropout: float


class CNNMut(nn.Module):
    DEFAULT_CONFIG = CNNMutConfig(
        mut_dim=3008,
        kernal_size=32,
        out_dim=768,
        dropout=0.1,
    )

    def __init__(self, conf: CNNMutConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.conf = conf
        stride = 2
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=conf.kernal_size, stride=stride),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=conf.kernal_size, stride=stride),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=1, kernel_size=conf.kernal_size, stride=stride),
            nn.ReLU(),
            nn.Dropout(p=conf.dropout),
        )

        encoder_out_dim = 0
        input_dim = conf.mut_dim
        for _ in range(3):
            encoder_out_dim = int((input_dim - conf.kernal_size) / stride) + 1
            input_dim = encoder_out_dim

        self.out = nn.Linear(encoder_out_dim, conf.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-2) == 1
        x = x.float()
        x_dim = x.dim()
        x = x.unsqueeze(0) if x_dim != 3 else x
        encoder_out = self.encoder(x)
        out = self.out(encoder_out)
        return out.squeeze(0) if x_dim != 3 else out
