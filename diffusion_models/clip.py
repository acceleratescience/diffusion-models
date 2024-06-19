import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(392, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.encoder(x)
    

class LabelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )

    def forward(self, x):
        return self.encoder(x)
    

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=64, projection_dim=128):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout()
        self.l2_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.l2_norm(x)
        return x
    

class BasicCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = DigitEncoder()
        self.label_encoder = LabelEncoder()
        self.temperature = nn.Parameter(torch.tensor(np.log(1/0.07)))

        self.W_i = ProjectionHead(64, 128)
        self.W_t = ProjectionHead(64, 128)

    def forward(self, imgs, labels):
        I_f = self.image_encoder(imgs)
        T_f = self.label_encoder(labels)

        I_e = self.W_i(I_f)
        T_e = self.W_t(T_f)

        # l2 normalize
        I_e = F.normalize(I_e, p=2, dim=1)
        T_e = F.normalize(T_e, p=2, dim=1)

        logits = I_e @ T_e.T * self.temperature

        return logits
    

def loss_function(logits, N):
    loss_i = F.cross_entropy(logits, torch.arange(N))
    loss_t = F.cross_entropy(logits.T, torch.arange(N))

    loss = (loss_i + loss_t)/2

    return loss