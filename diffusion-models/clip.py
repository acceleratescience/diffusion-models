import torch
import torch.nn as nn
import numpy as np


class DigitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.encoder(x)
    

class LabelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

    def forward(self, x):
        return self.encoder(x)
    

class BasicCLIP(nn.Module):
    def __init__(self):
        super().__init__(latent_dim=32)

        self.image_encoder = DigitEncoder()
        self.label_encoder = LabelEncoder()
        self.temperature = nn.Parameter(torch.tensor(np.log(1/0.07)))

        self.W_i = nn.Linear(32, 128)
        self.W_t = nn.Linear(32, 128)

    def forward(self, imgs, labels):
        I_f = self.image_encoder(imgs)
        T_f = self.label_encoder(labels)

        I_e = self.W_i(I_f)
        T_e = self.W_t(T_f)

        I_e = torch.nn.functional.normalize(I_e, p=2, dim=1)
        T_e = torch.nn.functional.normalize(T_e, p=2, dim=1)

        logits = I_e @ T_e.T * self.temperature

        return logits