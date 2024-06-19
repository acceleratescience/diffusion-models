import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time):
        device = time.device
        half = self.embedding_dim // 2
        embeddings = math.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        layers = [
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Unflatten(dim=1, unflattened_size=(out_features, 1, 1)),

        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.in_features)
        return self.model(input)


class ContextBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        layers = [
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Unflatten(dim=1, unflattened_size=(out_features, 1, 1))
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.in_features)
        return self.model(input)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.pool1(x1)
        return x1, x2
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, residual_channels, out_channels):
        super().__init__()

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels + residual_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=out_channels//2, kernel_size=3, padding=1)


    def forward(self, x1, residual):
        x3 = self.upconv1(x1)
        x3 = torch.cat([x3, residual], dim=1)
        x4 = F.relu(self.bn1(self.conv1(x3)))
        return x4
            

class UNetSmol(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1, block_channels: list = [16, 32, 64]):
        """A small U-Net model

        Args:
            input_channels (int, optional): How many channels in image input. Defaults to 1.
            output_channels (int, optional): How many channels in image output. Defaults to 1.
            block_channels (list, optional): Number of channels in each block. Defaults to [16, 32, 64].
        """
        super().__init__()
        block_1_channels, block_2_channels, block_3_channels = block_channels

        assert len(block_channels) == 3, "The block_channels must have a length of 3"

        # Encoder
        self.down1 = DownBlock(in_channels=input_channels,
                               out_channels=block_1_channels)
        self.down2 = DownBlock(in_channels=block_1_channels,
                               out_channels=block_2_channels)

        # Bottleneck
        self.conv3 = nn.Conv2d(in_channels=block_2_channels,
                               out_channels=block_3_channels,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=block_3_channels)

        # Decoder
        self.up1 = UpBlock(in_channels=block_3_channels,
                           residual_channels=block_2_channels,
                           out_channels=block_2_channels)
        self.up2 = UpBlock(in_channels=block_2_channels,
                           residual_channels=block_1_channels,
                           out_channels=block_1_channels)
        self.conv6 = nn.Conv2d(in_channels=block_1_channels,
                               out_channels=output_channels,
                               kernel_size=1)

        self.time_embedder = SinusoidBlock(embedding_dim=block_2_channels)
        self.time_embedder_1 = TimeBlock(in_features=block_2_channels,
                                         out_features=block_3_channels)
        self.time_embedder_2 = TimeBlock(in_features=block_2_channels,
                                         out_features=block_2_channels)
        self.label_encoder_1 = ContextBlock(in_features=10,
                                            out_features=block_3_channels)
        self.label_encoder_2 = ContextBlock(in_features=10,
                                            out_features=block_2_channels)


    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input noisey image
            t (torch.Tensor): Timestep
            labels (torch.Tensor, optional): Class labels. Defaults to None.

        Returns:
            torch.Tensor: Predicted noise
        """
        # Down
        x1, x2 = self.down1(x)
        x3, x4 = self.down2(x2)

        # Bottleneck
        x5 = F.relu(self.bn3(self.conv3(x4)))

        # Up
        t = self.time_embedder(t)
        time_embedding_1 = self.time_embedder_1(t)

        label_embedding_1 = self.label_encoder_1(labels)
        x5 = label_embedding_1 * x5 + time_embedding_1
        
        x6 = self.up1(x5, x3)

        time_embedding_2 = self.time_embedder_2(t)
        label_embedding_2 = self.label_encoder_2(labels)
        x6 = label_embedding_2 * x6 + time_embedding_2

        x7 = self.up2(x6, x1)
        

        x8 = self.conv6(x7)
        return x8