from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        target = image.clone()
        # onehot labels
        labels = np.zeros(10)
        labels[label] = 1
        
        return image, target, torch.tensor(labels, dtype=torch.float32)
    

def get_mnist(batch_size: int = 32) -> DataLoader | DataLoader:
    """Get the MNIST training and testing data loaders.

    Args:
        batch_size (int, optional): Defaults to 32.

    Returns:
        DataLoader | DataLoader: Training and testing data loaders
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

    train_noisy_dataset = MNISTDataset(train_dataset)
    test_noisy_dataset = MNISTDataset(test_dataset)

    train_loader = DataLoader(train_noisy_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_noisy_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader