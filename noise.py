import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class MyDataset(Dataset):
    def __init__(self, train=True, transform=None, noise_transform=None):
        self.cifar10 = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        self.noise_transform = noise_transform

    def __getitem__(self, index):
        img, target = self.cifar10[index]
        if self.noise_transform:
            noisy_img = self.noise_transform(img)
        else:
            noisy_img = img
        return {'src': noisy_img, 'target': img}

    def __len__(self):
        return len(self.cifar10)

def get_cifar10_dataloaders(batch_size, noise_mean=0.0, noise_std=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])  # 标准化
    ])
    noise_transform = AddGaussianNoise(noise_mean, noise_std)
    
    train_dataset = MyDataset(train=True, transform=transform, noise_transform=noise_transform)
    test_dataset = MyDataset(train=False, transform=transform, noise_transform=noise_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, test_loader
