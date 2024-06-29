import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class AddNoise(object):
    def __init__(self, noise_type='gaussian', params={}):
        self.noise_type = noise_type
        self.params = params

    def __call__(self, tensor):
        if self.noise_type == 'gaussian':
            mean = self.params.get('mean', 0.0)
            std = self.params.get('std', 1.0)
            noise = torch.randn(tensor.size()) * std + mean
            return torch.clamp(tensor + noise, 0., 1.)
        
        elif self.noise_type == 'salt_and_pepper':
            amount = self.params.get('amount', 0.05)
            salt_vs_pepper = self.params.get('salt_vs_pepper', 0.5)
            
            mask = torch.rand(tensor.size())
            salt = mask < (amount * salt_vs_pepper)
            pepper = (mask >= (amount * salt_vs_pepper)) & (mask < amount)
            
            tensor[salt] = 1.0
            tensor[pepper] = 0.0
            return tensor
        
        elif self.noise_type == 'speckle':
            mean = self.params.get('mean', 0.0)
            std = self.params.get('std', 0.1)
            noise = torch.randn(tensor.size()) * std + mean
            return torch.clamp(tensor + tensor * noise, 0., 1.)
        
        elif self.noise_type == 'poisson':
            tensor = torch.clamp(tensor, 0., 1.)  # Clamp to [0, 1]
            noise = torch.poisson(tensor * 255.0) / 255.0
            return torch.clamp(noise, 0., 1.)  # Clamp result to [0, 1]
        
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def __repr__(self):
        return self.__class__.__name__ + f'(noise_type={self.noise_type}, params={self.params})'

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

def get_cifar10_dataloaders(batch_size, noise_type='gaussian', noise_params={}):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    noise_transform = AddNoise(noise_type, noise_params)
    
    train_dataset = MyDataset(train=True, transform=transform, noise_transform=noise_transform)
    test_dataset = MyDataset(train=False, transform=transform, noise_transform=noise_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, test_loader