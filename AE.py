import torch
import os
from torch import optim
from torch.cuda.amp import GradScaler
from inference_utils import SSIMLoss,psnr,lpips_fn,save_img_tensor
from inference_models import get_init_noise, get_model,from_noise_to_image
from inference_image0 import get_image0
from predict import predict_image_tensor
import argparse
import numpy as np
import complexity
import cv2
from noise import get_cifar10_dataloaders
from unet_model import UNet
import multiprocessing


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AE=UNet().to(device)
    optimizer_autoencoder = optim.Adam(AE.parameters(), lr=0.0001)
    criterion_autoencoder = torch.nn.MSELoss()
    train_loader=get_cifar10_dataloaders()
    AE.train()
    for batch in train_loader:
        src = batch['src'].to(device=device, dtype=torch.float32)
        target = batch['target'].to(device=device, dtype=torch.float32)

        optimizer_autoencoder.zero_grad()
        output = autoencoder(src)
        loss_autoencoder = criterion_autoencoder(output, target)
        loss_autoencoder.backward()
        optimizer_autoencoder.step()

    print(f'Autoencoder Loss: {loss_autoencoder.item()}')
    
