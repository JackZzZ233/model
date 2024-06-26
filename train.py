import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from unet_model import UNet
from noise import get_cifar10_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

dir_checkpoint = 'checkpoints/'


    
def train_model(net,
              device,
              epochs=1000,
              batch_size=16,
              lr=0.0001,
              save_cp=True,
              noise_mean=0.0,
              noise_std=0.1,
              out_channel=3):
    # 数据加载器
    train_loader, val_loader = get_cifar10_dataloaders(batch_size, noise_mean, noise_std)
    
    
    # 优化器和损失函数
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        global_step = 0
        for batch in train_loader:
            src = batch['src']
            target = batch['target']
            src = src.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            src_pred = net(src)

            loss = criterion(src_pred, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print('Global step:', global_step, ' Loss:', loss.item())
            global_step += 1

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')

        # 验证模型
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src']
                target = batch['target']
                src = src.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)

                src_pred = net(src)
                loss = criterion(src_pred, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        # 调整学习率
        scheduler.step(val_loss)

        if save_cp and epoch % 10 == 0:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved !')

    torch.save(net.state_dict(), 'model.pth')
    print('Model saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('-b', '--batchsize', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learningrate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-p', '--pretrained', dest='pretrained', type=bool, default=False,
                        help='Whether to load pretrained model from internet')
    parser.add_argument('-n', '--noise_mean', type=float, default=0.0,
                        help='Mean of the Gaussian noise')
    parser.add_argument('-s', '--noise_std', type=float, default=0.1,
                        help='Standard deviation of the Gaussian noise')
    parser.add_argument('-o', '--out_channel', dest='out_channel', type=int, default=3,
                        help='Channel of the images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    


    model = train_model(model,device, epochs=1000, batch_size=16, lr=0.0001,save_cp=True,noise_mean=0.0,noise_std=0.1,out_channel=3)
    torch.save(model.state_dict(), 'denoising_model.pth')
    try:
        train_model(net=model,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  noise_mean=args.noise_mean,
                  noise_std=args.noise_std,
                  out_channel=args.out_channel)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
