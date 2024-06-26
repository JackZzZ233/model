import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from unet_model import UNet
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def load_model(model_path, device, out_channel=3):
    model = UNet(n_channels=out_channel)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device=device)
    model.eval()
    return model

def preprocess_image(image_path, img_scale=1):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((int(img.height * img_scale), int(img.width * img_scale))),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    return img

def predict_image(model, img_tensor, device):
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = model(img_tensor)
    return output.cpu().numpy()

def predict_image_tensor(model, img_tensor, device):
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = model(img_tensor)
    return output

def save_output(output, output_path):
    output_img = np.transpose(output[0], (1, 2, 0))  # Reshape the output
    
    # Check output range and normalize if necessary
    output_min = output_img.min()
    output_max = output_img.max()
    
    print(f"Min pixel value: {output_min}")
    print(f"Max pixel value: {output_max}")
    
    # Normalize to range [0, 255] if needed
    if output_min < 0 or output_max > 1:
        output_img = (output_img - output_min) / (output_max - output_min)
    
    output_img = (output_img * 255).astype(np.uint8)
    output_img = Image.fromarray(output_img)
    output_img.save(output_path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device, out_channel=args.out_channel)
    
    img_tensor = preprocess_image(args.image_path, img_scale=args.scale)
    output = predict_image(model, img_tensor, device)
    
    save_output(output, args.output_path)
    print(f"Prediction saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict images using a trained VGG16UNet model')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-i', '--image-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('-s', '--scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-p', '--pretrained', type=bool, default=False, help='Load pretrained model from the internet')
    parser.add_argument('-c', '--out_channel', type=int, default=3, help='Output channels of the model')
    
    args = parser.parse_args()
    main(args)
