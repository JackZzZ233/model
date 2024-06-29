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
dir_checkpoint = 'checkpoints/'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
    parser.add_argument("--distance_metric", default="l2", type=str, help="The path of dev set.")
    parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
    parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

    parser.add_argument("--lr", default=1e-2, type=float, help="")
    parser.add_argument("--dataset_index", default=None, type=int, help="")
    parser.add_argument("--bs", default=8, type=int, help="")
    parser.add_argument("--write_txt_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
    parser.add_argument("--strategy", default="mean", type=str, help="The path of dev set.")
    parser.add_argument("--mixed_precision", action="store_true", help="The path of dev set.")
    parser.add_argument("--sd_prompt", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")

    #filter and noise
    #parser.add_argument("--noise_type",default=None,type=str,help="")
    parser.add_argument("--mean",default=None,type=float,help="")
    parser.add_argument("--std",default=None,type=float,help="")
    parser.add_argument("--salt_prob",default=None,type=float,help="")
    parser.add_argument("--filter_type",default=None,type=str,help="")
    parser.add_argument("--kernel_size",default=None,type=float,help="")
    #

    #UNet
    parser.add_argument("--noise_type", default="gaussian", type=str, help="Type of noise to add (gaussian, salt_and_pepper, speckle, poisson)")
    parser.add_argument("--noise_mean", default=0.0, type=float, help="Mean for Gaussian or Speckle noise")
    parser.add_argument("--noise_std", default=0.1, type=float, help="Standard deviation for Gaussian or Speckle noise")
    parser.add_argument("--noise_amount", default=0.05, type=float, help="Amount of Salt and Pepper noise")
    parser.add_argument("--salt_vs_pepper", default=0.5, type=float, help="Ratio of salt vs pepper noise")

    parser.add_argument('-b', '--batchsize', metavar='B', type=int, nargs='?', default=16,help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learningrate', metavar='LR', type=float, nargs='?', default=0.0001,help='Learning rate', dest='lr')
    #

    args = parser.parse_args()

    noise_params = {
    'mean': args.noise_mean,
    'std': args.noise_std,
    'amount': args.noise_amount,
    'salt_vs_pepper': args.salt_vs_pepper
    }
    train_loader, val_loader = get_cifar10_dataloaders(args.batchsize, args.noise_type, noise_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    optimizer_Unet= optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_UNet = torch.nn.MSELoss()
    #

    args.cur_model = get_model(args.model_type,args.model_path_,args)
    image0, gt_noise = get_image0(args)
    image0 = image0.detach()
    image0_Unet=image0.clone()
    #image0=image0.squeeze(0).cpu().numpy().transpose(1, 2, 0)  
    #image0 = cv2.GaussianBlur(image0, (5, 5), 0)
    #image0 = torch.tensor(image0.transpose(2, 0, 1)).unsqueeze(0).cuda().float() 
    init_noise = get_init_noise(args,args.model_type,args.cur_model,bs=args.bs)

    if args.model_type in ["sd"]:
        cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
        optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
    elif args.model_type in ["sd_unet"]:
        args.cur_model.unet.eval()
        args.cur_model.vae.eval()
        cur_noise_0 = torch.nn.Parameter(torch.tensor(init_noise[0])).cuda()
        optimizer = torch.optim.Adam([cur_noise_0], lr=args.lr)
    else:
        cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
        optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
        
    if args.distance_metric == "l1":
        criterion = torch.nn.L1Loss(reduction='none')
    elif args.distance_metric == "l2":
        criterion = torch.nn.MSELoss(reduction='none')
    elif args.distance_metric == "ssim":
        criterion = SSIMLoss().cuda()
    elif args.distance_metric == "psnr":
        criterion = psnr
    elif args.distance_metric == "lpips":
        criterion = lpips_fn
        
    import time
    args.measure = float("inf")

    measureUnet = float("inf")
    

    if args.mixed_precision:
        scaler = GradScaler()
    for i in range(0,1000):
        start_time = time.time()

    #Unet
        global_step = 0
        model.train()
        Unet_epoch_loss = 0
    #

        print("step:",i)

        #Training Unet
        for batch in train_loader:
                src = batch['src']
                target = batch['target']
                src = src.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)

                optimizer_Unet.zero_grad()
                src_pred = model(src)
                loss_Unet = criterion_UNet(src_pred, target)
                Unet_epoch_loss += loss_Unet.item()
                loss_Unet.backward()
                optimizer_Unet.step()

                if global_step % 100 == 0:
                    print('Global step:', global_step, ' Loss:', loss_Unet.item())
                global_step += 1   
        #
    #loss of unet
        print(f'Epoch {i + 1}/{1000}, Loss: {Unet_epoch_loss / len(train_loader)}')
        image_Unet=image0_Unet.clone()
        model.eval()
        image_Unet=predict_image_tensor(model,image_Unet,device)
    #
        if  i % 50 == 0:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{i + 1}.pth')
            print(f'Checkpoint {i + 1} saved !')

        if args.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
                loss = criterion(image0,image).mean()

                loss_Unet=criterion_UNet(image0_Unet,image).mean()
        else:
            image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
            loss = criterion(image0.detach(),image).mean()

            loss_Unet=criterion_UNet(image0_Unet.detach(),image).mean()

        if i%100==0:
            epoch_num_str=str(i)
            with torch.no_grad():
    #save Unet image
                save_img_tensor(image0_Unet,"./result_imgs/image0_possible_ori"+epoch_num_str+"_"+".png")
    #
                save_img_tensor(image,"./result_imgs/image_cur_"+args.input_selection+"_"+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")

        min_value = criterion(image0,image).mean(-1).mean(-1).mean(-1).min()
        mean_value = criterion(image0,image).mean()

        min_value_Unet=criterion_UNet(image_Unet,image).mean(-1).mean(-1).mean(-1).min()
        mean_value_Unet=criterion_UNet(image_Unet,image).mean()

        if (args.strategy == "min") and (min_value < args.measure):
            args.measure = min_value

            measureUnet = torch.tensor(min_value_Unet)

        if (args.strategy == "mean") and (mean_value < args.measure):
            args.measure = mean_value

            measureUnet = torch.tensor(mean_value_Unet)

        print("lowest loss now:",args.measure.item())

        print("lowest Unet loss now:",measureUnet.item())

        if args.distance_metric == "lpips":
            loss = loss.mean()
        print("loss "+args.input_selection+" "+args.distance_metric+":",loss.item())
        
        #
        print("Unet Loss:",loss_Unet.item())
#
        if args.mixed_precision:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        print("time for one iter: ",end_time-start_time)
        torch.cuda.empty_cache()


    cv2_img0 = (image0.squeeze(0).permute(1, 2, 0).cpu().numpy()* 255).astype(np.uint8)
    cv2_img0 = cv2.cvtColor(cv2_img0, cv2.COLOR_BGR2GRAY)

    print("*"*80)
    print("final lowest loss: ",args.measure.item())

    print("final lowest Unet loss: ",args.measureUnet.item())

    print("2D-entropy-based complexity: ", complexity.calcEntropy2dSpeedUp(cv2_img0, 3, 3))

    if args.write_txt_path:
        with open(args.write_txt_path,"a") as f:
            f.write(str(args.measure.item())+"\n")

    if args.sd_prompt:
        save_img_tensor(image0,"./result_imgs/ORI_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
        save_img_tensor(image,"./result_imgs/last_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    if args.input_selection_url:
        save_img_tensor(image0,"./result_imgs/ORI_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
        save_img_tensor(image,"./result_imgs/last_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()