import torch
from dataset import ImageDatasetTest
import cv2
import os
from networks import ResNet18
from test import test
import argparse
import utils
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
# def get_opt():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--output_dir", default="./output/test")
#     parser.add_argument("--gpu_ids", default="0")
#     parser.add_argument("--num_images", default="10", type=int)
#     parser.add_argument("--interval", default="30", type=int)
#     parser.add_argument('-j', '--workers', type=int, default=4)
#     parser.add_argument('-b', '--batch-size', type=int, default=16)
#     parser.add_argument("--test_dataroot", default="/home/sangyunlee/AI_Model/attractiveness/video_images")
#     parser.add_argument("--video_path", default="/home/sangyunlee/AI_Model/attractiveness/stopwatch.avi")
#     parser.add_argument('--ckpt', type=str, help='save checkpoint infos')
  
    
#     opt = parser.parse_args()
#     return opt

def extract_images(video_path, save_path, num_images=10, interval = 30):
    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while success:
        success, image = vidcap.read()
        if count % interval == 0:
            cv2.imwrite(os.path.join(save_path, f"{count // interval}.jpg"), image)     # save frame as JPEG file
            if count // interval == (num_images - 1):
                break
        count += 1
# Use like this:
# extract_images("/home/sangyunlee/AI_Model/attractiveness/stopwatch.avi", "./test_images")

def test(model, test_loader, output_dir):
    model.eval()
    model.cuda()

    for i, (image, path) in enumerate(test_loader):
        image = image.cuda()
        image.requires_grad_()

        output, feat = model(image, return_feat=True)
        with open(os.path.join(output_dir, "output.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")
        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()
        
        grayscale_cams = feat

        

        grayscale_cams = F.interpolate(grayscale_cams, size=image.shape[2:], mode='bicubic', align_corners=True).squeeze()
        for j in range(len(output)):
            # Select top/bottom 10% pixels
            grayscale_cam = grayscale_cams[j]
            print("grayscale_cam min max", grayscale_cam.min(), grayscale_cam.max())
            k = grayscale_cam.shape[-1] * grayscale_cam.shape[-2] // 10
            top_k, _ = torch.topk(grayscale_cam.flatten(), k=k)
            bottom_k, _ = torch.topk(-grayscale_cam.flatten(), k=k)
            bottom_k *= -1
        

            grayscale_cam_top = torch.maximum(top_k.min(), grayscale_cam)
            grayscale_cam_bot = torch.minimum(bottom_k.max(), grayscale_cam)

            # print("grayscale_cam_pos.shape", grayscale_cam_pos.shape, "grayscale_cam_neg.shape", grayscale_cam_neg.shape)
        
            grayscale_cam_bot = utils.normalize(grayscale_cam_bot).cpu().detach().numpy()
            grayscale_cam_top = utils.normalize(grayscale_cam_top).cpu().detach().numpy()
            grayscale_cam = utils.normalize(grayscale_cam).cpu().detach().numpy()
            
            img = (image[j]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            visualization = show_cam_on_image(img, grayscale_cam_top, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            visualization.save(os.path.join(output_dir, os.path.basename(path[j]))) 

            
        print(i)

def run_test(video_path, test_dataroot, num_images, interval, ckpt, output_dir, batch_size, gpu_ids = 0):
    """
    video_path: path to the video
    test_dataroot: path where the extracted images are saved
    num_images: number of images to extract
    interval: interval between two consecutive images
    ckpt: path to the checkpoint
    output_dir: path where the output files are saved
    
    """
    
    extract_images(video_path, test_dataroot, num_images, interval)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # Define model
    model = ResNet18()
    # model = resnext50_32x4d()
    # Load checkpoint
    model.load_state_dict(torch.load(ckpt))
    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # Inference
    test(model, test_loader, output_dir)

if __name__ == '__main__':
    run_test(video_path = "/home/sangyunlee/AI_Model/attractiveness/stopwatch.avi",
             test_dataroot = "/home/sangyunlee/AI_Model/attractiveness/test_images",
             num_images = 10,
             interval = 30,
             ckpt = "/home/sangyunlee/AI_Model/attractiveness/checkpoints/mask-reg/best_model.pth",
             output_dir = "/home/sangyunlee/AI_Model/attractiveness/output/test",
             batch_size = 16,
             gpu_ids = "0")