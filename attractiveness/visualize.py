# Read images and corresponding scores from a text file and visualize them.
from dataset import ImageDataset
import argparse
from torchvision.utils import save_image
from torchvision import transforms
import os
from PIL import Image
import torch
import numpy as np


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output/test/visualize")
    parser.add_argument("--dataroot", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/Images")
    parser.add_argument("--label_dir", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt")
    parser.add_argument("--display_count", type=int, default=100)

    
    opt = parser.parse_args()
    return opt
opt = get_opt()
# Check folder
if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
l = []
with open(opt.label_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        image_name = line[0]
        image_path = os.path.join(opt.dataroot, image_name)
        score = float(line[1])
        l.append((image_path, score))
l.sort(key=lambda x: x[1], reverse=True)

top = l[:opt.display_count]
bottom = l[-opt.display_count:]

# Visualize the top images
tmp = []
for path, score in top:
    image = Image.open(path)
    image_tensor = transforms.ToTensor()(image)
    tmp.append(image_tensor)
images = torch.stack(tmp)
save_image(images, os.path.join(opt.output_dir, "top.png"), nrow=int(opt.display_count**0.5))

# Visualize the bottom images
tmp = []
for path, score in bottom:
    image = Image.open(path)
    image_tensor = transforms.ToTensor()(image)
    tmp.append(image_tensor)
images = torch.stack(tmp)
save_image(images, os.path.join(opt.output_dir, "bottom.png"), nrow=int(opt.display_count**0.5))

