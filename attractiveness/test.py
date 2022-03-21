import torchvision
from torch import nn
import torch
from dataset import ImageDataset
from tensorboardX import SummaryWriter
from torchvision import transforms
from networks import RegressionNetwork
import argparse
import os

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./output/test")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument("--test_dataroot", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/Images")
    parser.add_argument("--test_label_dir", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt")

    parser.add_argument('--ckpt', type=str, help='save checkpoint infos')
  
    
    opt = parser.parse_args()
    return opt

def visualize(test_iter, model):
    image, label = test_iter.next()

def test(model, test_loader, opt):
    model.eval()
    model.cuda()
    # Check folder
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    for i, (image, _, path) in enumerate(test_loader):
        image = image.cuda()

        with torch.no_grad():
            output = model(image)
            output = output.cpu().numpy()
        with open(os.path.join(opt.output_dir, "output.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")
    
if __name__ == '__main__':
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # Define model
    model = RegressionNetwork()
    # Load checkpoint
    model.load_state_dict(torch.load(opt.ckpt))
    # Define dataloader
    test_dataset = ImageDataset(data_dir=opt.test_dataroot, label_dir=opt.test_label_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    # Train the model
    test(model, test_loader, opt)
