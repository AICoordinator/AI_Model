# python3 test.py --gpu_ids 0 --test_dataroot ./samples --ckpt ./checkpoints/8000.pth
import torch
from dataset import ImageDatasetTest
from networks import ResNet18
import argparse
import os
import torch.nn.functional as F
from PIL import Image
# Grad-CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from torch import nn
def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./output/test")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument("--test_dataroot", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/Images")

    parser.add_argument('--ckpt', type=str, help='save checkpoint infos')
  
    
    opt = parser.parse_args()
    return opt

def visualize(test_iter, model):
    image, label = test_iter.next()

def test(model, test_loader, opt):
    model.eval()
    model.cuda()
    # Check folder
    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
        os.mkdir(os.path.join(opt.output_dir, "saliency"))
        # os.mkdir(os.path.join(opt.output_dir, "gradcam"))
        os.mkdir(os.path.join(opt.output_dir, "cam-pos"))
        os.mkdir(os.path.join(opt.output_dir, "cam-neg"))
        os.mkdir(os.path.join(opt.output_dir, "cam-rgb"))
        os.mkdir(os.path.join(opt.output_dir, "cam"))
        
    for i, (image, path) in enumerate(test_loader):
        image = image.cuda()
        image.requires_grad_()

        output, feat = model(image, return_feat = True)
        with open(os.path.join(opt.output_dir, "output.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")
        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()
        # Visualize the saliency map
        for j in range(len(output)):
            saliency = torch.sum(image.grad[j].abs(), dim=0)
            saliency = F.relu(saliency)
            saliency = saliency/saliency.max()
            # Save the saliency map
            saliency = saliency.cpu().detach().numpy() * 255
            saliency = Image.fromarray(saliency)
            # Convert to L
            saliency = saliency.convert('L')

            # saliency = saliency * image[j]
            # saliency = torchvision.transforms.ToPILImage()((saliency * 0.5 + 0.5) * 255)
            
            saliency.save(os.path.join(opt.output_dir, "saliency", os.path.basename(path[j])))
        # Visualize the Class Activation Map
        weighted_feat = model.conv1x1(feat)
        cams = torch.sum(weighted_feat, dim=1, keepdim=True)
        cams_pos = F.relu(cams)
        cams_neg = F.relu(-cams)
        # print("cams.shape", cams.shape)
        cams = F.interpolate(cams, size=image.shape[2:], mode='bilinear', align_corners=True).squeeze()
        cams_pos = F.interpolate(cams_pos, size=image.shape[2:], mode='bilinear', align_corners=True).squeeze()
        cams_neg = F.interpolate(cams_neg, size=image.shape[2:], mode='bilinear', align_corners=True).squeeze()
        # print("mean, std, min, max of cams: ", cams.mean(), cams.std(), cams.min(), cams.max())
        for j in range(len(output)):
            cam = cams[j]
            cam_pos = cams_pos[j]
            cam_neg = cams_neg[j]

            cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam = cam.cpu().detach().numpy()
            cam_pos = (cam_pos - cam_pos.min()) / (cam_pos.max() - cam_pos.min())
            cam_pos = cam_pos.cpu().detach().numpy()
            cam_neg = (cam_neg - cam_neg.min()) / (cam_neg.max() - cam_neg.min())
            cam_neg = cam_neg.cpu().detach().numpy()

            img = image[j].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            
            cam_rgb = show_cam_on_image(img, cam, use_rgb=True)
            # cam_rgb = img * cam.reshape(cam.shape[0], cam.shape[1], 1)
            cam_pos = img * cam_pos.reshape(cam_pos.shape[0], cam_pos.shape[1], 1)
            cam_neg = img * cam_neg.reshape(cam_neg.shape[0], cam_neg.shape[1], 1)

            cam_rgb = (cam_rgb * 255).astype(np.uint8)
            cam_rgb = Image.fromarray(cam_rgb)
            cam_rgb.save(os.path.join(opt.output_dir, "cam-rgb", os.path.basename(path[j])))
            print("mean, std, min, max of cam: ", cam.mean(), cam.std(), cam.min(), cam.max())

            cam_pos = (cam_pos * 255).astype(np.uint8)
            cam_pos = Image.fromarray(cam_pos)
            cam_pos.save(os.path.join(opt.output_dir, "cam-pos", os.path.basename(path[j])))

            cam_neg = (cam_neg * 255).astype(np.uint8)
            cam_neg = Image.fromarray(cam_neg)
            cam_neg.save(os.path.join(opt.output_dir, "cam-neg", os.path.basename(path[j])))

            cam = (cam*255).astype(np.uint8)
            cam_img = Image.fromarray(cam).convert('L')
            cam_img.save(os.path.join(opt.output_dir, "cam", os.path.basename(path[j])))


        # Visualize the Grad-CAM
        weight = nn.AdaptiveAvgPool2d((1, 1))(feat.grad)
        grayscale_cams = torch.sum(weight * feat, dim=1, keepdim=True)
        grayscale_cams_neg = F.relu(-grayscale_cams)
        grayscale_cams = F.relu(grayscale_cams)
        
        print("grayscale_cams_neg.min/max/mean/std: ", grayscale_cams_neg.min(), grayscale_cams_neg.max(), grayscale_cams_neg.mean(), grayscale_cams_neg.std())

        grayscale_cams = F.interpolate(grayscale_cams, size=image.shape[2:], mode='bilinear', align_corners=True).squeeze()
        grayscale_cams = grayscale_cams.cpu().detach().numpy()
        grayscale_cams_neg = F.interpolate(grayscale_cams_neg, size=image.shape[2:], mode='bilinear', align_corners=True).squeeze()
        grayscale_cams_neg = grayscale_cams_neg.cpu().detach().numpy()
        for j in range(len(output)):
            grayscale_cam = (grayscale_cams[j] - grayscale_cams[j].min()) / (grayscale_cams[j].max() - grayscale_cams[j].min())
            grayscale_cam_neg = (grayscale_cams_neg[j] - grayscale_cams_neg[j].min()) / (grayscale_cams_neg[j].max() - grayscale_cams_neg[j].min())
            img = (image[j]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            visualization.save(os.path.join(opt.output_dir, "gradcam-pos", os.path.basename(path[j]))) 

            gradcam_img = (grayscale_cam*255).astype(np.uint8)
            print(f"gradcam_img.mean(): {gradcam_img.mean()}, gradcam_img.std(): {gradcam_img.std()}, gradcam_img.min(): {gradcam_img.min()}, gradcam_img.max(): {gradcam_img.max()}")
            gradcam_img = Image.fromarray(gradcam_img).convert('L')
            gradcam_img.save(os.path.join(opt.output_dir, "gradcam", os.path.basename(path[j])))

            visualization = show_cam_on_image(img, grayscale_cam_neg, use_rgb=True)
            visualization = Image.fromarray(visualization)
            visualization.save(os.path.join(opt.output_dir, "gradcam-neg", os.path.basename(path[j])))
    
        print(i)


    
if __name__ == '__main__':
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # Define model
    model = ResNet18()
    # Load checkpoint
    model.load_state_dict(torch.load(opt.ckpt))
    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=opt.test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    # Train the model
    test(model, test_loader, opt)
