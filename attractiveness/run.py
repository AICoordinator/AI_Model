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
from networks import BiSeNet
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
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
def crop_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img
    margin = 100
    for (x,y,w,h) in faces:
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w + 2 * margin, img.shape[1] - x)
        h = min(h + 2 * margin, img.shape[0] - y)
        img = img[y:y+h, x:x+w]
    return img


def extract_images(video_path, save_path, num_images=10, interval = 30):
    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    image = crop_face(image)
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
def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        print(f"part_colors, pi: {part_colors[pi]}, pi: {pi}")
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def test(model, model_mask, test_loader, output_dir):
    model.eval()
    model.cuda()
    model_mask.eval()
    model_mask.cuda()
    for i, (image, path, image_big) in enumerate(test_loader):
        image = image.cuda()
        image.requires_grad_()
        image_big = image_big.cuda()
        output, feat = model(image, return_feat=True)
        
        segmap = model_mask(image_big)[0]
        segmap = F.interpolate(segmap, size=(image.shape[2], image.shape[3]), mode='nearest')
        segmap = segmap.detach().argmax(1).unsqueeze(1)

        mask_skin = (segmap == 1).type(torch.float32) + (segmap == 2).type(torch.float32) + (segmap == 3).type(torch.float32)
        mask_nose = (segmap == 10).type(torch.float32)
        mask_eyes = (segmap == 4).type(torch.float32) + (segmap == 5).type(torch.float32)
        mask_ears = (segmap == 7).type(torch.float32) + (segmap == 8).type(torch.float32)
        mask_mouth = (segmap == 11).type(torch.float32) + (segmap == 12).type(torch.float32) + (segmap == 13).type(torch.float32)
        mask_hair = (segmap == 17).type(torch.float32)
        mask_others = torch.ones_like(mask_skin) - mask_skin - mask_nose - mask_eyes - mask_ears - mask_mouth - mask_hair

        region_skin = mask_skin * (image * 0.5 + 0.5)
        region_nose = mask_nose * (image * 0.5 + 0.5)
        region_eyes = mask_eyes * (image * 0.5 + 0.5)
        region_ears = mask_ears * (image * 0.5 + 0.5)
        region_mouth = mask_mouth * (image * 0.5 + 0.5)
        region_hair = mask_hair * (image * 0.5 + 0.5)

        # Save regions
        save_image(region_skin, os.path.join(output_dir, "skin.jpg"))
        save_image(region_nose, os.path.join(output_dir, "nose.jpg"))
        save_image(region_eyes, os.path.join(output_dir, "eyes.jpg"))
        save_image(region_ears, os.path.join(output_dir, "ears.jpg"))
        save_image(region_mouth, os.path.join(output_dir, "mouth.jpg"))
        save_image(region_hair, os.path.join(output_dir, "hair.jpg"))

        print(f"feat.shape : {feat.shape}, mask_skin.shape : {mask_skin.shape}")
        

        with open(os.path.join(output_dir, "output.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")
        
        
        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()
        
        grayscale_cams = feat
        print(f"grayscale_cams.shape : {grayscale_cams.shape}")
        print(f"mask_skin.shape : {mask_skin.shape}")
        

        grayscale_cams = F.interpolate(grayscale_cams, size=image.shape[2:], mode='bicubic', align_corners=True)
        print(f"grayscale_cam  s.shape : {grayscale_cams.shape}")

        offset = 20
        score_skin = offset - (grayscale_cams * mask_skin).sum(dim = (1,2,3)) / mask_skin.sum(dim = (1,2,3))
        score_nose = offset - (grayscale_cams * mask_nose).sum(dim = (1,2,3)) / mask_nose.sum(dim = (1,2,3))
        score_eyes = offset - (grayscale_cams * mask_eyes).sum(dim = (1,2,3)) / mask_eyes.sum(dim = (1,2,3))
        score_ears = offset - (grayscale_cams * mask_ears).sum(dim = (1,2,3)) / mask_ears.sum(dim = (1,2,3))
        score_mouth = offset - (grayscale_cams * mask_mouth).sum(dim = (1,2,3)) / mask_mouth.sum(dim = (1,2,3))
        score_hair = offset - (grayscale_cams * mask_hair).sum(dim = (1,2,3)) / mask_hair.sum(dim = (1,2,3))

        # if nan, assign 5
        score_skin[torch.isnan(score_skin)] = offset
        score_nose[torch.isnan(score_nose)] = offset
        score_eyes[torch.isnan(score_eyes)] = offset
        score_ears[torch.isnan(score_ears)] = offset
        score_mouth[torch.isnan(score_mouth)] = offset
        score_hair[torch.isnan(score_hair)] = offset

        std_skin = grayscale_cams[mask_skin.type(torch.bool)].std()
        std_nose = grayscale_cams[mask_nose.type(torch.bool)].std()
        std_eyes = grayscale_cams[mask_eyes.type(torch.bool)].std()
        std_ears = grayscale_cams[mask_ears.type(torch.bool)].std()
        std_mouth = grayscale_cams[mask_mouth.type(torch.bool)].std()
        std_hair = grayscale_cams[mask_hair.type(torch.bool)].std()

        # If std is less than 1e-4, std = 1
        std_skin[std_skin < 1e-4] = 1
        std_nose[std_nose < 1e-4] = 1
        std_eyes[std_eyes < 1e-4] = 1
        std_ears[std_ears < 1e-4] = 1
        std_mouth[std_mouth < 1e-4] = 1
        std_hair[std_hair < 1e-4] = 1


        # print std
        print(f"std_skin : {std_skin}")
        print(f"std_nose : {std_nose}")
        print(f"std_eyes : {std_eyes}")
        print(f"std_ears : {std_ears}")
        print(f"std_mouth : {std_mouth}")
        print(f"std_hair : {std_hair}")

        # Check if grayscale_cams contains nan
        if torch.isnan(grayscale_cams).any():
            print("grayscale_cams contains nan")
            print(grayscale_cams)
            raise NotImplementedError

        with open(os.path.join(output_dir, "output_skin.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_skin[j].item()) + "\n")
        with open(os.path.join(output_dir, "output_nose.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_nose[j].item()) + "\n")
        with open(os.path.join(output_dir, "output_eyes.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_eyes[j].item()) + "\n")
        with open(os.path.join(output_dir, "output_ears.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_ears[j].item()) + "\n")
        with open(os.path.join(output_dir, "output_mouth.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_mouth[j].item()) + "\n")
        with open(os.path.join(output_dir, "output_hair.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(score_hair[j].item()) + "\n")
        print(f"grayscale_cams.shape : {grayscale_cams.shape}")
        
        score_mean_skin = (offset - score_skin).mean()
        score_mean_nose = (offset - score_nose).mean()
        score_mean_eyes = (offset - score_eyes).mean()
        score_mean_ears = (offset - score_ears).mean()
        score_mean_mouth = (offset - score_mouth).mean()
        score_mean_hair = (offset - score_hair).mean()

        print(f"score_mean_skin: {score_mean_skin}")
        print(f"score_mean_nose: {score_mean_nose}")
        print(f"score_mean_eyes: {score_mean_eyes}")
        print(f"score_mean_ears: {score_mean_ears}")
        print(f"score_mean_mouth: {score_mean_mouth}")
        print(f"score_mean_hair: {score_mean_hair}")

        print("start --------------------------")
        for j in range(len(output)):
            # Select top/bottom 10% pixels
            grayscale_cam = grayscale_cams[j]
            print(f"grayscale_cam.shape : {grayscale_cam.shape}")
            # print("grayscale_cam min max", grayscale_cam.min(), grayscale_cam.max())
            # k = grayscale_cam.shape[-1] * grayscale_cam.shape[-2] // 10
            # top_k, _ = torch.topk(grayscale_cam.flatten(), k=k)
            # bottom_k, _ = torch.topk(-grayscale_cam.flatten(), k=k)
            # bottom_k *= -1
        

            # grayscale_cam_top = torch.maximum(top_k.min(), grayscale_cam)
            # grayscale_cam_bot = torch.minimum(bottom_k.max(), grayscale_cam)

            # print("grayscale_cam_pos.shape", grayscale_cam_pos.shape, "grayscale_cam_neg.shape", grayscale_cam_neg.shape)
        
            # grayscale_cam_bot = utils.normalize(grayscale_cam_bot).cpu().detach().numpy()
            # grayscale_cam_top = utils.normalize(grayscale_cam_top).cpu().detach().numpy()
            # grayscale_cam = utils.normalize(grayscale_cam).cpu().detach().numpy()
            



            # print(f"skin: {(grayscale_cam[mask_skin[j].type(torch.bool)]).mean()}")
            # print(f"nose: {(grayscale_cam[mask_nose[j].type(torch.bool)]).mean()}")
            # print(f"eyes: {(grayscale_cam[mask_eyes[j].type(torch.bool)]).mean()}")
            # print(f"ears: {(grayscale_cam[mask_ears[j].type(torch.bool)]).mean()}")
            # print(f"mouth: {(grayscale_cam[mask_mouth[j].type(torch.bool)]).mean()}")
            # print(f"hair: {(grayscale_cam[mask_hair[j].type(torch.bool)]).mean()}")
            
            
            grayscale_cam -= mask_skin[j] * score_mean_skin
            grayscale_cam -= mask_nose[j] * score_mean_nose
            grayscale_cam -= mask_eyes[j] * score_mean_eyes
            grayscale_cam -= mask_ears[j] * score_mean_ears
            grayscale_cam -= mask_mouth[j] * score_mean_mouth
            grayscale_cam -= mask_hair[j] * score_mean_hair

            
            # Divide by std

            

            grayscale_cam[mask_skin[j].type(torch.bool)] = grayscale_cam[mask_skin[j].type(torch.bool)] / std_skin
            grayscale_cam[mask_nose[j].type(torch.bool)] = grayscale_cam[mask_nose[j].type(torch.bool)] / std_nose
            grayscale_cam[mask_eyes[j].type(torch.bool)] = grayscale_cam[mask_eyes[j].type(torch.bool)] / std_eyes
            # grayscale_cam[mask_ears[j].type(torch.bool)] = grayscale_cam[mask_ears[j].type(torch.bool)] / std_ears
            grayscale_cam[mask_mouth[j].type(torch.bool)] = grayscale_cam[mask_mouth[j].type(torch.bool)] / std_mouth
            grayscale_cam[mask_hair[j].type(torch.bool)] = grayscale_cam[mask_hair[j].type(torch.bool)] / std_hair


            print(f"skin: {(grayscale_cam[mask_skin[j].type(torch.bool)]).mean()}")
            print(f"nose: {(grayscale_cam[mask_nose[j].type(torch.bool)]).mean()}")
            print(f"eyes: {(grayscale_cam[mask_eyes[j].type(torch.bool)]).mean()}")
            print(f"ears: {(grayscale_cam[mask_ears[j].type(torch.bool)]).mean()}")
            print(f"mouth: {(grayscale_cam[mask_mouth[j].type(torch.bool)]).mean()}")
            print(f"hair: {(grayscale_cam[mask_hair[j].type(torch.bool)]).mean()}")
            


            

            print(f"---------------------------")
            grayscale_cam = utils.normalize(grayscale_cam)
            grayscale_cam[mask_others[j].type(torch.bool)] = 0
            # save_image(grayscale_cam, f"{output_dir}/{j}_map.jpg")
            grayscale_cam = grayscale_cam.cpu().detach().numpy().squeeze()
            img = (image[j]).cpu().detach().numpy().transpose(1,2,0) * 0.5 + 0.5
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            visualization.save(os.path.join(output_dir, os.path.basename(path[j]))) 

            
        print(i)

def run_test(video_path, test_dataroot, num_images, interval, ckpt, ckpt_mask, output_dir, batch_size, gpu_ids = 0):
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
    model_mask = BiSeNet(n_classes=19)
    # Load checkpoint
    model.load_state_dict(torch.load(ckpt))
    model_mask.load_state_dict(torch.load(ckpt_mask))
    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # Inference
    test(model, model_mask, test_loader, output_dir)

if __name__ == '__main__':
    run_test(video_path = "/home/sangyunlee/AI_Model/attractiveness/test_video4.mp4",
             test_dataroot = "/home/sangyunlee/AI_Model/attractiveness/test_images4",
             num_images = 10,
             interval = 30,
             ckpt = "/home/sangyunlee/AI_Model/attractiveness/checkpoints/mask-reg/best_model.pth",
             ckpt_mask = "/home/sangyunlee/face-parsing.PyTorch/res/cp/79999_iter.pth",
             output_dir = "/home/sangyunlee/AI_Model/attractiveness/output/test",
             batch_size = 16,
             gpu_ids = "0")