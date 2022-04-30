import torch
import os
import torchvision
from torchvision import transforms
from PIL import Image
# Define dataset class that inherits from torch.utils.data.Dataset and reads the images and labels from the given directory.
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, mask_dir, transform):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transforms.Compose(transform)
        self.transform_mask = transforms.Compose(transform[:-1])
        self.image_paths = []
        self.labels = {}
        with open(self.label_dir, 'r') as f:
            for line in f.readlines():
                fname, label = line.strip().split(' ')
                self.labels[fname] = torch.tensor(float(label))

        for dir_path, dir_names, file_names in os.walk(self.data_dir):
            for file_name in file_names:
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    if file_name in self.labels:
                        self.image_paths.append(os.path.join(dir_path, file_name))
              
        
        
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.labels[image_name]
        mask = Image.open(os.path.join(self.mask_dir, image_name))
        mask = self.transform_mask(mask)
        # print("mask shape:", mask.shape)
        return image, label, mask, image_path
    def __len__(self):
        return len(self.image_paths)


class ImageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.mask_dir = "/home/sangyunlee/dataset/SCUT-FBP5500_v2/mask"
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                                  ])
        self.transform_mask =  transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                                                  ])
        self.image_paths = []
        

        for dir_path, dir_names, file_names in os.walk(self.data_dir):
            for file_name in file_names:
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    self.image_paths.append(os.path.join(dir_path, file_name))
              
        
        
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = self.transform(image)
        # mask = Image.open(os.path.join(self.mask_dir, image_name))
        # mask = self.transform_mask(mask)
        return image, image_path#, mask
    def __len__(self):
        return len(self.image_paths)


