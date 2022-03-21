import torch
import os
import torchvision
from torchvision import trnasforms
from PIL import Image
# Define dataset class that inherits from torch.utils.data.Dataset and reads the images and labels from the given directory.
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                                    transforms.ToTensor()])
        self.image_paths = []
        self.labels = []
        for dir_path, dir_names, file_names in os.walk(self.data_dir):
            for file_name in file_names:
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    self.image_paths.append(os.path.join(dir_path, file_name))
                    self.labels.append(int(dir_path.split('/')[-1])) # must be modified
        self.labels = torch.tensor(self.labels)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.image_paths)

class ImageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                                    transforms.ToTensor()])
        self.image_paths = []
        self.labels = []
        for dir_path, dir_names, file_names in os.walk(self.data_dir):
            for file_name in file_names:
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    self.image_paths.append(os.path.join(dir_path, file_name))
                    self.labels.append(int(dir_path.split('/')[-1])) # must be modified
        self.labels = torch.tensor(self.labels)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.image_paths)