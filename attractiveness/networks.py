import torchvision
from torch import nn
import torch
import torch.nn.functional as F
# Define model class

class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, 1)
        self.resnet = resnet
    def forward(self, x):
        x = self.resnet(x)
        # x = F.sigmoid(x) * 5 # scale the output to be between 0 and 5
        return x.squeeze()