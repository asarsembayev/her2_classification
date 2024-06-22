import torch
import torch.nn as nn
from torchvision import models

def initialize_model(num_classes=4):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 4 classes

    return model
