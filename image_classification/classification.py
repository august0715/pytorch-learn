from torchvision import models
import torch
 
dir(models)

import torch

# Option 1: passing weights param as string
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

# Option 2: passing weights param as enum
weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)