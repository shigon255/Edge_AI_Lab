# read in 0.9099_deit3_small_patch16_224.pth (pretrained DeiT-S) and show blocks

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader

# load model
model = torch.load('0.9099_deit3_small_patch16_224.pth', map_location='cpu')

# write output to txt file
with open('output.txt', 'w') as f:
    # write "print(model)" to file
    print(model, file=f)

