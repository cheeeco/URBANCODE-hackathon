from torchvision import io, utils
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
import PIL.Image
import torch
from torchvision import tv_tensors

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(contrast=0.5))
        transforms.append(T.RandomRotation(5))
        transforms.append(T.RandomPerspective(distortion_scale=0.2, p=0.8))
        transforms.append(T.RandomIoUCrop(min_scale=0.8))
        transforms.append(T.SanitizeBoundingBoxes())

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)