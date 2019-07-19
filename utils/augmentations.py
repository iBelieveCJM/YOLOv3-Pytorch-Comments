import torch
import numpy as np


def horisontal_flip(images, targets):
    flip_dim = images.dim()-1
    images = torch.flip(images, [flip_dim])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
