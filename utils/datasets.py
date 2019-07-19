import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.augmentations import horisontal_flip


def pad_to_square(img, pad_value):
    c, h, w  = img.shape
    dim_diff = int(np.abs(h - w))  # numpy.int64 ==> python int

    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    if image.dim()==3:
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    elif image.dim()==4:
        image = F.interpolate(image, size=size, mode="nearest")
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]

        # read the image and convert to PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))

        # Pad to square resolution
        img, _ = pad_to_square(img, 0)

        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.normalized_labels = normalized_labels

    def __getitem__(self, index):
        """running in the multithreading"""

        ##=== read image ===
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # read the image and convert to PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # record the shape of the orignial image
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution (called padded image)
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        ##=== read label and target bbox ===
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):

            # boxes.shape(num_bboxes_per_image, 5_vals), 
            # 5_vals=(label, center_x, center_y, weight, height)
            # the values are normalized to (0,1) (relative to original image)
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # 1.convert from center(xywh) to corner(xyxy)
            # 2.unnormalized the target bbox (relative to original image)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) # x of top-left corner
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2) # y of top-left corner
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2) # x of bottom-right corner
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2) # y of bottom-right corner

            # Adjust for added padding (relative to padded image)
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # convert from corner(xyxy) to center(xywh) relative to padding image
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w # center_x for padded image
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h # center_y for padded image
            boxes[:, 3] *= w_factor / padded_w       # width for padded image
            boxes[:, 4] *= h_factor / padded_h       # height for padded image

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        ##=== Apply augmentations ===
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):

        paths, imgs, targets = list(zip(*batch))

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[:, 0] = i

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        targets = torch.cat(targets, 0)

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
