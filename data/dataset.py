import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def pad_to_square(img, pad_value):  # 将图像填充为正方形
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True):
        self.img_files = []
        self.label_files = []
        
        # Check if list_path is a file or directory
        if os.path.isdir(list_path):
            # Get all images
            self.img_files = sorted(glob.glob(os.path.join(list_path, "*.jpg"))) + \
                             sorted(glob.glob(os.path.join(list_path, "*.png")))
        elif os.path.isfile(list_path):
            with open(list_path, "r") as file:
                self.img_files = file.readlines()
                self.img_files = [path.replace("\n", "") for path in self.img_files]
        
        # Assuming labels are in the same folder with .txt extension
        # Or you can adjust this logic to find labels in a parallel 'labels' directory
        self.label_files = [
            path.replace(".jpg", ".txt").replace(".png", ".txt")
            for path in self.img_files
        ]
        
        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.batch_count = 0

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        
        try:
            # Extract image
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return None, None, None

        # Handle images with less than 3 channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if False else (1, 1) 
        
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # Resize
        img = resize(img, self.img_size)
        
        # Labels
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        
        targets = None
        if os.path.exists(label_path):
            try:
                # class x_center y_center w h
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                
                # Extract coordinates for unpad
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                
                # Adjust for padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h
                
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes
            except Exception as e:
                # print(f"Could not read label '{label_path}'.")
                pass
        
        return img_path, img, targets

    def collate_fn(self, batch):
        # Filter out None values
        batch = [b for b in batch if b[0] is not None]
        if len(batch) == 0:
            return None, None, None
            
        paths, imgs, targets = list(zip(*batch))
        
        # Remove empty placeholders
        targets = [boxes for boxes in targets if boxes is not None]
        
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        
        if targets:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 6))
            
        imgs = torch.stack(imgs, 0)
        
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
