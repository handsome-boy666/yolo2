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
    c, h, w = img.shape  # 获取图像的通道数、高度、宽度
    dim_diff = np.abs(h - w)  # 计算高度和宽度的差值
    # 计算上下/左右的填充量（上/左 填充，下/右 填充）
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 确定填充维度：如果高度<=宽度则上下填充，否则左右填充
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # 执行填充操作，使用常数填充
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    # 将图像调整到指定尺寸（先增加batch维度，插值后再移除）
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True):
        self.img_files = []  # 存储图像文件路径的列表
        self.label_files = []  # 存储标签文件路径的列表
        
        # 检查输入路径是文件还是目录
        if os.path.isdir(list_path):
            # 如果是目录，获取所有jpg和png格式的图像文件
            self.img_files = sorted(glob.glob(os.path.join(list_path, "*.jpg"))) + \
                             sorted(glob.glob(os.path.join(list_path, "*.png")))
        elif os.path.isfile(list_path):
            # 如果是文件，读取文件中的每一行作为图像路径
            with open(list_path, "r") as file:
                self.img_files = file.readlines()
                self.img_files = [path.replace("\n", "") for path in self.img_files]
        
        # 生成对应的标签文件路径（假设标签与图像同目录，后缀为txt）
        # 也可以根据需要调整为从并行的'labels'目录中查找标签
        self.label_files = [
            path.replace(".jpg", ".txt").replace(".png", ".txt")
            for path in self.img_files
        ]
        
        self.img_size = img_size  # 图像目标尺寸
        self.augment = augment  # 是否启用数据增强
        self.multiscale = multiscale  # 是否启用多尺度训练
        self.batch_count = 0  # 批次计数器

    def __getitem__(self, index):
        # 获取指定索引的图像路径（处理索引超出范围的情况）
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        
        try:
            # 读取图像并转换为Tensor格式（转为RGB以确保3通道）
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"无法读取图像 '{img_path}'。")
            return None, None, None

        # 处理通道数不足3的图像（如灰度图）
        if len(img.shape) != 3:
            img = img.unsqueeze(0)  # 增加通道维度
            img = img.expand((3, img.shape[1:]))  # 扩展为3通道

        _, h, w = img.shape  # 获取原始图像的高度和宽度
        h_factor, w_factor = (h, w) if False else (1, 1)  # 尺寸缩放因子（当前未启用）
        
        # 将图像填充为正方形
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape  # 获取填充后的图像尺寸

        # 将图像调整到目标尺寸
        img = resize(img, self.img_size)
        
        # 获取对应的标签文件路径
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        
        targets = None  # 存储处理后的标签
        if os.path.exists(label_path):
            try:
                # 读取标签文件（格式：class x_center y_center w h）
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                
                # 计算未填充前的边界框坐标（反归一化）
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)  # 左上角x
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)  # 左上角y
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)  # 右下角x
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)  # 右下角y
                
                # 根据填充量调整边界框坐标
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                
                # 重新计算归一化的中心坐标和宽高
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w  # 归一化x中心
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h  # 归一化y中心
                boxes[:, 3] *= w_factor / padded_w  # 归一化宽度
                boxes[:, 4] *= h_factor / padded_h  # 归一化高度
                
                # 构建目标张量（格式：[batch_idx, class, x, y, w, h]）
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes
            except Exception as e:
                # print(f"无法读取标签 '{label_path}'。")
                pass
        
        return img_path, img, targets

    def collate_fn(self, batch):
        # 过滤掉batch中的None值（读取失败的样本）
        batch = [b for b in batch if b[0] is not None]
        if len(batch) == 0:
            return None, None, None
            
        # 解包batch中的路径、图像和标签
        paths, imgs, targets = list(zip(*batch))
        
        # 移除空的标签占位符
        targets = [boxes for boxes in targets if boxes is not None]
        
        # 为每个标签添加样本索引（batch内的序号）
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        
        # 拼接所有标签
        if targets:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 6))
            
        # 堆叠所有图像形成批量张量
        imgs = torch.stack(imgs, 0)
        
        return paths, imgs, targets

    def __len__(self):
        # 返回数据集的总样本数
        return len(self.img_files)