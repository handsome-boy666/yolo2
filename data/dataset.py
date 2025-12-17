"""YOLOv2 数据集模块
====================

此模块实现用于 YOLOv2 的数据集类与批处理函数，核心功能包括：
- 读取图像与 YOLO 标签（class x y w h，归一化到 [0,1]）
- Letterbox（居中填充为正方形）与缩放到指定尺寸
- 标签坐标随几何变换同步映射
- Anchor 多框编码；支持 K-Means（IoU 距离）自动聚类并缓存
- 训练与测试的 `collate_fn`
"""

import math
import os
import time
import random
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

EPS: float = 1e-6  # 避免除零错误


class MyDataset(Dataset):
    def __init__(self, data_dir, if_train, base_size: int = 416):
        print(f"初始化数据集: {data_dir}")
        self.data_dir = data_dir
        self.img_size = base_size
        self.if_train = if_train

        if if_train:
            self.image_dir = os.path.join(data_dir, 'train/images')
            self.label_dir = os.path.join(data_dir, 'train/labels')
        else:
            self.image_dir = os.path.join(data_dir, 'test/images')
            self.label_dir = os.path.join(data_dir, 'test/labels')
        
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError("数据集目录不存在")

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in img_exts]  # 收集所有图片文件
        label_names = {os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.lower().endswith('.txt')}  # 收集所有标签文件（不包含扩展名）

        self.samples: List[Tuple[str, str]] = []  # 存储图片路径与标签路径的元组
        for img_file in sorted(image_files):
            name = os.path.splitext(img_file)[0]  # 图片文件名（不包含扩展名）
            if name in label_names:
                self.samples.append((
                    os.path.join(self.image_dir, img_file),
                    os.path.join(self.label_dir, name + '.txt')
                ))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"在 '{self.image_dir}' 与 '{self.label_dir}' 下未找到配对的图片与标签文件")

        # 构建数据集变换
        self.transform = self._build_transform()
        print(f"数据集样本数: {len(self.samples)}")
        print(f"默认图像尺寸: {self.img_size}")

    def _build_transform(self):
        """
        构建图像变换管线
        """
        if self.if_train:
            # 训练时加入颜色抖动：亮度、对比度、饱和度、色调
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        """返回数据集样本数"""
        return len(self.samples)

    def set_img_size(self, img_size: int):
        """设置图像尺寸"""
        img_size = img_size if img_size % 32 == 0 else math.ceil(img_size / 32) * 32
        self.img_size = img_size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        获取一个样本

        Args:
            index: 样本索引

        Returns:
            img_tensor: 变换后的图像张量 (3, H, W)
            target_tensor: 变换后的标签张量 (N, 5), 每行为 [class, x, y, w, h]
        """
        img_path, label_path = self.samples[index]

        # 1. 读取图像
        img = Image.open(img_path).convert('RGB')
        w_orig, h_orig = img.size   # 原始图像尺寸

        # 2. 读取并解析标签
        boxes = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([cls_id, x_c, y_c, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)

        # 3. 数据增强 (仅训练)
        if self.if_train:
            img, boxes = self._random_crop(img, boxes)

        # 4. 图像预处理 (Letterbox)
        img_tensor, boxes = self._letterbox(img, boxes, self.img_size)

        return img_tensor, boxes

    def _random_crop(self, img: Image.Image, boxes: Tensor) -> Tuple[Image.Image, Tensor]:
        """
        随机裁剪图像并调整标签
        """
        if random.random() < 0.5:
            return img, boxes
            
        w, h = img.size
        
        # 随机裁剪比例 (0.4 ~ 1.0)
        min_scale = 0.4
        scale = random.uniform(min_scale, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        
        # 随机裁剪位置
        dx = random.randint(0, w - nw)
        dy = random.randint(0, h - nh)
        
        # 裁剪图像
        img = img.crop((dx, dy, dx + nw, dy + nh))
        
        # 调整标签
        if len(boxes) > 0:
            # 将归一化坐标转换为绝对坐标
            boxes[:, 1] *= w
            boxes[:, 2] *= h
            boxes[:, 3] *= w
            boxes[:, 4] *= h
            
            # 平移坐标 (减去裁剪偏移)
            boxes[:, 1] -= dx
            boxes[:, 2] -= dy
            
            # 过滤：中心点必须在裁剪区域内
            center_x = boxes[:, 1]
            center_y = boxes[:, 2]
            mask = (center_x >= 0) & (center_x < nw) & (center_y >= 0) & (center_y < nh)
            boxes = boxes[mask]
            
            if len(boxes) > 0:
                # 将坐标限制在裁剪图像范围内 (Clamp)
                # 转换回 x1, y1, x2, y2
                x1 = boxes[:, 1] - boxes[:, 3] / 2
                y1 = boxes[:, 2] - boxes[:, 4] / 2
                x2 = boxes[:, 1] + boxes[:, 3] / 2
                y2 = boxes[:, 2] + boxes[:, 4] / 2
                
                x1.clamp_(0, nw)
                y1.clamp_(0, nh)
                x2.clamp_(0, nw)
                y2.clamp_(0, nh)
                
                # 转换回 cx, cy, w, h
                boxes[:, 1] = (x1 + x2) / 2
                boxes[:, 2] = (y1 + y2) / 2
                boxes[:, 3] = x2 - x1
                boxes[:, 4] = y2 - y1
                
                # 过滤掉过小的框 (例如长或宽小于 5 像素)
                mask = (boxes[:, 3] > 5) & (boxes[:, 4] > 5)
                boxes = boxes[mask]
                
                # 重新归一化到新尺寸
                boxes[:, 1] /= nw
                boxes[:, 2] /= nh
                boxes[:, 3] /= nw
                boxes[:, 4] /= nh
                
        return img, boxes

    def _letterbox(self, img: Image.Image, boxes: Tensor, new_shape: int) -> Tuple[Tensor, Tensor]:
        """
        调整图像大小并进行填充，保持纵横比（Letterbox），同时调整标签坐标

        Args:
            img: 原始 PIL 图像
            boxes: 原始标签 (N, 5), [class, x, y, w, h], 归一化坐标
            new_shape: 目标尺寸 (正方形边长)

        Returns:
            img_tensor: 变换后的图像张量
            new_boxes: 变换后的标签张量
        """
        w, h = img.size
        scale = min(new_shape / w, new_shape / h)
        nw, nh = int(w * scale), int(h * scale)

        # 调整图像大小
        img_resized = img.resize((nw, nh), Image.BICUBIC)

        # 创建新图像并填充随机颜色
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        new_img = Image.new('RGB', (new_shape, new_shape), random_color)
        
        # 计算偏移量，使其居中
        dx = (new_shape - nw) // 2
        dy = (new_shape - nh) // 2
        new_img.paste(img_resized, (dx, dy))

        # 转换为 Tensor
        img_tensor = self.transform(new_img)

        # 调整标签坐标
        new_boxes = boxes.clone()
        if len(boxes) > 0:
            # 将归一化坐标转换为原始像素坐标
            boxes[:, 1] = boxes[:, 1] * w
            boxes[:, 2] = boxes[:, 2] * h
            boxes[:, 3] = boxes[:, 3] * w
            boxes[:, 4] = boxes[:, 4] * h

            # 缩放
            boxes[:, 1:] *= scale

            # 平移
            boxes[:, 1] += dx
            boxes[:, 2] += dy

            # 归一化到新尺寸
            new_boxes[:, 1] = boxes[:, 1] / new_shape
            new_boxes[:, 2] = boxes[:, 2] / new_shape
            new_boxes[:, 3] = boxes[:, 3] / new_shape
            new_boxes[:, 4] = boxes[:, 4] / new_shape

            # 裁剪超出范围的框 (可选，通常 YOLO 数据集标注是正确的)
            new_boxes[:, 1:].clamp_(0, 1)

        return img_tensor, new_boxes


    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        自定义的 collate_fn，用于 DataLoader

        Args:
            batch: 一个 batch 的样本列表，每个样本为 (img_tensor, target_tensor)

        Returns:
            images: 堆叠后的图像张量 (B, 3, H, W)
            targets: 处理后的标签张量 (N, 6), 每行为 [batch_index, class, x, y, w, h]
        """
        images = []
        targets = []

        for i, (img, box) in enumerate(batch):
            images.append(img)
            
            # 为每个框添加 batch 索引
            if box.shape[0] > 0:
                batch_idx = torch.full((box.shape[0], 1), i, dtype=torch.float32)
                # 拼接: [batch_index, class, x, y, w, h]
                target = torch.cat((batch_idx, box), dim=1)
                targets.append(target)
        
        images = torch.stack(images, dim=0)
        
        if len(targets) > 0:
            targets = torch.cat(targets, dim=0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return images, targets
