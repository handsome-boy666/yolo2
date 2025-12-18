"""
YOLOv2 预测脚本
适配本项目配置与数据处理流程
"""
import os
import json
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from model.yolov2 import YOLOv2
from utils.utils import decode_yolo_output, non_max_suppression
from utils.visualize import plot_boxes

def load_config(path: str) -> dict:
    """读取并解析配置"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("predict", {})

def load_anchors(data_dir, num_anchors):
    """根据约定加载 anchors"""
    anchor_path = os.path.join(data_dir, f'yolov2_anchors_k{num_anchors}.json')
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")
    with open(anchor_path, 'r') as f:
        data = json.load(f)
    return data['cluster_centers']

def load_classes(data_dir):
    """根据约定加载 classes"""
    class_path = os.path.join(data_dir, 'classes.txt')
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Class file not found: {class_path}")
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def load_model(ckpt_path: str, num_classes: int, anchors: list, device: torch.device):
    """构建模型并加载权重"""
    print(f"Loading model from {ckpt_path}...")
    model = YOLOv2(num_classes=num_classes, anchors=anchors)
    
    if not os.path.exists(ckpt_path):
         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 处理不同的模型保存格式
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        print("Detected checkpoint format, loading 'model_state'...")
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Loading state_dict directly...")
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def letterbox_image(img, new_shape):
    """
    调整图像大小并进行填充，保持纵横比（Letterbox）
    返回: new_img, scale, dx, dy
    """
    w, h = img.size
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(w * scale), int(h * scale)

    # 调整图像大小
    img_resized = img.resize((nw, nh), Image.BICUBIC)

    # 创建新图像并填充灰色
    new_img = Image.new('RGB', (new_shape, new_shape), (128, 128, 128))
    
    # 计算偏移量，使其居中
    dx = (new_shape - nw) // 2
    dy = (new_shape - nh) // 2
    new_img.paste(img_resized, (dx, dy))
    
    return new_img, scale, dx, dy

def process_single_image(model, image_path, config, anchors, classes, device, output_path=None):
    """处理单张图片"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img_size = config["img_size"]
    
    # 1. 读取与预处理
    img = Image.open(image_path).convert('RGB')
    w_orig, h_orig = img.size
    
    img_input, scale, dx, dy = letterbox_image(img, img_size)
    
    transform = transforms.ToTensor()
    img_tensor = transform(img_input).unsqueeze(0).to(device)

    # 2. 推理
    with torch.no_grad():
        output = model(img_tensor)
        decoded_output = decode_yolo_output(output, anchors, config["num_classes"], (img_size, img_size))
        detections = non_max_suppression(decoded_output, config["conf_thresh"], config["nms_thresh"])[0]

    # 3. 后处理与可视化
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if detections is not None:
        boxes = detections.cpu().numpy()
        
        # 还原坐标到原图
        # 去除 padding 偏移
        boxes[:, 0] -= dx
        boxes[:, 1] -= dy
        boxes[:, 2] -= dx
        boxes[:, 3] -= dy
        
        # 去除缩放
        boxes[:, :4] /= scale
        
        # 裁剪超出范围的框
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)
        
        # 绘制
        plot_boxes(img_cv, boxes, classes)
        
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_cv)
        print(f"Saved prediction to {output_path}")

def run_single(config, model, anchors, classes, device):
    """单图预测模式"""
    image_path = config["image_path"]
    out_dir = config["out_dir"]
    filename = os.path.basename(image_path)
    output_path = os.path.join(out_dir, f"pred_{filename}")
    
    print(f"Processing single image: {image_path}")
    process_single_image(model, image_path, config, anchors, classes, device, output_path)

from torch.utils.data import DataLoader
from data.dataset import MyDataset
from utils.utils import DetectionMetrics

def save_txt_results(detections, output_path, classes):
    """保存检测结果到 txt 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # detections: [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            score = det[4] * det[5]
            cls_id = int(det[6])
            # 格式: class_name score x1 y1 x2 y2
            f.write(f"{classes[cls_id]} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

def run_test_batch(config, model, anchors, classes, device):
    """批量测试模式：评估指标 + 保存结果 + 可视化"""
    print("Running batch test evaluation...")
    
    # 1. 准备数据
    data_dir = config["data_dir"]
    img_size = config["img_size"]
    batch_size = config["test_batch_size"]
    
    try:
        # 使用 MyDataset 加载测试集 (包含标签)
        dataset = MyDataset(data_dir, if_train=False, base_size=img_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=config["num_workers"],
            collate_fn=MyDataset.collate_fn,
            pin_memory=True
        )
    except FileNotFoundError as e:
        print(f"Dataset load failed: {e}")
        print("Falling back to image-only processing...")
        # 如果没有标签，回退到仅预测模式 (之前的实现)
        run_inference_only(config, model, anchors, classes, device)
        return

    # 2. 准备指标计算器
    metrics = DetectionMetrics(
        iou_thresh=config["iou_thresh"], 
        conf_thresh=config["conf_thresh"],
        nms_thresh=config["nms_thresh"]
    )
    
    # 3. 准备输出目录
    save_dir = config["save_dir"]
    vis_dir = os.path.join(save_dir, "test_vis")
    txt_dir = os.path.join(save_dir, "test_txt")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    print(f"Results will be saved to {save_dir}")
    
    # 4. 批量推理与评估
    model.eval()
    
    # 记录全局索引以便找到对应的文件名
    global_idx = 0
    
    with torch.no_grad():
        for batch_i, (imgs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 推理
            output = model(imgs)
            
            # 更新指标统计 (DetectionMetrics 内部处理解码、NMS和匹配)
            metrics.update(output, targets, anchors, config["num_classes"], (img_size, img_size))
            
            # 为了保存结果和可视化，我们需要自己再解码一次 (或者修改 DetectionMetrics 返回结果，但修改库代码风险较大)
            # 这里选择重新解码处理，虽然有计算冗余，但逻辑清晰且安全
            decoded = decode_yolo_output(output, anchors, config["num_classes"], (img_size, img_size))
            batch_detections = non_max_suppression(decoded, config["conf_thresh"], config["nms_thresh"])
            
            # 处理当前 batch 的每张图片
            for i, detections in enumerate(batch_detections):
                # 获取对应的文件名和原始路径
                if global_idx < len(dataset):
                    img_path, _ = dataset.samples[global_idx]
                    filename = os.path.basename(img_path)
                    basename = os.path.splitext(filename)[0]
                    
                    # 读取原图尺寸 (为了还原坐标)
                    # 注意: DataLoader 中的 img 已经是 resize 过的 tensor，我们需要重新读取原图或者通过 dataset 获取原图信息
                    # 这里为了简单准确，直接读取原图
                    # 优化: 可以在 Dataset 中返回 scale 信息，但需要修改 Dataset
                    # 目前方案: 再次读取原图，虽然慢但稳健
                    
                    orig_img = Image.open(img_path).convert('RGB')
                    w_orig, h_orig = orig_img.size
                    
                    # 准备可视化画布
                    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                    
                    final_boxes = []
                    
                    if detections is not None:
                        # detections: [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred] (在 img_size 尺度下，且包含 padding)
                        boxes = detections.cpu().numpy()
                        
                        # 计算 padding 和 scale
                        # 逻辑同 letterbox_image
                        scale = min(img_size / w_orig, img_size / h_orig)
                        nw, nh = int(w_orig * scale), int(h_orig * scale)
                        dx = (img_size - nw) // 2
                        dy = (img_size - nh) // 2
                        
                        # 还原坐标
                        boxes[:, 0] -= dx
                        boxes[:, 1] -= dy
                        boxes[:, 2] -= dx
                        boxes[:, 3] -= dy
                        
                        boxes[:, :4] /= scale
                        
                        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
                        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
                        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
                        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)
                        
                        # 保存处理后的框用于 txt 和可视化
                        # boxes columns: x1, y1, x2, y2, obj_conf, cls_conf, cls_pred
                        final_boxes = boxes
                        
                        # 可视化
                        plot_boxes(img_cv, boxes, classes)
                    
                    # 保存可视化图片
                    cv2.imwrite(os.path.join(vis_dir, filename), img_cv)
                    
                    # 保存 txt 结果
                    # 格式: class_name score x1 y1 x2 y2 (适配一些通用评估工具)
                    # 或者: class_id score x1 y1 x2 y2
                    # 这里使用 class_name 以便人类阅读，或者根据用户需求调整
                    save_txt_results(final_boxes, os.path.join(txt_dir, f"{basename}.txt"), classes)
                
                global_idx += 1

    # 5. 计算并打印最终指标
    precision, recall, map_score, miou = metrics.compute()
    print("\n" + "="*40)
    print(f"Test Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP:       {map_score:.4f}")
    print(f"mIoU:      {miou:.4f}")
    print("="*40 + "\n")
    
    # 保存指标到文件
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "precision": precision,
            "recall": recall,
            "mAP": map_score,
            "mIoU": miou
        }, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

def run_inference_only(config, model, anchors, classes, device):
    """(回退函数) 仅推理模式，用于没有标签的情况"""
    data_dir = config["data_dir"]
    test_img_dir = os.path.join(data_dir, "test", "images")
    save_dir = os.path.join(config["save_dir"], "test_inference_only")
    
    if not os.path.exists(test_img_dir):
        print(f"Test image directory not found: {test_img_dir}")
        return

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(test_img_dir) if os.path.splitext(f)[1].lower() in img_exts]
    image_files.sort()
    
    print(f"Found {len(image_files)} images in {test_img_dir}")
    print(f"Saving results to {save_dir}")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(test_img_dir, img_file)
        out_path = os.path.join(save_dir, f"pred_{img_file}")
        process_single_image(model, img_path, config, anchors, classes, device, out_path)


def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("Config file not found.")
        return
        
    config = load_config(config_path)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载元数据
    anchors = load_anchors(config["data_dir"], config["num_anchors"])
    classes = load_classes(config["data_dir"])
    
    # 加载模型
    model = load_model(config["ckpt_path"], config["num_classes"], anchors, device)
    
    if config["run_test"]:
        run_test_batch(config, model, anchors, classes, device)
    else:
        run_single(config, model, anchors, classes, device)

if __name__ == "__main__":
    main()
