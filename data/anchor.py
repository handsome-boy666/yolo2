import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm

def iou(box, clusters):
    """
    计算单个边界框与所有聚类中心的IoU（交并比）
    Args:
        box: (w, h) - 单个边界框的宽度和高度（归一化后）
        clusters: (k, 2) 数组，聚类中心的宽高集合
    Returns:
        (k,) 数组，当前框与每个聚类中心的IoU值
    """
    # 计算交集：交集宽度取框宽和聚类宽的最小值，交集高度取框高和聚类高的最小值
    # 由于所有框都以(0,0)为原点对齐，直接计算即可
    min_w = np.minimum(clusters[:, 0], box[0])
    min_h = np.minimum(clusters[:, 1], box[1])
    
    intersection = min_w * min_h  # 交集面积
    
    box_area = box[0] * box[1]    # 单个框的面积
    cluster_area = clusters[:, 0] * clusters[:, 1]  # 各聚类中心的面积
    
    union = box_area + cluster_area - intersection  # 并集面积
    
    # 避免除以零的情况
    return intersection / (union + 1e-16)

def kmeans(boxes, k, max_iter=300):
    """
    基于IoU作为距离度量的K-means聚类算法（YOLO锚框专用）
    Args:
        boxes: (N, 2) 数组，所有边界框的宽高集合（归一化后）
        k: 聚类数量（锚框数量）
        max_iter: 最大迭代次数
    Returns:
        (k, 2) 数组，最终的聚类中心（锚框尺寸）
    """
    rows = boxes.shape[0]  # 边界框总数
    
    distances = np.empty((rows, k))  # 存储每个框到各聚类中心的距离（1-IoU）
    last_clusters = np.zeros((rows,))  # 上一轮的聚类分配结果
    
    np.random.seed(42)  # 设置随机种子保证结果可复现
    
    # 初始化聚类中心：随机选择k个边界框作为初始中心
    if rows >= k:
        clusters = boxes[np.random.choice(rows, k, replace=False)]
    else:
        # 边界框数量少于k时的降级处理（真实数据集几乎不会出现）
        clusters = boxes[np.random.choice(rows, k, replace=True)]
    
    for _ in range(max_iter):
        # 计算每个框到所有聚类中心的距离（1 - IoU）
        # IoU越大表示距离越近，因此用1-IoU作为距离度量
        for i in range(rows):
            ious = iou(boxes[i], clusters)
            distances[i] = 1 - ious
        
        # 将每个框分配到距离最近的聚类（最小距离=最大IoU）
        nearest_clusters = np.argmin(distances, axis=1)
        
        # 检查是否收敛（聚类分配无变化则停止迭代）
        if (last_clusters == nearest_clusters).all():
            break
            
        last_clusters = nearest_clusters
        
        # 更新聚类中心：取每个聚类内所有框的宽高均值
        for i in range(k):
            mask = (nearest_clusters == i)  # 筛选当前聚类的所有框
            if np.any(mask):
                clusters[i] = np.mean(boxes[mask], axis=0)
                
    return clusters

def anchors_generate(data_dir, num_anchors):
    """
    为YOLO训练生成适配数据集的锚框（Anchor）
    流程：读取标签和图像 → 预处理（模拟正方形填充） → K-means聚类 → 保存锚框到JSON文件
    Args:
        data_dir: 数据集根目录（相对/绝对路径）
        num_anchors: 要生成的锚框数量（如6/9）
    """
    print(f"开始生成锚框：数据集目录={data_dir}，锚框数量={num_anchors}")
    
    # 处理路径：统一转为绝对路径
    if not os.path.isabs(data_dir):
        full_data_dir = os.path.abspath(data_dir)
    else:
        full_data_dir = data_dir

    # 检查锚框文件是否存在
    output_filename = f'yolov2_anchors_k{num_anchors}.json'
    output_file = os.path.join(full_data_dir, output_filename)
    
    if os.path.exists(output_file):
        print(f"检测到锚框文件已存在：{output_file}，直接加载使用")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            if 'cluster_centers' in result:
                anchors = np.array(result['cluster_centers'])
                print("锚框加载成功")
                return anchors
        except Exception as e:
            print(f"加载锚框文件失败：{e}，将重新生成")

    label_dir = os.path.join(full_data_dir, 'train', 'labels')  # 训练集标签目录
    image_dir = os.path.join(full_data_dir, 'train', 'images')  # 训练集图像目录
    
    print(f"正在查找标签文件：{label_dir}")
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))  # 读取所有txt标签文件
    
    if not label_files:
        print(f"错误：在{label_dir}中未找到任何标签文件")
        return

    box_dims = []  # 存储所有边界框的宽高（归一化并适配正方形填充后）
    
    print(f"找到{len(label_files)}个标签文件，开始处理...")
    
    for label_file in tqdm(label_files):  # 进度条显示处理进度
        # 查找对应的图像文件
        basename = os.path.splitext(os.path.basename(label_file))[0]
        
        # 检查常见的图像扩展名
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp_path = os.path.join(image_dir, basename + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break
        
        if image_path is None:
            # print(f"警告：未找到{basename}对应的图像文件")
            continue
        
        # 读取图像获取尺寸（兼容Windows中文路径）
        try:
            img_array = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]  # 图像原始高、宽
        except Exception as e:
            print(f"读取图像{image_path}失败：{e}")
            continue
        
        # 计算正方形填充后的缩放因子（模拟YOLO预处理的正方形填充）
        max_dim = max(h_img, w_img)  # 正方形边长（取最大维度）
        scale_w = w_img / max_dim    # 宽度缩放因子
        scale_h = h_img / max_dim    # 高度缩放因子
        
        # 读取标签文件
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # 标签格式：class x_center y_center width height（均为归一化值）
                        # 仅提取宽和高
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        
                        # 转换为正方形填充后的归一化坐标
                        w_final = w_norm * scale_w
                        h_final = h_norm * scale_h
                        
                        box_dims.append([w_final, h_final])
        except Exception as e:
            print(f"读取标签{label_file}失败：{e}")
            continue

    if not box_dims:
        print("未在标签中找到有效的边界框数据")
        return
        
    box_dims = np.array(box_dims)
    print(f"共收集到{len(box_dims)}个边界框，开始执行K-means聚类...")
    
    # 执行K-means聚类生成锚框
    anchors = kmeans(box_dims, num_anchors)
    
    # 按面积（宽×高）对锚框排序
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    print("生成的锚框尺寸（宽度, 高度）：")
    print(anchors)
    
    # 保存锚框到JSON文件（保存到数据集目录）
    # 仅保留聚类中心字段，移除重复的anchors字段
    result = {
        "cluster_centers": anchors.tolist()
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        print(f"锚框已成功保存到：{output_file}")
    except Exception as e:
        print(f"保存锚框文件失败：{e}")
    
    return anchors

if __name__ == "__main__":
    # 测试用例（运行脚本时可在此配置参数）
    # 示例：调用生成6个锚框
    # anchors_generate(data_dir="./my_dataset", num_anchors=6)
    pass