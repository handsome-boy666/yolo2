import os
import yaml
import torch
from datetime import datetime
from typing import Dict, Any

from data.anchor import anchors_generate
from data.dataset import MyDataset
from model.yolov2 import YOLOv2
from utils.trainer import Trainer
from utils.logger import setup_logger, TrainingRecorder


def read_train_config(path: str) -> Dict[str, Any]:
    """
    读取并解析训练配置。

    Args:
        path: 配置文件路径

    Returns:
        包含配置信息的字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件未找到: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg.get('train', {})
    return {
        'data_dir': tcfg.get('data_dir'),
        'num_anchors': int(tcfg.get('num_anchors', 5)),
        'continue_model': tcfg.get('continue_model'),
        'batch_size': int(tcfg.get('batch_size')),
        'num_workers': int(tcfg.get('num_workers')),
        'base_img_size': int(tcfg.get('base_img_size', 416)),
        'epochs': int(tcfg.get('epochs')),
        'save_interval': int(tcfg.get('save_interval', 1)),
        'device': str(tcfg.get('device', 'cuda:0')),
        'num_classes': int(tcfg.get('num_classes', 20)),  # 默认为 VOC 类别数
        # 训练超参数
        'learning_rate': float(tcfg.get('learning_rate', 0.001)),
        'momentum': float(tcfg.get('momentum', 0.9)),
        'weight_decay': float(tcfg.get('weight_decay', 0.0005)),
        # 阈值配置
        'conf_thresh': float(tcfg.get('conf_thresh', 0.25)),
        'nms_thresh': float(tcfg.get('nms_thresh', 0.45)),
        'iou_thresh': float(tcfg.get('iou_thresh', 0.5)),
        # 输出
        'checkpoint_dir': tcfg.get('checkpoint_dir', 'checkpoints'),
        'log_dir': tcfg.get('log_dir', 'logs'),
    }


def main():
    # 1. 读取配置
    try:
        config = read_train_config('config.yaml')
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 2. 初始化环境和日志
    device = torch.device(config['device'])
    
    # 生成本次运行的唯一 ID (时间戳)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置目录
    run_log_dir = os.path.join(config['log_dir'], run_id)
    # 将 checkpoint 保存到 log 文件夹中
    run_ckpt_dir = os.path.join(run_log_dir, 'checkpoints')
    os.makedirs(run_ckpt_dir, exist_ok=True)
    
    # 初始化 Logger 和 Recorder
    logger = setup_logger(run_log_dir)
    recorder = TrainingRecorder(run_log_dir)
    
    logger.info(f"本次训练 Log 目录: {run_log_dir}", f"使用设备：{device}")
    
    # 3. 数据准备
    logger.info("正在初始化锚框...")
    anchors = anchors_generate(data_dir=config['data_dir'], num_anchors=config['num_anchors'])
    
    logger.info("正在初始化数据集...")
    dataset = MyDataset(data_dir=config['data_dir'], if_train=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=dataset.collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 4. 模型与优化器
    logger.info("正在初始化模型...")
    model = YOLOv2(num_classes=config['num_classes'], anchors=anchors)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'], 
        momentum=config['momentum'], 
        weight_decay=config['weight_decay']
    )
    
    # 5. 初始化训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        logger=logger,
        recorder=recorder,
        config=config,
        save_dir=run_ckpt_dir,
        anchors=anchors
    )
    
    # 6. 开始训练
    trainer.train(
        epochs=config['epochs'],
        save_interval=config['save_interval']
    )


if __name__ == '__main__':
    main()
