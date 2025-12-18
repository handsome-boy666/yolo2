import os
import yaml
import torch
from datetime import datetime
from typing import Dict, Any, Tuple

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
        'num_anchors': int(tcfg.get('num_anchors')),
        'continue_model': tcfg.get('continue_model'),
        'batch_size': int(tcfg.get('batch_size')),
        'num_workers': int(tcfg.get('num_workers')),
        'base_img_size': int(tcfg.get('base_img_size')),
        'fix_img_size': bool(tcfg.get('fix_img_size')),
        'epochs': int(tcfg.get('epochs')),
        'save_interval': int(tcfg.get('save_interval')),
        'device': str(tcfg.get('device', 'cuda:0')),
        'num_classes': int(tcfg.get('num_classes')), 
        # 训练超参数
        'learning_rate': float(tcfg.get('learning_rate')),
        'momentum': float(tcfg.get('momentum')),
        'weight_decay': float(tcfg.get('weight_decay')),
        'lambda_coord': float(tcfg.get('lambda_coord')),
        'lambda_noobj': float(tcfg.get('lambda_noobj')),
        # 阈值配置
        'conf_thresh': float(tcfg.get('conf_thresh')),
        'nms_thresh': float(tcfg.get('nms_thresh')),
        'iou_thresh': float(tcfg.get('iou_thresh')),
        # 输出
        'checkpoint_dir': tcfg.get('checkpoint_dir', 'checkpoints'),
        'log_dir': tcfg.get('log_dir', 'logs'),
    }


def setup_training_env(config: Dict[str, Any], device: torch.device) -> Tuple[str, Any, Any]:
    """
    初始化训练环境：创建目录、初始化日志和记录器
    """
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
    
    logger.info(f"训练配置：Device: {device}  Data directory: {config['data_dir']}  Number of anchors: {config['num_anchors']}  Number of classes: {config['num_classes']}")
    logger.info(f"超参数：Learning rate: {config['learning_rate']}  Momentum: {config['momentum']}  Weight decay: {config['weight_decay']}")
    logger.info(f"阈值设置：Confidence threshold: {config['conf_thresh']}  NMS threshold: {config['nms_thresh']}  IoU threshold: {config['iou_thresh']}")
    logger.info(f"数据加载：Batch size: {config['batch_size']}  Number of workers: {config['num_workers']}")
    
    return run_ckpt_dir, logger, recorder


def main():
    # 1. 读取配置
    config = read_train_config('config.yaml')
    
    # 2. 初始化环境和日志
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
    run_ckpt_dir, logger, recorder = setup_training_env(config, device)
    
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
        collate_fn=dataset.collate_fn,  # 自定义的 collate_fn 用于处理不同大小的输入
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
        ckpt_dir=run_ckpt_dir,
        anchors=anchors,
        num_classes=config['num_classes']
    )
    
    # 6. 开始训练
    trainer.train(
        epochs=config['epochs'],
        save_interval=config['save_interval']
    )


if __name__ == '__main__':
    main()
