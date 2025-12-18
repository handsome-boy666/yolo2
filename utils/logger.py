import os
import logging
import json
from datetime import datetime
from typing import Optional

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    """
    配置并返回一个日志记录器。
    
    Args:
        log_dir: 日志文件存储目录
        name: logger 名称
        
    Returns:
        配置好的 logger 实例
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加 handler
    if not logger.handlers: # 防止重复添加 handler
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log"), 
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

class TrainingRecorder:
    """
    负责将训练过程中的指标记录到 JSONL 文件中。
    """
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.batches_path = os.path.join(run_dir, "batches.jsonl")

    def _append_jsonl(self, path: str, data: dict) -> None:
        """辅助方法：追加写入 JSONL"""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def record_epoch(self, epoch: int, loss: float, precision: float, recall: float, map_score: float, miou: float, lr: Optional[float] = None) -> None:
        """记录 Epoch 级别的指标"""
        data = {
            "epoch": int(epoch),
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "map": float(map_score),
            "miou": float(miou),
            "lr": (None if lr is None else float(lr)),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._append_jsonl(self.metrics_path, data)

    def record_batch(self, epoch: int, batch_idx: int, loss: float) -> None:
        """记录 Batch 级别的指标"""
        data = {
            "epoch": int(epoch),
            "batch": int(batch_idx),
            "loss": float(loss),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._append_jsonl(self.batches_path, data)

def get_lr(optimizer) -> float:
    """从优化器中获取当前学习率"""
    try:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    except Exception:
        return 0.0
