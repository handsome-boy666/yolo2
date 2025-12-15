import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.append(os.getcwd())

from model.yolov2 import YOLOv2
from model.loss import YOLOLoss
from data.dataset import ListDataset
from utils.logger import setup_logger

def train():
    # Load config
    config_path = "e:\\my_workspace\\yolo2_QR\\config.yaml"
    if not os.path.exists(config_path):
        config_path = "config.yaml"
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger(config['output']['log_dir'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model
    logger.info("Initializing model...")
    model = YOLOv2(num_classes=config['model']['num_classes'], anchors=config['model']['anchors']).to(device)
    
    # Loss
    criterion = YOLOLoss(anchors=config['model']['anchors'], 
                         num_classes=config['model']['num_classes'], 
                         img_size=config['model']['input_size'])
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['train']['learning_rate'], 
                          momentum=config['train']['momentum'], 
                          weight_decay=config['train']['weight_decay'])
                          
    # DataLoader
    logger.info("Initializing data loader...")
    # Check if train path exists
    if not os.path.exists(config['data']['train_path']):
        logger.warning(f"Train path {config['data']['train_path']} does not exist. Creating it.")
        os.makedirs(config['data']['train_path'], exist_ok=True)
        
    dataset = ListDataset(config['data']['train_path'], img_size=config['model']['input_size'][0])
    
    if len(dataset) == 0:
        logger.warning("No images found in training path. Skipping training loop.")
        return

    dataloader = DataLoader(dataset, 
                            batch_size=config['train']['batch_size'], 
                            shuffle=True, 
                            num_workers=config['train']['num_workers'], 
                            collate_fn=dataset.collate_fn)
                            
    logger.info(f"Starting training for {config['train']['epochs']} epochs")
    
    for epoch in range(config['train']['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            if imgs is None: continue
            
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss, loss_components = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_i % 10 == 0:
                logger.info(f"[Epoch {epoch+1}/{config['train']['epochs']}] [Batch {batch_i}/{len(dataloader)}] [Loss: {loss.item():.4f}]")
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['train']['save_interval'] == 0:
            save_path = os.path.join(config['output']['checkpoint_dir'], f"yolov2_epoch_{epoch+1}.pth")
            model.save_weights(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Save final model
    final_path = os.path.join(config['output']['checkpoint_dir'], "yolov2_final.pth")
    model.save_weights(final_path)
    logger.info(f"Saved final model to {final_path}")

if __name__ == "__main__":
    train()
