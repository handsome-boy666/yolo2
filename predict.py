import argparse
import os
import yaml
import torch
import cv2
import numpy as np
import sys
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.getcwd())

from model.yolov2 import YOLOv2
from utils.utils import non_max_suppression, plot_boxes
from data.dataset import pad_to_square, resize

def decode_yolo_output(output, anchors, num_classes, img_size):
    # output: (B, num_anchors * (5 + num_classes), H, W)
    nB = output.size(0)
    nH = output.size(2)
    nW = output.size(3)
    stride = img_size[0] / nH
    
    anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes
    
    prediction = output.view(nB, num_anchors, bbox_attrs, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
    
    # Get outputs
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])
    pred_cls = torch.sigmoid(prediction[..., 5:])
    
    # Calculate offsets for each grid
    grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type_as(x)
    grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type_as(y)
    
    # Anchor dimensions
    anchor_w = torch.tensor([a[0] for a in anchors]).type_as(x).view(1, num_anchors, 1, 1)
    anchor_h = torch.tensor([a[1] for a in anchors]).type_as(x).view(1, num_anchors, 1, 1)
    
    # Final coordinates
    pred_boxes = torch.zeros_like(prediction[..., :4])
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    
    # Scale to image size
    pred_boxes *= stride
    
    output = torch.cat((pred_boxes.view(nB, -1, 4), 
                        conf.view(nB, -1, 1), 
                        pred_cls.view(nB, -1, num_classes)), -1)
    return output

def predict(image_path, weights_path, output_path="output.jpg", conf_thres=0.5, nms_thres=0.4):
    # Load config
    config_path = "e:\\my_workspace\\yolo2_QR\\config.yaml"
    if not os.path.exists(config_path):
        config_path = "config.yaml"
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = YOLOv2(num_classes=config['model']['num_classes'], anchors=config['model']['anchors']).to(device)
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights file not found: {weights_path}")
        return

    model.eval()
    
    # Load image
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img_pil)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Pad and Resize
    img_tensor, pad = pad_to_square(img_tensor, 0)
    img_tensor = resize(img_tensor, config['model']['input_size'])
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = decode_yolo_output(outputs, config['model']['anchors'], config['model']['num_classes'], config['model']['input_size'])
        detections = non_max_suppression(outputs, conf_thres, nms_thres)
        
    if detections and detections[0] is not None:
        print(f"Detected {len(detections[0])} objects.")
        
        # Rescale boxes to original image
        # Note: pad_to_square and resize logic needs to be inverted.
        # But for visualization on the padded/resized image, it's easier.
        # Let's visualize on the resized image for now.
        
        # Convert tensor back to numpy image for plotting
        img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = np.ascontiguousarray(img_np * 255, dtype=np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        detections = detections[0].cpu().numpy()
        img_np = plot_boxes(img_np, detections, config['data']['class_names'])
        
        cv2.imwrite(output_path, img_np)
        print(f"Saved result to {output_path}")
    else:
        print("No detections.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to output image")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="NMS threshold")
    
    args = parser.parse_args()
    
    predict(args.image, args.weights, args.output, args.conf_thres, args.nms_thres)
