import cv2
import numpy as np
from typing import Optional, List, Tuple

def plot_boxes(img: np.ndarray, boxes: np.ndarray, class_names: Optional[List[str]] = None, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        cls_id = int(box[6])
        cls_conf = box[5]
        if color is not None:
            c = color
        else:
            h = (cls_id * 0.618033988749895) % 1
            h_deg = int(h * 180)
            mat = np.uint8([[[h_deg, 255, 255]]])
            bgr = cv2.cvtColor(mat, cv2.COLOR_HSV2BGR)[0][0]
            c = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        if class_names:
            label = f"{class_names[cls_id]} {cls_conf:.2f}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), c, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
