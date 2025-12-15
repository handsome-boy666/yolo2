import torch
import torch.nn as nn
import math

def bbox_iou_wh(anchor, gt_wh):
    # anchor: [w, h]
    # gt_wh: [N, 2]
    # Assume centered at 0,0
    
    # Ensure anchor is tensor
    if not isinstance(anchor, torch.Tensor):
        anchor = torch.tensor(anchor).type_as(gt_wh)

    w1, h1 = anchor[0], anchor[1]
    w2, h2 = gt_wh[:, 0], gt_wh[:, 1]
    
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    
    return inter_area / (union_area + 1e-16)

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.bbox_attrs = 5 + num_classes

    def forward(self, input, targets):
        # input: (B, num_anchors * (5 + num_classes), H, W)
        # targets: (N, 6) -> (batch_idx, class, x, y, w, h)
        
        nB = input.size(0)
        nH = input.size(2)
        nW = input.size(3)
        stride = self.img_size[0] / nH
        
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        
        prediction = input.view(nB, self.num_anchors, self.bbox_attrs, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls prob.
        
        # Build targets
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
            targets, scaled_anchors, nB, nH, nW
        )
        
        # Move to device
        mask = mask.type_as(input)
        noobj_mask = noobj_mask.type_as(input)
        tx = tx.type_as(input)
        ty = ty.type_as(input)
        tw = tw.type_as(input)
        th = th.type_as(input)
        tconf = tconf.type_as(input)
        tcls = tcls.type_as(input)

        # Loss
        loss_x = nn.MSELoss(reduction='sum')(x * mask, tx * mask)
        loss_y = nn.MSELoss(reduction='sum')(y * mask, ty * mask)
        loss_w = nn.MSELoss(reduction='sum')(w * mask, tw * mask)
        loss_h = nn.MSELoss(reduction='sum')(h * mask, th * mask)
        
        loss_conf = nn.MSELoss(reduction='sum')(conf * mask, tconf * mask) + \
                    0.5 * nn.MSELoss(reduction='sum')(conf * noobj_mask, tconf * noobj_mask)
                    
        loss_cls = nn.MSELoss(reduction='sum')(pred_cls * mask, tcls * mask) 
        
        loss = (loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls) / nB
        
        return loss, {
            "x": loss_x.item() / nB,
            "y": loss_y.item() / nB,
            "w": loss_w.item() / nB,
            "h": loss_h.item() / nB,
            "conf": loss_conf.item() / nB,
            "cls": loss_cls.item() / nB,
            "total": loss.item()
        }

    def build_targets(self, targets, anchors, nB, nH, nW):
        n_anchors = len(anchors)
        n_classes = self.num_classes
        
        mask = torch.zeros(nB, n_anchors, nH, nW)
        noobj_mask = torch.ones(nB, n_anchors, nH, nW)
        tx = torch.zeros(nB, n_anchors, nH, nW)
        ty = torch.zeros(nB, n_anchors, nH, nW)
        tw = torch.zeros(nB, n_anchors, nH, nW)
        th = torch.zeros(nB, n_anchors, nH, nW)
        tconf = torch.zeros(nB, n_anchors, nH, nW)
        tcls = torch.zeros(nB, n_anchors, nH, nW, n_classes)

        for b in range(nB):
            t = targets[targets[:, 0] == b]
            if len(t) == 0:
                continue
            
            # Convert to position relative to box
            gxy = t[:, 2:4] * torch.tensor([nW, nH]).type_as(t)
            gwh = t[:, 4:6] * torch.tensor([nW, nH]).type_as(t)
            
            # Get anchors with best iou
            # anchors list of tuples -> tensor
            # Note: anchors here are scaled anchors (relative to grid size)
            
            # Check if we have any anchors
            if len(anchors) == 0:
                continue
                
            ious = torch.stack([bbox_iou_wh(anchor, gwh) for anchor in anchors])
            best_ious, best_n = ious.max(0)
            
            # Separate target variables
            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gi, gj = gxy.long().t()
            
            # Clamp indices
            gi = torch.clamp(gi, 0, nW - 1)
            gj = torch.clamp(gj, 0, nH - 1)
            
            # Set masks
            mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0
            
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            
            # Width and height
            anchor_w = torch.tensor([anchors[n][0] for n in best_n]).type_as(t)
            anchor_h = torch.tensor([anchors[n][1] for n in best_n]).type_as(t)
            
            tw[b, best_n, gj, gi] = torch.log(gw / anchor_w + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh / anchor_h + 1e-16)
            
            # Object
            tconf[b, best_n, gj, gi] = 1
            
            # One-hot class
            # t[:, 1] is class index
            for i, idx in enumerate(best_n):
               cls_idx = int(t[i, 1])
               if cls_idx < n_classes:
                   tcls[b, idx, gj[i], gi[i], cls_idx] = 1
            
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
