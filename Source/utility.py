"""
Utility functions for Face Mask Detection
"""

import torch
import torchvision
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pathlib import Path

def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if isinstance(x, torch.Tensor):
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    else:
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def labels_to_targets(labels, batch_idx):
    """Convert labels to target format for YOLO loss calculation"""
    if labels is None or (isinstance(labels, np.ndarray) and len(labels) == 0):
        return torch.zeros((0, 6))
    
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).float()
    
    # Create targets tensor
    targets = torch.zeros((labels.shape[0], 6))
    targets[:, 0] = batch_idx  # image index in batch
    targets[:, 1:] = labels  # class, x, y, w, h
    
    return targets

def process_yolo_output(outputs, conf_thres=0.25, iou_thres=0.45, nc=3):
    """
    Process YOLO model outputs to get final predictions
    
    Args:
        outputs: Raw model outputs
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        nc: Number of classes
    
    Returns:
        List of predictions for each image in batch
    """
    # Extract predictions tensor
    if isinstance(outputs, (list, tuple)):
        predictions = outputs[0]
    else:
        predictions = outputs
    
    batch_size = predictions.shape[0]
    batch_predictions = []
    
    for batch_idx in range(batch_size):
        image_pred = predictions[batch_idx]
        
        # Extract components
        boxes = image_pred[:, :4]  # x, y, w, h
        obj_conf = image_pred[:, 4]  # objectness
        class_probs = image_pred[:, 5:5+nc]  # class probabilities
        
        # Get best class and score
        class_scores, class_preds = class_probs.max(dim=1)
        
        # Combined confidence
        scores = obj_conf * class_scores
        
        # Filter by confidence
        mask = scores > conf_thres
        
        if not mask.any():
            batch_predictions.append(None)
            continue
        
        # Apply mask
        boxes = boxes[mask]
        scores = scores[mask]
        class_preds = class_preds[mask]
        
        # Convert to xyxy
        boxes = xywh2xyxy(boxes)
        
        # Apply NMS
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        
        if len(keep) > 0:
            final_pred = torch.zeros((len(keep), 6), device=predictions.device)
            final_pred[:, :4] = boxes[keep]
            final_pred[:, 4] = scores[keep]
            final_pred[:, 5] = class_preds[keep].float()
            batch_predictions.append(final_pred)
        else:
            batch_predictions.append(None)
    
    return batch_predictions

def calculate_iou_batch(boxes1, boxes2):
    """Calculate IoU between two sets of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter_area
    
    return inter_area / (union + 1e-6)

def compute_ap(recall, precision):
    """Compute AP using 11-point interpolation"""
    mrec = torch.cat([torch.tensor([0.]), recall, torch.tensor([1.])])
    mpre = torch.cat([torch.tensor([0.]), precision, torch.tensor([0.])])
    
    # Compute precision envelope
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    i = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = ((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum()
    
    return ap

def validate_model_simple(model, val_dataloader, device, conf_thres=0.25, iou_thres=0.45, 
                         iou_threshold_metric=0.5, img_size=640):
    """
    Validate YOLO model
    
    Returns:
        dict with mAP, precision, recall and per-class metrics
    """
    model.eval()
    
    # Get number of classes
    nc = model.nc if hasattr(model, 'nc') else 3
    
    # Storage for stats
    stats = []
    
    with torch.no_grad():
        for inputs, labels_list in tqdm(val_dataloader, desc="Validating"):
            inputs = inputs.to(device)
            
            # Get predictions
            outputs = model(inputs)
            predictions = process_yolo_output(outputs, conf_thres, iou_thres, nc)
            
            # Process each image
            for pred, labels in zip(predictions, labels_list):
                # Convert labels to tensor
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).to(device)
                elif isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                else:
                    labels = torch.zeros((0, 5)).to(device)
                
                # Skip if no labels
                if len(labels) == 0:
                    if pred is not None:
                        stats.append((
                            torch.zeros(0, dtype=torch.bool).cpu(),
                            torch.zeros(0).cpu(),
                            torch.zeros(0).cpu(),
                            torch.zeros(0).cpu()
                        ))
                    continue
                
                # Extract ground truth
                gt_classes = labels[:, 0]
                gt_boxes = labels[:, 1:5]
                
                # Convert normalized coordinates if needed
                if gt_boxes.max() <= 1.0:
                    gt_boxes = gt_boxes * img_size
                
                # Convert to xyxy
                gt_boxes = xywh2xyxy(gt_boxes)
                
                if pred is None:
                    stats.append((
                        torch.zeros(0, dtype=torch.bool).cpu(),
                        torch.zeros(0).cpu(),
                        torch.zeros(0).cpu(),
                        gt_classes.cpu()
                    ))
                    continue
                
                # Move predictions to device
                pred = pred.to(device)
                
                # Extract prediction info
                pred_boxes = pred[:, :4]
                pred_scores = pred[:, 4]
                pred_classes = pred[:, 5]
                
                # Initialize correct array
                correct = torch.zeros(len(pred), dtype=torch.bool, device=device)
                
                if len(gt_boxes) > 0:
                    # Calculate IoU
                    iou = calculate_iou_batch(pred_boxes, gt_boxes)
                    
                    # Match predictions to ground truth
                    matched = set()
                    for gt_idx, gt_class in enumerate(gt_classes):
                        class_mask = pred_classes == gt_class
                        if not class_mask.any():
                            continue
                        
                        gt_ious = iou[class_mask, gt_idx]
                        
                        if gt_ious.numel() > 0:
                            best_iou, best_idx_in_class = gt_ious.max(0)
                            
                            if best_iou >= iou_threshold_metric:
                                actual_idx = class_mask.nonzero(as_tuple=True)[0][best_idx_in_class]
                                
                                if actual_idx.item() not in matched:
                                    correct[actual_idx] = True
                                    matched.add(actual_idx.item())
                
                # Store results
                stats.append((correct.cpu(), pred_scores.cpu(), pred_classes.cpu(), gt_classes.cpu()))
    
    # Compute metrics
    if len(stats) == 0:
        return {
            'mAP': 0,
            'precision': 0,
            'recall': 0,
            'ap_per_class': [],
            'p_per_class': [],
            'r_per_class': []
        }
    
    # Concatenate stats
    stats = [torch.cat(x, 0) for x in zip(*stats)]
    
    # Calculate per-class metrics
    ap_per_class = []
    p_per_class = []
    r_per_class = []
    
    for c in range(nc):
        pred_mask = stats[2] == c
        gt_mask = stats[3] == c
        
        n_pred = pred_mask.sum()
        n_gt = gt_mask.sum()
        
        if n_pred == 0 and n_gt == 0:
            ap_per_class.append(1.0)
            p_per_class.append(1.0)
            r_per_class.append(1.0)
            continue
        elif n_pred == 0:
            ap_per_class.append(0.0)
            p_per_class.append(0.0)
            r_per_class.append(0.0)
            continue
        
        # Sort by confidence
        sorted_indices = torch.argsort(stats[1][pred_mask], descending=True)
        correct_sorted = stats[0][pred_mask][sorted_indices]
        
        # Calculate cumulative TP and FP
        tp_cumsum = correct_sorted.cumsum(0).float()
        fp_cumsum = (~correct_sorted)