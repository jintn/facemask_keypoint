"""
Validation utilities for Face Mask Detection
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


def process_yolo_output_v5(outputs, conf_thres=0.25, iou_thres=0.45, nc=3, img_size=640):
    """
    Process YOLOv5 model outputs properly
    
    Args:
        outputs: Raw model outputs from YOLOv5
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        nc: Number of classes
        img_size: Input image size
    
    Returns:
        List of predictions for each image in batch
    """
    # YOLOv5 outputs are already processed through Detect layer
    # Format: [batch, num_detections, 5 + num_classes]
    # where 5 = [x, y, w, h, objectness]
    
    if isinstance(outputs, (list, tuple)):
        predictions = outputs[0]
    else:
        predictions = outputs
    
    # Check if we need to apply sigmoid (for raw outputs)
    if predictions.max() > 1.0:
        # Apply sigmoid to objectness and class scores
        predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])
    
    batch_size = predictions.shape[0]
    batch_predictions = []
    
    for batch_idx in range(batch_size):
        image_pred = predictions[batch_idx]
        
        # Filter by objectness score first
        obj_mask = image_pred[:, 4] > conf_thres
        image_pred = image_pred[obj_mask]
        
        if len(image_pred) == 0:
            batch_predictions.append(None)
            continue
        
        # Get class scores and predictions
        class_scores, class_preds = image_pred[:, 5:].max(1)
        
        # Combined confidence
        scores = image_pred[:, 4] * class_scores
        
        # Filter by final confidence
        conf_mask = scores > conf_thres
        
        if not conf_mask.any():
            batch_predictions.append(None)
            continue
        
        # Apply confidence mask
        boxes = image_pred[conf_mask, :4]
        scores = scores[conf_mask]
        class_preds = class_preds[conf_mask]
        
        # Scale boxes if they appear to be in grid coordinates
        if boxes.max() < 2.0:  # Likely normalized
            boxes *= img_size
        
        # Convert to xyxy
        boxes = xywh2xyxy(boxes)
        
        # Clamp to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_size)
        
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


# Fallback to original if needed
process_yolo_output = process_yolo_output_v5


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
                         iou_threshold_metric=0.5, img_size=640, use_yolov5_nms=True):
    """
    Validate YOLO model with proper YOLOv5 handling
    
    Returns:
        dict with mAP, precision, recall and per-class metrics
    """
    model.eval()
    
    # Get number of classes
    nc = model.nc if hasattr(model, 'nc') else 3
    
    # Storage for stats
    stats = []
    
    # If model has native NMS, use it
    if use_yolov5_nms and hasattr(model, 'model') and hasattr(model.model[-1], 'inplace'):
        # Try to use YOLOv5's built-in NMS
        try:
            from yolov5.utils.general import non_max_suppression
            use_native_nms = True
            print("Using YOLOv5 native NMS")
        except:
            use_native_nms = False
            print("Using custom NMS")
    else:
        use_native_nms = False
    
    with torch.no_grad():
        for inputs, labels_list in tqdm(val_dataloader, desc="Validating"):
            inputs = inputs.to(device)
            
            # Get predictions
            outputs = model(inputs)
            
            if use_native_nms:
                # Use YOLOv5's native NMS
                predictions = non_max_suppression(
                    outputs,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    classes=None,
                    agnostic=False,
                    max_det=300
                )
            else:
                # Use our custom processing
                predictions = process_yolo_output_v5(
                    outputs, conf_thres, iou_thres, nc, img_size
                )
            
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
                
                # Convert normalized coordinates to pixels
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
        fp_cumsum = (~correct_sorted).cumsum(0).float()
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (n_gt + 1e-6)
        
        # Calculate AP
        ap = compute_ap(recall, precision)
        
        ap_per_class.append(ap.item())
        p_per_class.append(precision[-1].item() if len(precision) > 0 else 0)
        r_per_class.append(recall[-1].item() if len(recall) > 0 else 0)
    
    # Calculate mean metrics
    mAP = np.mean(ap_per_class)
    precision = np.mean(p_per_class)
    recall = np.mean(r_per_class)
    
    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'ap_per_class': ap_per_class,
        'p_per_class': p_per_class,
        'r_per_class': r_per_class
    }


def validate_epochs_simple(ckpt_dir, model, val_dataloader, device, conf_thres=0.25, iou_thres=0.45):
    """
    Validate all epochs in checkpoint directory
    """
    ckpt_dir = Path(ckpt_dir)
    checkpoint_files = sorted(ckpt_dir.glob("epoch*.pt"))
    
    results = []
    
    for ckpt_path in checkpoint_files:
        # Extract epoch number
        epoch = int(ckpt_path.stem.replace('epoch', ''))
        
        print(f"\nValidating Epoch {epoch}...")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Load model state
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        # Validate
        metrics = validate_model_simple(model, val_dataloader, device, conf_thres, iou_thres)
        metrics['epoch'] = epoch
        results.append(metrics)
        
        print(f"Epoch {epoch}: mAP={metrics['mAP']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    # Save results
    if results:
        save_results_simple(results, ckpt_dir)
    
    return results


def save_results_simple(results, save_dir):
    """Save validation results and create plots"""
    save_dir = Path(save_dir)
    
    # Save JSON
    with open(save_dir / 'validation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    epochs = [r['epoch'] for r in results]
    maps = [r['mAP'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    plt.figure(figsize=(15, 5))
    
    # mAP plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, maps, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP@0.5', fontsize=12)
    plt.title('Mean Average Precision', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Precision plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, precisions, 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Recall plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, recalls, 'r-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'validation_metrics.png', dpi=150)
    plt.close()
    
    # Find best epoch
    best_epoch = max(results, key=lambda x: x['mAP'])
    print(f"\nBest epoch: {best_epoch['epoch']} with mAP={best_epoch['mAP']:.4f}")
    
    # Save best epoch info
    with open(save_dir / 'best_epoch.txt', 'w') as f:
        f.write(f"Best epoch: {best_epoch['epoch']}\n")
        f.write(f"mAP: {best_epoch['mAP']:.4f}\n")
        f.write(f"Precision: {best_epoch['precision']:.4f}\n")
        f.write(f"Recall: {best_epoch['recall']:.4f}\n")