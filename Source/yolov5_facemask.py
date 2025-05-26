#!/usr/bin/env python3
"""
Training script for Face Mask Detection using YOLOv5 with heatmap channel
"""

import argparse
import yaml
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

# Get the absolute path to the current file's directory
CURRENT_DIR = Path(__file__).parent.absolute()

# Add YOLOv5 directory to Python path
YOLOV5_DIR = CURRENT_DIR / 'yolov5'
if YOLOV5_DIR.exists():
    sys.path.insert(0, str(YOLOV5_DIR))
else:
    print(f"ERROR: YOLOv5 directory not found at {YOLOV5_DIR}")
    sys.exit(1)

# Now import YOLOv5 modules
try:
    from utils.loss import ComputeLoss
    from utils.general import non_max_suppression, xywh2xyxy, box_iou
    from models.yolo import Model
except ImportError as e:
    print(f"ERROR: Failed to import YOLOv5 modules: {e}")
    print("Make sure YOLOv5 is properly cloned in the Source/yolov5 directory")
    sys.exit(1)


def load_config(path):
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class FacemaskDataset(Dataset):
    """Dataset for face mask detection with heatmap channel"""
    
    def __init__(self, image_folder, heatmap_folder, label_folder=None, img_size=(640, 640), transform=None):
        self.image_folder = Path(image_folder)
        self.heatmap_folder = Path(heatmap_folder)
        self.label_folder = Path(label_folder) if label_folder else None
        self.img_size = tuple(img_size)  # (width, height)
        self.transform = transform
        
        # List only image files
        self.files = [p for p in self.image_folder.iterdir()
                     if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.image_folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load heatmap
        hm_path = self.heatmap_folder / f"{img_path.stem}.npy"
        if hm_path.exists():
            heatmap = np.load(str(hm_path))
        else:
            heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        
        # Add channel dimension to heatmap
        if heatmap.ndim == 2:
            heatmap = heatmap[..., np.newaxis]
        
        # Stack and resize
        combined = np.concatenate([img, heatmap], axis=-1)
        combined = cv2.resize(combined, self.img_size)
        combined = np.ascontiguousarray(combined)
        
        # Load labels if present
        labels = []
        if self.label_folder:
            lbl_path = self.label_folder / f"{img_path.stem}.txt"
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            labels.append(list(map(float, line.strip().split())))
        
        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        
        # Convert to tensor: C×H×W, normalized
        inp = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
        
        return inp, labels


def custom_collate_fn(batch):
    """Custom collate function to handle variable number of labels"""
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs, 0)
    return inputs, labels


def labels_to_targets(labels, img_idx):
    """Convert labels to target format for YOLO loss"""
    if isinstance(labels, np.ndarray) and len(labels) > 0:
        labels = torch.from_numpy(labels).float()
    elif isinstance(labels, list) and len(labels) > 0:
        labels = torch.tensor(labels).float()
    else:
        return torch.zeros((0, 6))
    
    # Add image index as first column
    batch_idx = torch.full((labels.shape[0], 1), img_idx, dtype=torch.float32)
    targets = torch.cat([batch_idx, labels], 1)
    
    return targets


def make_dataloaders(cfg):
    """Create training and validation dataloaders"""
    train_ds = FacemaskDataset(
        image_folder=cfg['data']['train_images'],
        heatmap_folder=cfg['data']['train_heatmaps'],
        label_folder=cfg['data']['train_labels'],
        img_size=(cfg['train']['img_size'], cfg['train']['img_size'])
    )
    
    val_ds = FacemaskDataset(
        image_folder=cfg['data']['val_images'],
        heatmap_folder=cfg['data']['val_heatmaps'],
        label_folder=cfg['data']['val_labels'],
        img_size=(cfg['val']['img_size'], cfg['val']['img_size'])
    )
    
    print(f"Train dataset: {len(train_ds)} images")
    print(f"Val dataset: {len(val_ds)} images")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['val']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def build_model(cfg):
    """Build YOLOv5 model with custom number of input channels"""
    # Load model configuration
    model = Model(
        cfg['model']['cfg_path'],
        ch=cfg['model']['input_channels'],  # 4 channels for RGB + heatmap
        nc=cfg['model']['num_classes'],
        anchors=None
    )
    
    # Load hyperparameters
    hyp_path = YOLOV5_DIR / 'data' / 'hyps' / 'hyp.scratch-low.yaml'
    if hyp_path.exists():
        with open(hyp_path, 'r') as f:
            model.hyp = yaml.safe_load(f)
    else:
        # Default hyperparameters if file not found
        model.hyp = {
            'lr0': 0.01,  # initial learning rate
            'lrf': 0.01,  # final OneCycleLR learning rate
            'momentum': 0.937,  # SGD momentum/Adam beta1
            'weight_decay': 0.0005,  # optimizer weight decay
            'warmup_epochs': 3.0,  # warmup epochs
            'warmup_momentum': 0.8,  # warmup initial momentum
            'warmup_bias_lr': 0.1,  # warmup initial bias lr
            'box': 0.05,  # box loss gain
            'cls': 0.5,  # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,  # obj loss gain
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'iou_t': 0.20,  # IoU training threshold
            'anchor_t': 4.0,  # anchor-multiple threshold
            'fl_gamma': 0.0,  # focal loss gamma
            'hsv_h': 0.015,  # image HSV-Hue augmentation
            'hsv_s': 0.7,  # image HSV-Saturation augmentation
            'hsv_v': 0.4,  # image HSV-Value augmentation
            'degrees': 0.0,  # image rotation
            'translate': 0.1,  # image translation
            'scale': 0.5,  # image scale
            'shear': 0.0,  # image shear
            'perspective': 0.0,  # image perspective
            'flipud': 0.0,  # image flip up-down
            'fliplr': 0.5,  # image flip left-right
            'mosaic': 1.0,  # image mosaic
            'mixup': 0.0,  # image mixup
            'copy_paste': 0.0  # segment copy-paste
        }
    
    # Set device
    device = torch.device(cfg['train']['device'])
    model = model.to(device)
    
    # Load pretrained weights if specified
    if cfg['model']['pretrained_weights']:
        checkpoint = torch.load(cfg['model']['pretrained_weights'], map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {cfg['model']['pretrained_weights']}")
    
    return model


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (imgs, labels_list) in enumerate(pbar):
        imgs = imgs.to(device)
        
        # Convert labels to targets
        targets = []
        for i, labels in enumerate(labels_list):
            target = labels_to_targets(labels, i)
            targets.append(target)
        
        if targets:
            targets = torch.cat(targets, 0).to(device)
        else:
            targets = torch.zeros((0, 6)).to(device)
        
        # Forward pass
        pred = model(imgs)
        loss, loss_items = loss_fn(pred, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(train_loader)


def validate(model, val_loader, cfg):
    """Run validation"""
    # Import validation function
    try:
        from validation import validate_model_simple
    except ImportError:
        print("WARNING: validation.py not found. Using simplified validation.")
        return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    device = torch.device(cfg['train']['device'])
    
    metrics = validate_model_simple(
        model=model,
        val_dataloader=val_loader,
        device=device,
        conf_thres=cfg['val']['conf_thres'],
        iou_thres=cfg['val']['iou_thres'],
        img_size=cfg['val']['img_size']
    )
    
    return metrics


def train(cfg):
    """Main training function"""
    # Create output directory
    save_dir = Path(cfg['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(cfg)
    
    # Build model
    model = build_model(cfg)
    device = torch.device(cfg['train']['device'])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    
    # Create loss function
    model.class_weights = torch.ones(cfg['model']['num_classes']).to(device)
    loss_fn = ComputeLoss(model)
    
    # Training loop
    best_map = 0.0
    train_losses = []
    val_metrics = []
    
    print(f"\nStarting training for {cfg['train']['epochs']} epochs...")
    
    for epoch in range(1, cfg['train']['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']} - Train loss: {train_loss:.4f}")
        
        # Validate every N epochs
        if epoch % 1 == 0 or epoch == cfg['train']['epochs']:
            print("Running validation...")
            metrics = validate(model, val_loader, cfg)
            val_metrics.append({'epoch': epoch, **metrics})
            
            print(f"Validation - mAP@0.5: {metrics['mAP']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}")
            
            # Save best model
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': best_map,
                    'config': cfg
                }, save_dir / 'best_model.pt')
                print(f"Saved best model with mAP: {best_map:.4f}")
        
        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': cfg
            }, save_dir / f'epoch{epoch:03d}.pt')
    
    # Save final model
    torch.save({
        'epoch': cfg['train']['epochs'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'config': cfg
    }, save_dir / 'final_model.pt')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_metrics': val_metrics
        }, f, indent=2)
    
    print(f"\nTraining completed! Best mAP: {best_map:.4f}")
    print(f"Models saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Train model
    train(cfg)


if __name__ == '__main__':
    main()