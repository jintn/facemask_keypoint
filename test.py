import argparse
import os
import sys
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Fix path issues
CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CURRENT_DIR))

# Add Source directory to path
SOURCE_DIR = CURRENT_DIR / "Source"
if SOURCE_DIR.exists():
    sys.path.insert(0, str(SOURCE_DIR))

# Add YOLOv5 directory to path
YOLOV5_DIR = SOURCE_DIR / "yolov5"
if YOLOV5_DIR.exists():
    sys.path.insert(0, str(YOLOV5_DIR))
else:
    print(f"ERROR: YOLOv5 directory not found at {YOLOV5_DIR}")
    sys.exit(1)

# Import custom modules
from dataset import FacemaskDataset, custom_collate_fn
from validation import validate_model_simple, process_yolo_output

# Import YOLOv5 modules
try:
    from models.yolo import Model
    from utils.general import xywh2xyxy
except ImportError as e:
    print(f"ERROR: Failed to import YOLOv5 modules: {e}")
    sys.exit(1)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def debug_predictions(model, test_loader, device, config, num_samples=5):
    """Debug function to visualize predictions"""
    model.eval()
    
    output_dir = Path(config['test']['output_dir']) / "debug_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (imgs, labels_list) in enumerate(test_loader):
            if batch_idx >= 1:  # Only process first batch
                break
                
            imgs = imgs.to(device)
            
            # Get raw outputs
            outputs = model(imgs)
            print(f"\nRaw output shape: {outputs[0].shape}")
            print(f"Sample raw predictions (first 5):")
            print(outputs[0][0, :5, :])  # First image, first 5 predictions
            
            # Process outputs
            predictions = process_yolo_output(
                outputs,
                conf_thres=config['test']['conf_thres'],
                iou_thres=config['test']['iou_thres'],
                nc=config['model']['num_classes']
            )
            
            # Visualize each image
            for img_idx in range(min(num_samples, len(imgs))):
                img = imgs[img_idx]
                gt_labels = labels_list[img_idx]
                pred = predictions[img_idx]
                
                # Convert image to numpy
                img_np = img.cpu().numpy()
                # Take only RGB channels (first 3)
                img_rgb = img_np[:3].transpose(1, 2, 0)
                img_rgb = (img_rgb * 255).astype(np.uint8)
                img_vis = img_rgb.copy()
                
                # Draw ground truth boxes (green)
                print(f"\nImage {img_idx}:")
                print(f"  Ground truth boxes: {len(gt_labels)}")
                for label in gt_labels:
                    cls, x, y, w, h = label
                    # Convert normalized coords to pixels
                    x1 = int((x - w/2) * config['test']['img_size'])
                    y1 = int((y - h/2) * config['test']['img_size'])
                    x2 = int((x + w/2) * config['test']['img_size'])
                    y2 = int((y + h/2) * config['test']['img_size'])
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_vis, f"GT:{int(cls)}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw predictions (red)
                if pred is not None:
                    print(f"  Predicted boxes: {len(pred)}")
                    for det in pred:
                        x1, y1, x2, y2, conf, cls = det
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img_vis, f"P:{int(cls)}:{conf:.2f}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    print(f"  Predicted boxes: 0")
                
                # Save visualization
                save_path = output_dir / f"debug_{batch_idx}_{img_idx}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
                print(f"  Saved to: {save_path}")
                
                # Also visualize heatmap channel
                if img.shape[0] > 3:
                    heatmap = img_np[3]
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_rgb)
                    plt.title("RGB Image")
                    plt.subplot(1, 2, 2)
                    plt.imshow(heatmap, cmap='hot')
                    plt.title("Heatmap Channel")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"heatmap_{batch_idx}_{img_idx}.jpg")
                    plt.close()


def test_model(config_path=None, model_path=None, test_data_path=None):
    """
    Main testing function
    
    Args:
        config_path: Path to configuration file (default: ./config.yaml)
        model_path: Path to saved model (overrides config if provided)
        test_data_path: Path to test data (overrides config if provided)
    """
    
    # Load configuration
    if config_path is None:
        config_path = './config.yaml'
    config = load_config(config_path)
    
    # Override paths if provided
    if model_path is not None:
        config['test']['model_path'] = model_path
    if test_data_path is not None:
        config['data']['test_images'] = test_data_path
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Loading model architecture from: {config['model']['cfg_path']}")
    model = Model(
        config['model']['cfg_path'],
        ch=config['model']['input_channels'],
        nc=config['model']['num_classes']
    ).to(device)
    
    # Set model attributes that might be missing
    model.class_weights = torch.ones(config['model']['num_classes']).to(device)
    
    # Load hyperparameters (important for loss calculation)
    hyp_path = YOLOV5_DIR / 'data' / 'hyps' / 'hyp.scratch-low.yaml'
    if hyp_path.exists():
        with open(hyp_path, 'r') as f:
            model.hyp = yaml.safe_load(f)
    
    # Load trained weights
    print(f"Loading weights from: {config['test']['model_path']}")
    checkpoint = torch.load(config['test']['model_path'], map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create test dataset
    print(f"\nLoading test dataset from: {config['data']['test_images']}")
    test_dataset = FacemaskDataset(
        image_folder=config['data']['test_images'],
        heatmap_folder=config['data']['test_heatmaps'],
        label_folder=config['data']['test_labels'],
        transform=None
    )
    
    # Check ground truth distribution
    print("\nGround truth distribution:")
    class_counts = {i: 0 for i in range(config['model']['num_classes'])}
    total_boxes = 0
    for i in range(len(test_dataset)):
        _, labels = test_dataset[i]
        total_boxes += len(labels)
        for label in labels:
            class_counts[int(label[0])] += 1
    
    for cls, count in class_counts.items():
        class_name = config['class_names'].get(cls, f'Class {cls}')
        print(f"  {class_name}: {count} boxes")
    print(f"  Total boxes: {total_boxes}")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test']['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    print(f"\nTest dataset size: {len(test_dataset)} images")
    print(f"Number of batches: {len(test_loader)}")
    
    # Debug predictions on a few samples
    print("\n" + "="*50)
    print("DEBUGGING PREDICTIONS")
    print("="*50)
    debug_predictions(model, test_loader, device, config)
    
    # Run validation
    print("\n" + "="*50)
    print("RUNNING FULL EVALUATION")
    print("="*50)
    
    # Try with different confidence thresholds
    conf_thresholds = [0.001, 0.01, 0.1, 0.25]
    for conf_thresh in conf_thresholds:
        print(f"\nTesting with conf_thresh={conf_thresh}")
        results = validate_model_simple(
            model=model,
            val_dataloader=test_loader,
            device=device,
            conf_thres=conf_thresh,
            iou_thres=config['test']['iou_thres'],
            img_size=config['test']['img_size']
        )
        
        print(f"  mAP@0.5: {results['mAP']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        
        if results['mAP'] > 0:
            break
    
    # Use best results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"mAP@0.5: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # Per-class results
    if 'ap_per_class' in results and results['ap_per_class']:
        print("\nPer-class results:")
        print("-"*30)
        for i, (ap, p, r) in enumerate(zip(
            results['ap_per_class'],
            results['p_per_class'],
            results['r_per_class']
        )):
            class_name = config['class_names'].get(i, f'Class {i}')
            print(f"{class_name:15s}: AP={ap:.3f}, P={p:.3f}, R={r:.3f}")
    
    # Save results
    if config['test']['save_predictions']:
        output_dir = Path(config['test']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config used
        with open(output_dir / 'test_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Test Face Mask Detection Model')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (overrides config)')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data (overrides config)')
    
    args = parser.parse_args()
    
    # Run testing
    results = test_model(
        config_path=args.config,
        model_path=args.model,
        test_data_path=args.test_data
    )
    
    # Return success
    return 0 if results['mAP'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())