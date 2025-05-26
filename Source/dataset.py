"""
Dataset classes for Face Mask Detection with heatmap channel
"""

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

def fixed_resize(combined_input, target_size=(640, 640)):
    """Resize combined input to target size"""
    resized = cv2.resize(combined_input, target_size)
    return resized

class FacemaskDataset(Dataset):
    """Dataset for face mask detection with heatmap channel"""
    
    def __init__(self, image_folder, heatmap_folder, label_folder=None, transform=None):
        """
        Args:
            image_folder: Directory with RGB images
            heatmap_folder: Directory with corresponding heatmaps (.npy files)
            label_folder: Directory with label files (optional)
            transform: Optional transform to apply
        """
        self.image_folder = image_folder
        self.heatmap_folder = heatmap_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or create heatmap
        heatmap_filename = os.path.splitext(img_filename)[0] + ".npy"
        heatmap_path = os.path.join(self.heatmap_folder, heatmap_filename)
        
        if os.path.exists(heatmap_path):
            heatmap = np.load(heatmap_path)
        else:
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Resize image and heatmap **separately**
        image = cv2.resize(image, (640, 640))
        heatmap = cv2.resize(heatmap, (640, 640))

        # Add channel to heatmap if needed
        if heatmap.ndim == 2:
            heatmap = heatmap[..., np.newaxis]

        # Apply transform to both if defined
        if self.transform:
            image = self.transform(image)
            heatmap = self.transform(heatmap)

        # Concatenate after resizing
        combined_input = np.concatenate([image, heatmap], axis=-1)
        combined_input = np.ascontiguousarray(combined_input)

        # Load labels
        label = None
        if self.label_folder is not None:
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(self.label_folder, label_filename)
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                label = [list(map(float, line.strip().split())) for line in lines if line.strip()]
                label = np.array(label) if label else np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((0, 5), dtype=np.float32)

        return combined_input, label

class FacemaskDatasetRGB(Dataset):
    """Standard RGB dataset for comparison"""
    
    def __init__(self, image_folder, label_folder=None, transform=None):
        """
        Args:
            image_folder: Directory with RGB images
            label_folder: Directory with label files (optional)
            transform: Optional transform to apply
        """
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Resize to fixed size
        image = cv2.resize(image, (640, 640))
        
        # Ensure contiguous
        image = np.ascontiguousarray(image)
        
        # Load labels if available
        label = None
        if self.label_folder is not None:
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(self.label_folder, label_filename)
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                label = []
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    label.append(parts)
                label = np.array(label) if label else np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((0, 5), dtype=np.float32)
        
        return image, label

def custom_collate_fn(batch):
    """Custom collate function for 4-channel input"""
    inputs = []
    labels = []
    
    for combined_input, label in batch:
        # Convert to tensor and permute to channel-first
        tensor = torch.from_numpy(combined_input).permute(2, 0, 1).float() / 255.0
        inputs.append(tensor)
        
        # Handle labels
        if label is None or len(label) == 0:
            labels.append(torch.empty(0, 5))
        else:
            labels.append(torch.from_numpy(label).float())
    
    # Stack inputs
    inputs = torch.stack(inputs, dim=0)
    return inputs, labels

def custom_collate_fn_rgb(batch):
    """Custom collate function for RGB only (3 channels)"""
    inputs = []
    labels = []
    
    for image, label in batch:
        # Convert to tensor and permute to channel-first
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        inputs.append(tensor)
        
        # Handle labels
        if label is None or len(label) == 0:
            labels.append(torch.empty(0, 5))
        else:
            labels.append(torch.from_numpy(label).float())
    
    # Stack inputs
    inputs = torch.stack(inputs, dim=0)
    return inputs, labels