import os
import cv2
import torch
import json
import pandas as pd
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

class TriModalSegDataset(Dataset):
    def __init__(self, csv_file, root_dir="dataset/MM5", target_size=(480, 640), split="train"):
        """
        Args:
            csv_file (str): Path to 'train_dataset.csv' or 'eval_dataset.csv'.
            root_dir (str): Directory containing the sensor folders.
            target_size (tuple): Target resize dimensions (H, W).
            split (str): "train" applies heavy multi-modal augmentations. "eval" applies none.
        """
        csv_columns = ["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"]
        self.data_frame = pd.read_csv(csv_file, header=0, names=csv_columns)
        self.root_dir = root_dir
        self.target_size = target_size
        self.split = split
        
        self.class_mapping, self.num_classes = self._load_json_mapping()

        # 1. Synchronized Spatial Transformations (Applied identically across all arrays)
        self.spatial_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # Modernized API for shifting, scaling, and rotating
            A.Affine(
                scale=(0.95, 1.05),               
                translate_percent=(-0.05, 0.05),  
                rotate=(-15, 15),                 
                border_mode=cv2.BORDER_CONSTANT,  # <-- Corrected Albumentations argument
                fill=0,                           # <-- Corrected Albumentations argument (was cval)
                fill_mask=0,                      # <-- Corrected Albumentations argument (was cval_mask)
                p=0.5
            ),
        ], additional_targets={'depth': 'mask', 'thermal': 'mask'})

    def _load_json_mapping(self):
        json_path = os.path.join(self.root_dir, "label_mapping.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        mapping = {0: "Background"}
        for k, v in data.items():
            if isinstance(v, int) or str(v).isdigit(): mapping[int(v)] = str(k)
            elif isinstance(k, int) or str(k).isdigit(): mapping[int(k)] = str(v)
            
        num_classes = max(mapping.keys()) + 1
        return mapping, num_classes

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = str(self.data_frame['ID'].iloc[idx]).strip()
        filename = f"{img_id}.png"

        # --- 1. Load Raw Base Data ---
        rgb = cv2.imread(os.path.join(self.root_dir, "RGB", filename))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))

        depth = cv2.imread(os.path.join(self.root_dir, "Depth", filename), cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        thermal = cv2.imread(os.path.join(self.root_dir, "Thermal", filename), cv2.IMREAD_UNCHANGED)
        thermal = cv2.resize(thermal, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        mask = cv2.imread(os.path.join(self.root_dir, "Class_Annotations", filename), cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # --- 2. Apply Augmentations (TRAIN SPLIT ONLY) ---
        if self.split == "train":
            # A. Synchronized Spatial Execution
            augmented = self.spatial_transform(image=rgb, mask=mask, depth=depth, thermal=thermal)
            rgb = augmented['image']
            mask = augmented['mask']
            depth = augmented['depth']
            thermal = augmented['thermal']

            # B. Modality-Isolated Photometric Execution
            # Modulate RGB lighting without destroying thermal/depth context
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.7, 1.3) # Contrast
                beta = np.random.uniform(-30, 30)   # Brightness
                rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
            
            # Simulate sensor calibration drift on the Thermal feed
            if np.random.rand() < 0.3:
                thermal = thermal * np.random.uniform(0.85, 1.15)
                
            # Simulate lidar/depth scattering noise
            if np.random.rand() < 0.1:
                noise = np.random.normal(0, depth.std() * 0.05, depth.shape).astype(np.float32)
                depth = depth + noise

        # --- 3. Strict Normalization ---
        # Crucial: Normalization must happen AFTER augmentations to maintain standard ranges
        rgb = rgb.astype(np.float32) / 255.0

        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min: depth = (depth - d_min) / (d_max - d_min)

        t_min, t_max = thermal.min(), thermal.max()
        if t_max > t_min: thermal = (thermal - t_min) / (t_max - t_min)

        # --- 4. Tensor Stacking ---
        rgbdt = np.dstack([rgb, depth, thermal])
        tensor_rgbdt = torch.from_numpy(rgbdt).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).long()

        return tensor_rgbdt, tensor_mask