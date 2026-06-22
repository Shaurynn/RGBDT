import os
import cv2
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TriModalSegDataset(Dataset):
    def __init__(self, csv_file, root_dir="dataset/MM5", target_size=(480, 640)):
        """
        Args:
            csv_file (str): Full path to 'train_dataset.csv' or 'eval_dataset.csv'.
            root_dir (str): Directory containing RGB, Depth, Thermal, Class_Annotations, and label_mapping.json.
        """
        csv_columns = ["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"]
        self.data_frame = pd.read_csv(csv_file, header=0, names=csv_columns)
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Dynamically load the true class mapping and determine the required YOLO head size
        self.class_mapping, self.num_classes = self._load_json_mapping()

    def _load_json_mapping(self):
        """Parses label_mapping.json to determine the exact number of classes for the neural network."""
        json_path = os.path.join(self.root_dir, "label_mapping.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"CRITICAL: {json_path} not found. Cannot determine network output size.")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        mapping = {}
        if "categories" in data:
            for cat in data["categories"]:
                mapping[int(cat["id"])] = cat["name"]
        else:
            for k, v in data.items():
                if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                    mapping[int(v)] = str(k)
                elif isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                    mapping[int(k)] = str(v)
        
        # In PyTorch, if your max class ID is 31, your network must output 32 channels (0 through 31).
        if not mapping:
            raise ValueError("label_mapping.json is empty or poorly formatted.")
            
        max_id = max(mapping.keys())
        num_classes = max_id + 1
        
        print(f"Dataset Initialized: Found {len(mapping)} classes. Network head requires {num_classes} channels.")
        return mapping, num_classes

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Explicitly pull from the "ID" column
        img_id = str(self.data_frame['ID'].iloc[idx]).strip()
        filename = f"{img_id}.png"

        # 1. Load RGB
        rgb_path = os.path.join(self.root_dir, "RGB", filename)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
        rgb = rgb.astype(np.float32) / 255.0

        # 2. Load Depth
        depth_path = os.path.join(self.root_dir, "Depth", filename)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)

        # 3. Load Thermal LWIR
        thermal_path = os.path.join(self.root_dir, "Thermal", filename)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        thermal = cv2.resize(thermal, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        thermal = thermal.astype(np.float32)
        thermal_min, thermal_max = thermal.min(), thermal.max()
        if thermal_max > thermal_min:
            thermal = (thermal - thermal_min) / (thermal_max - thermal_min)

        # 4. Load Semantic Mask
        mask_path = os.path.join(self.root_dir, "Class_Annotations", filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 5. Stack and convert
        rgbdt = np.dstack([rgb, depth, thermal])
        tensor_rgbdt = torch.from_numpy(rgbdt).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).long()

        return tensor_rgbdt, tensor_mask

if __name__ == '__main__':
    TRAIN_CSV = "dataset/MM5/train_dataset.csv"
    if os.path.exists(TRAIN_CSV):
        train_dataset = TriModalSegDataset(csv_file=TRAIN_CSV)
        print(f"Success! Model num_classes is exposed as: {train_dataset.num_classes}")