import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TriModalSegDataset(Dataset):
    def __init__(self, csv_file, root_dir="dataset/MM5", target_size=(480, 640)):
        """
        Args:
            csv_file (str): Full path to 'train_dataset.csv' or 'eval_dataset.csv'.
            root_dir (str): Directory containing RGB, Depth, Thermal, and Class_Annotations.
        """
        self.data_frame = pd.read_csv(csv_file, header=0, names=["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"])
        self.root_dir = root_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = str(self.data_frame.iloc[idx, 0]).strip()
        filename = f"{img_id}.png"

        # 1. Load RGB (8-bit, 3 channels)
        rgb_path = os.path.join(self.root_dir, "RGB", filename)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
        rgb = rgb.astype(np.float32) / 255.0

        # 2. Load Depth (16-bit, 1 channel)
        depth_path = os.path.join(self.root_dir, "Depth", filename)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)

        # 3. Load Thermal LWIR (16-bit, 1 channel)
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
        
        # 5. Stack and convert to PyTorch standard [C, H, W]
        rgbdt = np.dstack([rgb, depth, thermal])
        tensor_rgbdt = torch.from_numpy(rgbdt).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).long()

        return tensor_rgbdt, tensor_mask

# Example Usage with updated pathing
if __name__ == '__main__':
    TRAIN_CSV = "dataset/MM5/train_dataset.csv"
    
    if os.path.exists(TRAIN_CSV):
        train_dataset = TriModalSegDataset(csv_file=TRAIN_CSV)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        print("DataLoader successfully reading from updated CSV layout location.")