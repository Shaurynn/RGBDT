import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import TriModalYOLOSeg

# --- Configurations ---
DATASET_DIR = "dataset/MM5"
WEIGHTS_PATH = "weights/best_trimodal_seg.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (480, 640)

class SemanticSegmentationTarget:
    """Target function that extracts activation gradients for an exact class channel."""
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[0, self.category, :, :].sum()

def load_mapping():
    """Loads JSON map and guarantees strict integer key lookups."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {0: "Background"}
    for k, v in data.items():
        mapping[int(v)] = str(k)
    return mapping

def load_sample(filename):
    """Generates standardized 5-channel model tensors and visualization feeds."""
    rgb_path = os.path.join(DATASET_DIR, "RGB", filename)
    depth_path = os.path.join(DATASET_DIR, "Depth", filename)
    thermal_path = os.path.join(DATASET_DIR, "Thermal", filename)
    mask_path = os.path.join(DATASET_DIR, "Class_Annotations", filename)

    rgb_vis = cv2.imread(rgb_path)
    rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
    rgb_vis = cv2.resize(rgb_vis, (TARGET_SIZE[1], TARGET_SIZE[0]))
    rgb_float = np.float32(rgb_vis) / 255.0

    # Read and unify structural dimensions
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if depth.max() > depth.min(): 
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.resize(thermal, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if thermal.max() > thermal.min(): 
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min())

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    gt_mask = cv2.resize(gt_mask, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)

    # Pack into 5-channel structure
    rgbdt = np.dstack([rgb_float, depth, thermal])
    tensor = torch.from_numpy(rgbdt).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, rgb_float, thermal, gt_mask

def main():
    print(f"Initializing Grad-CAM Explainability Pipeline on {DEVICE}")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Run train.py first to generate: {WEIGHTS_PATH}")
        return

    mapping = load_mapping()
    num_classes = max(mapping.keys()) + 1  # Standardizes to 32 channels
    
    model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # Targets final structural decoder integration layer
    target_layers = [model.dec3]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Randomly sample validation item
    eval_csv = os.path.join(DATASET_DIR, "eval_dataset.csv")
    df = pd.read_csv(eval_csv)
    sample_row = df.sample(1).iloc[0]
    filename = f"{str(sample_row['ID']).strip()}.png"
    print(f"Targeting Visual Validation File: {filename}")

    tensor, rgb_float, thermal_arr, gt_mask = load_sample(filename)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_classes = [c for c in np.unique(pred_mask) if c != 0]

    if not predicted_classes:
        print("Model predicted only background for this file. Re-run to sample an anomaly target.")
        return

    os.makedirs("visualizations", exist_ok=True)

    for target_class in predicted_classes:
        class_name = mapping.get(target_class, f"Class_{target_class}")
        print(f"Extracting feature activations for category: {class_name}")

        grayscale_cam = cam(input_tensor=tensor, targets=[SemanticSegmentationTarget(target_class)])[0, :]
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

        # Plot structural subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Multi-Modal Explainability Profile: {class_name} (ID: {filename})", fontsize=14)

        axes[0, 0].imshow(rgb_float)
        axes[0, 0].set_title("RGB Source Profile")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(thermal_arr, cmap='inferno')
        axes[0, 1].set_title("Thermal Modality Profile")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(gt_mask, cmap='nipy_spectral', vmin=0, vmax=num_classes)
        axes[1, 0].set_title("Ground Truth Segmentation Target")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=num_classes)
        axes[1, 1].set_title("Model Spatial Prediction")
        axes[1, 1].axis('off')

        axes[0, 2].imshow(grayscale_cam, cmap='jet')
        axes[0, 2].set_title("Raw Activation Heatmap (dec3)")
        axes[0, 2].axis('off')

        axes[1, 2].imshow(cam_image)
        axes[1, 2].set_title("Grad-CAM Structural Alignment")
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = f"visualizations/cam_{filename.split('.')[0]}_{class_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved explainability visual map to: {save_path}")
        plt.show()

if __name__ == "__main__":
    main()