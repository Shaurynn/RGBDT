import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import your architecture
from model import TriModalYOLOSeg

# --- Configurations ---
DATASET_DIR = "dataset/MM5"
WEIGHTS_PATH = "weights/best_trimodal_seg.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (480, 640)

class SemanticSegmentationTarget:
    """
    Custom Grad-CAM target for Semantic Segmentation.
    Instead of a single scalar, it sums the spatial logits for a specific class.
    This forces Grad-CAM to highlight all pixels that contributed to finding this class.
    """
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # model_output shape: [1, num_classes, H, W]
        # We sum the activations of the specific class channel across the spatial dimensions
        return model_output[0, self.category, :, :].sum()

def load_mapping():
    """Loads the JSON to map integers back to human-readable strings."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    mapping = {}
    if "categories" in data:
        for cat in data["categories"]: mapping[int(cat["id"])] = cat["name"]
    else:
        for k, v in data.items():
            if isinstance(v, int) or str(v).isdigit(): mapping[int(v)] = str(k)
            elif isinstance(k, int) or str(k).isdigit(): mapping[int(k)] = str(v)
    return mapping

def load_sample(filename):
    """Loads and standardizes the 5-channel tensor and the base 8-bit visual images."""
    rgb_path = os.path.join(DATASET_DIR, "RGB", filename)
    depth_path = os.path.join(DATASET_DIR, "Depth", filename)
    thermal_path = os.path.join(DATASET_DIR, "Thermal", filename)
    mask_path = os.path.join(DATASET_DIR, "Class_Annotations", filename)

    # Load 8-bit RGB for visual overlay later
    rgb_vis = cv2.imread(rgb_path)
    rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
    rgb_vis = cv2.resize(rgb_vis, (TARGET_SIZE[1], TARGET_SIZE[0]))
    rgb_float = np.float32(rgb_vis) / 255.0  # Required by Grad-CAM overlay

    # Build the 5-Channel Tensor
    rgb_t = rgb_float.copy()
    
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min: depth = (depth - d_min) / (d_max - d_min)
    
    thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.resize(thermal, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    t_min, t_max = thermal.min(), thermal.max()
    if t_max > t_min: thermal = (thermal - t_min) / (t_max - t_min)

    # Load Ground Truth Mask
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    gt_mask = cv2.resize(gt_mask, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)

    rgbdt = np.dstack([rgb_t, depth, thermal])
    tensor = torch.from_numpy(rgbdt).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, rgb_float, thermal, gt_mask

def main():
    print(f"Initializing Grad-CAM Explainability Pipeline on {DEVICE}")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}. Please train the model first.")
        return

    mapping = load_mapping()
    num_classes = max(mapping.keys()) + 1
    
    # 1. Load Model
    model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # 2. Define the Target Layer
    # We hook into `dec3`, which is the final major integration block before the upsample to full resolution.
    # This block contains the highest-level semantic features combined with spatial awareness.
    target_layers = [model.dec3]

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Randomly select an image from the Evaluation Split
    eval_csv = os.path.join(DATASET_DIR, "eval_dataset.csv")
    df = pd.read_csv(eval_csv, header=0, names=["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"])
    
    # Randomly sample 1 image ID
    sample_row = df.sample(1).iloc[0]
    filename = f"{str(sample_row['ID']).strip()}.png"
    print(f"Analyzing Image: {filename} ({sample_row['Category_Subcategory']})")

    # Load Data
    tensor, rgb_float, thermal_arr, gt_mask = load_sample(filename)
    tensor = tensor.to(DEVICE)

    # 4. Forward Pass to get Predictions
    with torch.no_grad():
        logits = model(tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Find which classes the model actually predicted in this image (ignore 0/Background)
    predicted_classes = np.unique(pred_mask)
    predicted_classes = [c for c in predicted_classes if c != 0]

    if not predicted_classes:
        print("Model predicted only background for this image. Run again to sample another.")
        return

    # 5. Generate Visualizations for each predicted class
    os.makedirs("visualizations", exist_ok=True)

    for target_class in predicted_classes:
        class_name = mapping.get(target_class, f"Class_{target_class}")
        print(f"Generating Activation Map for: {class_name}")

        # Define custom target
        targets = [SemanticSegmentationTarget(target_class)]

        # Compute CAM
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]

        # Overlay the heatmap heavily onto the RGB image
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

        # --- Plotting ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Grad-CAM Diagnostics: {class_name} (Image ID: {filename})", fontsize=16)

        # Row 1: The Raw Inputs
        axes[0, 0].imshow(rgb_float)
        axes[0, 0].set_title("Input RGB")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(thermal_arr, cmap='inferno')
        axes[0, 1].set_title("Input Thermal (LWIR)")
        axes[0, 1].axis('off')
        
        # Row 2: The Logic
        axes[1, 0].imshow(gt_mask, cmap='nipy_spectral', vmin=0, vmax=num_classes)
        axes[1, 0].set_title("Ground Truth Mask")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=num_classes)
        axes[1, 1].set_title("Model Prediction")
        axes[1, 1].axis('off')

        # The Star of the Show: The Activation Map
        axes[0, 2].imshow(grayscale_cam, cmap='jet')
        axes[0, 2].set_title("Raw Heatmap (Target Layer: dec3)")
        axes[0, 2].axis('off')

        axes[1, 2].imshow(cam_image)
        axes[1, 2].set_title(f"Grad-CAM Overlay ({class_name})")
        axes[1, 2].axis('off')

        plt.tight_layout()
        
        # Save to disk
        save_path = f"visualizations/cam_{filename.split('.')[0]}_{class_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved diagnostic plot to: {save_path}")
        
        # Display interactively
        plt.show()

if __name__ == "__main__":
    main()