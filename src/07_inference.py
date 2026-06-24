import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import your custom architecture
from models import TriModalYOLOSeg

# --- Configurations ---
WEIGHTS_PATH = os.path.join("weights", "best_trimodal_seg.pt")
DATASET_DIR = os.path.join("dataset", "MM5")
OUTPUT_DIR = "inference_results"
TARGET_SIZE = (480, 640)

def load_label_mapping():
    """Loads the JSON mapping to understand the output classes."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Label mapping not found at {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    mapping = {0: "Background"}
    for k, v in data.items():
        if isinstance(v, int) or str(v).isdigit(): mapping[int(v)] = str(k)
        elif isinstance(k, int) or str(k).isdigit(): mapping[int(k)] = str(v)
        
    num_classes = max(mapping.keys()) + 1
    return mapping, num_classes

def generate_color_palette(num_classes):
    """Generates distinct colors for the anomalies. Class 0 (Background) is black."""
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background is completely black
    return colors

def preprocess_multimodal_input(rgb_path, depth_path, thermal_path):
    """Loads and precisely aligns the 5-channel input tensor."""
    if not all(os.path.exists(p) for p in [rgb_path, depth_path, thermal_path]):
        raise FileNotFoundError("One or more input sensor files are missing.")

    # 1. Load and Resize
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (TARGET_SIZE[1], TARGET_SIZE[0]))

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.resize(thermal, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # 2. Strict Normalization (Must match dataset.py exactly)
    rgb_norm = rgb.astype(np.float32) / 255.0

    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else depth

    t_min, t_max = thermal.min(), thermal.max()
    thermal_norm = (thermal - t_min) / (t_max - t_min) if t_max > t_min else thermal

    # 3. Stack into 5-channel tensor
    rgbdt = np.dstack([rgb_norm, depth_norm, thermal_norm])
    tensor_rgbdt = torch.from_numpy(rgbdt).permute(2, 0, 1).float().unsqueeze(0) # Add batch dimension

    return tensor_rgbdt, rgb, depth, thermal

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Cross-Platform Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing inference on: {device}")

    # 2. Load Model Configuration
    print("Loading architecture and weights...")
    class_mapping, num_classes = load_label_mapping()
    colors = generate_color_palette(num_classes)
    
    model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes).to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.eval()
        print("Weights loaded successfully.")
    else:
        print(f"[!] WARNING: Weights file {WEIGHTS_PATH} not found.")
        print("Model will output random noise until the Hero Run finishes.")

    # 3. Define a sample to test (Change 'sample_id' to an actual ID from your eval set)
    sample_id = "00001" # <--- UPDATE THIS TO A REAL FILENAME IN YOUR DATASET
    
    rgb_path = os.path.join(DATASET_DIR, "RGB", f"{sample_id}.png")
    depth_path = os.path.join(DATASET_DIR, "Depth", f"{sample_id}.png")
    thermal_path = os.path.join(DATASET_DIR, "Thermal", f"{sample_id}.png")

    print(f"Processing structural data for sample: {sample_id}...")
    tensor_input, rgb_vis, depth_vis, thermal_vis = preprocess_multimodal_input(rgb_path, depth_path, thermal_path)
    tensor_input = tensor_input.to(device)

    # 4. Neural Network Execution
    with torch.no_grad():
        logits = model(tensor_input)
        # Collapse the 31 channels down to a single 2D grid containing the highest-confidence class IDs
        predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 5. Diagnostic Visualization
    print("Generating visual diagnostic report...")
    
    # Convert prediction grid to an RGB mask
    pred_mask_rgb = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    unique_classes_detected = np.unique(predictions)
    
    for cls_idx in unique_classes_detected:
        pred_mask_rgb[predictions == cls_idx] = colors[cls_idx]

    # Blend the prediction mask with the original RGB image for structural context
    alpha = 0.5
    blended_image = cv2.addWeighted(rgb_vis, 1 - alpha, pred_mask_rgb, alpha, 0)

    # Build the Matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"TriModal Structural Segmentation - Sample: {sample_id}", fontsize=16)

    axes[0, 0].imshow(rgb_vis)
    axes[0, 0].set_title("Raw RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(blended_image)
    axes[0, 1].set_title("AI Prediction Overlay")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(depth_vis, cmap='plasma')
    axes[1, 0].set_title("Depth Map (Distance)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(thermal_vis, cmap='inferno')
    axes[1, 1].set_title("Thermal Signature")
    axes[1, 1].axis("off")

    # Create dynamic legend based on what was actually detected
    legend_elements = []
    for cls_idx in unique_classes_detected:
        if cls_idx == 0: continue # Skip labeling the background
        color_normalized = colors[cls_idx] / 255.0
        legend_elements.append(Patch(facecolor=color_normalized, edgecolor='w', label=class_mapping[cls_idx]))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    output_filepath = os.path.join(OUTPUT_DIR, f"{sample_id}_diagnostic.png")
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Done! Diagnostic report saved to: {output_filepath}")

if __name__ == "__main__":
    main()