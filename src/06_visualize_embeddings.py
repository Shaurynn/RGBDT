import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import fiftyone as fo
import fiftyone.brain as fob
from tqdm import tqdm

# Import the architecture we defined in the previous step
from model import TriModalYOLOSeg

DATASET_DIR = "dataset/MM5"
WEIGHTS_PATH = "weights/best_trimodal_seg.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingExtractor(nn.Module):
    """
    Wraps the TriModalYOLOSeg model to extract the 512D bottleneck features
    instead of the final segmentation mask.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Global Average Pooling flattens the [B, 512, H, W] spatial map into a [B, 512, 1, 1] vector
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Pass the tensor through the encoder blocks
        x_stem = self.base_model.stem(x)
        x1 = self.base_model.layer1(x_stem)
        x2 = self.base_model.layer2(x1)
        x3 = self.base_model.layer3(x2)  # The deep bottleneck (512 channels)
        
        # Pool and flatten
        emb = self.pool(x3)
        return emb.view(emb.size(0), -1)

def get_num_classes():
    """Reads the JSON mapping to dynamically determine the YOLO head size."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    if "categories" in data:
        for cat in data["categories"]: mapping[int(cat["id"])] = cat["name"]
    else:
        for k, v in data.items():
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit()): mapping[int(v)] = str(k)
            elif isinstance(k, int) or (isinstance(k, str) and k.isdigit()): mapping[int(k)] = str(v)
            
    return max(mapping.keys()) + 1

def load_raw_tensor(filename, target_size=(480, 640)):
    """Reconstructs the 5-channel tensor exactly as it was during training."""
    rgb_path = os.path.join(DATASET_DIR, "RGB", filename)
    depth_path = os.path.join(DATASET_DIR, "Depth", filename)
    thermal_path = os.path.join(DATASET_DIR, "Thermal", filename)

    # 1. RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_size[1], target_size[0])).astype(np.float32) / 255.0

    # 2. Depth
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min: depth = (depth - d_min) / (d_max - d_min)

    # 3. Thermal
    thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    thermal = cv2.resize(thermal, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    t_min, t_max = thermal.min(), thermal.max()
    if t_max > t_min: thermal = (thermal - t_min) / (t_max - t_min)

    # Stack and convert to PyTorch standard [1, 5, 480, 640]
    rgbdt = np.dstack([rgb, depth, thermal])
    return torch.from_numpy(rgbdt).permute(2, 0, 1).float().unsqueeze(0)

def main():
    print(f"Preparing Embedding Space on: {DEVICE}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Could not find trained weights at {WEIGHTS_PATH}.")
        return

    # 1. Load the Model
    num_classes = get_num_classes()
    base_model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes)
    base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    base_model.eval()
    
    # Wrap it to output embeddings instead of segmentation logits
    extractor = EmbeddingExtractor(base_model).to(DEVICE)

    # 2. Load the FiftyOne Dataset
    dataset_name = "MM5-TriModal-Structural-POC"
    if dataset_name not in fo.list_datasets():
        print("Error: FiftyOne dataset not found. Please run curation script first.")
        return
        
    dataset = fo.load_dataset(dataset_name)
    
    # Isolate only the RGB view (since the scene groups are anchored to RGB)
    rgb_view = dataset.match(fo.ViewField("group.name") == "rgb")
    
    print(f"Extracting 512D deep features for {len(rgb_view)} scenes...")
    
    embeddings = []
    
    # 3. Forward Pass Loop
    with torch.no_grad():
        for sample in tqdm(rgb_view, desc="Computing Tensors"):
            # The curation script saved proxy paths like 'MM5_Proxies/RGB/123.png'
            # We must extract just '123.png' to load the raw data arrays
            filename = os.path.basename(sample.filepath)
            
            tensor = load_raw_tensor(filename).to(DEVICE)
            
            # Extract and move to CPU
            emb = extractor(tensor).cpu().numpy().squeeze()
            embeddings.append(emb)
            
    embeddings = np.array(embeddings)
    
    # 4. Compute UMAP Dimensionality Reduction natively in FiftyOne Brain
    print("Computing UMAP projection...")
    fob.compute_visualization(
        rgb_view,
        embeddings=embeddings,
        num_dims=2,
        method="umap",
        brain_key="yolo_embeddings",
        verbose=True
    )
    
    print("Launch sequence complete. Opening FiftyOne App...")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()