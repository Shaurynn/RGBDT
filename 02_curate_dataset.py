import os
import json
import cv2
import numpy as np
import pandas as pd
import fiftyone as fo
from PIL import Image

DATASET_DIR = "dataset/MM5"
PROXY_DIR = "dataset/MM5_Proxies"
MASK_MULTIPLIER = 8  # The visual scaling factor for the UI

def load_json_mapping():
    """Parses label_mapping.json to guarantee exact integer-to-string matching."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Masks will not render correctly.")
        return {}
        
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
            
    print(f"Loaded {len(mapping)} classes from label_mapping.json")
    return mapping

def create_ui_proxy(raw_path, filename, modality, mapping=None):
    """Generates 8-bit UI proxies. Applies visual scaling to masks."""
    if not os.path.exists(raw_path):
        return None, None
        
    modality_dir = os.path.join(PROXY_DIR, modality)
    os.makedirs(modality_dir, exist_ok=True)
    proxy_path = os.path.join(modality_dir, filename)
    
    if modality == "Class_Annotations":
        # 1. Load via PIL to extract the true integer IDs
        img = Image.open(raw_path)
        img_arr = np.array(img)
        
        if len(img_arr.shape) > 2:
            img_arr = img_arr[:, :, 0]
            
        img_arr = img_arr.astype(np.uint8)
        unique_ids = np.unique(img_arr).tolist()
        
        # 2. Extract string names based on the ORIGINAL IDs
        present_classes = []
        if mapping:
            present_classes = [mapping[uid] for uid in unique_ids if uid in mapping and mapping[uid].lower() != "background"]
            
        # 3. Apply the Visual Scaling Factor (Multiply by 8)
        # We use clip to ensure we don't overflow the 8-bit 255 limit
        visual_img_arr = np.clip(img_arr.astype(np.uint16) * MASK_MULTIPLIER, 0, 255).astype(np.uint8)
            
        if not os.path.exists(proxy_path):
            cv2.imwrite(proxy_path, visual_img_arr)
            
        return proxy_path, present_classes

    # Return cached proxy if it exists for sensor feeds
    if os.path.exists(proxy_path):
        return proxy_path, None

    if modality == "RGB":
        img = cv2.imread(raw_path)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(proxy_path, img)
    elif modality == "Depth":
        img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(proxy_path, img)
    elif modality == "Thermal":
        img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        cv2.imwrite(proxy_path, img)
        
    return proxy_path, None

def create_fiftyone_dataset():
    dataset_name = "MM5-TriModal-Structural-POC"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = True
    
    # --- The Dictionary Shift ---
    # Shift the target dictionary keys to match the multiplied proxy pixels
    raw_mask_targets = load_json_mapping()
    if raw_mask_targets:
        visual_mask_targets = {k * MASK_MULTIPLIER: v for k, v in raw_mask_targets.items()}
        dataset.default_mask_targets = visual_mask_targets
        
    dataset.add_group_field("group", default="rgb")

    splits = {
        "Train": os.path.join(DATASET_DIR, "train_dataset.csv"),
        "Eval": os.path.join(DATASET_DIR, "eval_dataset.csv")
    }
    csv_columns = ["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"]
    
    samples = []
    
    for split_tag, csv_path in splits.items():
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path, header=0, names=csv_columns)
        img_ids = df["ID"].tolist()
        
        print(f"Compiling {split_tag} split and applying visual mask scaling...")
        for img_id in img_ids:
            filename = f"{str(img_id).strip()}.png"
            
            raw_rgb = os.path.join(DATASET_DIR, "RGB", filename)
            raw_depth = os.path.join(DATASET_DIR, "Depth", filename)
            raw_thermal = os.path.join(DATASET_DIR, "Thermal", filename)
            raw_mask = os.path.join(DATASET_DIR, "Class_Annotations", filename)
            
            if not os.path.exists(raw_rgb): continue

            # Generate Proxies
            proxy_rgb, _ = create_ui_proxy(raw_rgb, filename, "RGB")
            proxy_depth, _ = create_ui_proxy(raw_depth, filename, "Depth")
            proxy_thermal, _ = create_ui_proxy(raw_thermal, filename, "Thermal")
            # The mask returned here has been multiplied by MASK_MULTIPLIER
            proxy_mask, present_classes = create_ui_proxy(raw_mask, filename, "Class_Annotations", mapping=raw_mask_targets)

            scene_labels = None
            if present_classes:
                scene_labels = fo.Classifications(
                    classifications=[fo.Classification(label=cls) for cls in present_classes]
                )

            scene_group = fo.Group()
            
            # --- RGB Sample ---
            sample_rgb = fo.Sample(filepath=proxy_rgb)
            sample_rgb["group"] = scene_group.element("rgb")
            sample_rgb.tags.append(split_tag)
            
            # --- Depth Sample ---
            if proxy_depth:
                sample_depth = fo.Sample(filepath=proxy_depth)
                sample_depth["group"] = scene_group.element("depth")
                sample_depth.tags.append(split_tag)
                
            # --- Thermal Sample ---
            if proxy_thermal:
                sample_thermal = fo.Sample(filepath=proxy_thermal)
                sample_thermal["group"] = scene_group.element("thermal")
                sample_thermal.tags.append(split_tag)
                
            # --- Attach Native Labels and Scaled Masks ---
            if proxy_mask:
                seg_mask = fo.Segmentation(mask_path=proxy_mask)
                
                sample_rgb["ground_truth"] = seg_mask
                if scene_labels: sample_rgb["scene_classes"] = scene_labels
                
                if 'sample_depth' in locals():
                    sample_depth["ground_truth"] = seg_mask
                    if scene_labels: sample_depth["scene_classes"] = scene_labels
                
                if 'sample_thermal' in locals():
                    sample_thermal["ground_truth"] = seg_mask
                    if scene_labels: sample_thermal["scene_classes"] = scene_labels

            samples.append(sample_rgb)
            if 'sample_depth' in locals(): samples.append(sample_depth)
            if 'sample_thermal' in locals(): samples.append(sample_thermal)

            if 'sample_depth' in locals(): del sample_depth
            if 'sample_thermal' in locals(): del sample_thermal
            
    dataset.add_samples(samples)
    print(f"\nIngestion Complete.")
    
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    create_fiftyone_dataset()