import os
import pandas as pd
import fiftyone as fo

def create_fiftyone_dataset():
    dataset_dir = "dataset/MM5"
    dataset_name = "MM5-TriModal-Structural-POC"
    
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = True
    
    # Absolute lookup path referencing files inside dataset/MM5/
    splits = {
        "Train": os.path.join(dataset_dir, "train_dataset.csv"),
        "Eval": os.path.join(dataset_dir, "eval_dataset.csv")
    }
    
    samples = []
    for split_tag, csv_path in splits.items():
        if not os.path.exists(csv_path):
            print(f"Error: Missing split tracking document at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path, header=0, names=["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"])
        img_ids = df["ID"].tolist()
        
        for img_id in img_ids:
            filename = f"{str(img_id).strip()}.png"
            
            rgb_path = os.path.join(dataset_dir, "RGB", filename)
            depth_path = os.path.join(dataset_dir, "Depth", filename)
            thermal_path = os.path.join(dataset_dir, "Thermal", filename)
            mask_path = os.path.join(dataset_dir, "Class_Annotations", filename)
            
            if not os.path.exists(rgb_path):
                continue
            
            sample = fo.Sample(filepath=rgb_path)
            sample.tags.append(split_tag)
            sample["depth_source_path"] = depth_path
            sample["thermal_source_path"] = thermal_path
            
            if os.path.exists(mask_path):
                sample["ground_truth"] = fo.Segmentation(mask_path=mask_path)
            
            samples.append(sample)
            
    dataset.add_samples(samples)
    print(f"Ingested {len(dataset)} items into FiftyOne UI dashboard database successfully.")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    create_fiftyone_dataset()