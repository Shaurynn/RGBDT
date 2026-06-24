import os
import shutil
import pandas as pd

# Define source (raw download) and target (clean structure)
RAW_DIR = "data/MM5/"
TARGET_DIR = "dataset/MM5/"

# Folder mapping: Target -> Source
TARGET_FOLDERS = {
    "RGB": "RGB3",
    "Depth": "D16",
    "Thermal": "T16",
    "Class_Annotations": "ANNO_CLASS"
}

def setup_directories():
    for target in TARGET_FOLDERS.keys():
        os.makedirs(os.path.join(TARGET_DIR, target), exist_ok=True)

def migrate_data():
    # Construct the paths to the CSVs inside the target directory
    train_csv_path = os.path.join(TARGET_DIR, "train_dataset.csv")
    eval_csv_path = os.path.join(TARGET_DIR, "eval_dataset.csv")
    
    if not os.path.exists(train_csv_path) or not os.path.exists(eval_csv_path):
        print(f"Error: Ensure train_dataset.csv and eval_dataset.csv are placed in {TARGET_DIR} before running.")
        return

    # Read the CSVs out of dataset/MM5/
    train_df = pd.read_csv(train_csv_path, header=None, names=["id"])
    eval_df = pd.read_csv(eval_csv_path, header=None, names=["id"])
    all_ids = pd.concat([train_df, eval_df])["id"].tolist()

    print(f"Found {len(all_ids)} total image IDs to migrate based on CSV logs.")

    missing_files = []
    for img_id in all_ids:
        filename = f"{str(img_id).strip()}.png"

        for target_folder, source_folder in TARGET_FOLDERS.items():
            src_path = os.path.join(RAW_DIR, source_folder, filename)
            tgt_path = os.path.join(TARGET_DIR, target_folder, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, tgt_path)
            else:
                missing_files.append(src_path)

    if missing_files:
        print(f"WARNING: Could not find {len(missing_files)} source files. Showing first 5:")
        for f in missing_files[:5]:
            print(f" - {f}")
    else:
        print("All files migrated successfully.")

if __name__ == "__main__":
    print("Initializing clean tri-modal dataset structure...")
    setup_directories()
    migrate_data()
    print("Migration complete.")