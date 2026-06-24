import os
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "dataset/MM5"
# Point directly to the new internal location of the split metadata
TRAIN_CSV_FILE = os.path.join(DATASET_DIR, "train_dataset.csv")
EVAL_CSV_FILE = os.path.join(DATASET_DIR, "eval_dataset.csv")

def prepare_for_display(img, title, colormap=None):
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "FILE MISSING", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return img

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if colormap is not None:
        img_bgr = cv2.applyColorMap(img_norm, colormap)
    elif len(img_norm.shape) == 2:
        img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_norm

    img_resized = cv2.resize(img_bgr, (640, 480))
    overlay = img_resized.copy()
    cv2.rectangle(overlay, (0, 0), (220, 50), (0, 0, 0), -1)
    img_resized = cv2.addWeighted(overlay, 0.6, img_resized, 0.4, 0)
    cv2.putText(img_resized, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_resized

def run_visualizer():
    # Read the dataset IDs
    if not os.path.exists(TRAIN_CSV_FILE):
        print(f"Error: {TRAIN_CSV_FILE} not found. Please ensure it is in the same directory.")
        return

    if not os.path.exists(EVAL_CSV_FILE):
        print(f"Error: {EVAL_CSV_FILE} not found. Please ensure it is in the same directory.")
        return

    train_df = pd.read_csv(TRAIN_CSV_FILE, header=0, names=["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"])
    eval_df = pd.read_csv(EVAL_CSV_FILE, header=0, names=["ID", "Sequence", "Category", "Subcategory", "Challenges", "Category_Subcategory"])
    ids = pd.concat([train_df, eval_df])["ID"].tolist()
    
    print(f"Loaded {len(ids)} images. Press <SPACE> to progress, <Q> to close.")
    window_name = "MM5 Tri-Modal Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for img_id in ids:
        filename = f"{str(img_id).strip()}.png"
        
        rgb = cv2.imread(os.path.join(DATASET_DIR, "RGB", filename))
        depth = cv2.imread(os.path.join(DATASET_DIR, "Depth", filename), cv2.IMREAD_UNCHANGED)
        thermal = cv2.imread(os.path.join(DATASET_DIR, "Thermal", filename), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(DATASET_DIR, "Class_Annotations", filename), cv2.IMREAD_UNCHANGED)

        q1_rgb = prepare_for_display(rgb, "RGB")
        q2_depth = prepare_for_display(depth, "Depth (16-bit)", cv2.COLORMAP_VIRIDIS)
        q3_thermal = prepare_for_display(thermal, "Thermal (16-bit)", cv2.COLORMAP_INFERNO)
        q4_mask = prepare_for_display(mask, "Class Annotations", cv2.COLORMAP_TWILIGHT_SHIFTED)

        grid = np.vstack((np.hstack((q1_rgb, q2_depth)), np.hstack((q3_thermal, q4_mask))))
        cv2.imshow(window_name, grid)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32:
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visualizer()