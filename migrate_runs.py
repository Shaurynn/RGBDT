import os
import json
import shutil
import datetime

def main():
    print("--- Starting MLOps State Migration ---")

    # 1. Define Old Paths
    old_weights_path = os.path.join("weights", "best_trimodal_seg.pt")
    old_logs_dir = os.path.join("runs", "TriModal_Hero_Run")

    # Ensure the old weights actually exist before proceeding
    if not os.path.exists(old_weights_path):
        raise FileNotFoundError(f"[!] Could not find {old_weights_path}. Check your directory.")

    # 2. Define New Paths
    model_name = "TriModalYOLOSeg"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # We label this as 'Hero' so the state machine knows to start 'Microtune' next
    new_run_dir = os.path.join("results", model_name, f"{timestamp}_Hero")
    
    os.makedirs(new_run_dir, exist_ok=True)
    os.makedirs(os.path.join(new_run_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(new_run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(new_run_dir, "explainability"), exist_ok=True)

    print(f"[*] Created new R&D directory: {new_run_dir}")

    # 3. Migrate the Weights
    new_weights_path = os.path.join(new_run_dir, "best_model.pt")
    shutil.copy2(old_weights_path, new_weights_path)
    print(f"[*] Successfully copied weights to {new_weights_path}")

    # 4. Migrate the TensorBoard Logs (if they exist)
    if os.path.exists(old_logs_dir):
        for item in os.listdir(old_logs_dir):
            s = os.path.join(old_logs_dir, item)
            d = os.path.join(new_run_dir, "logs", item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print("[*] Successfully migrated TensorBoard logs.")
    else:
        print("[!] Old TensorBoard logs not found. Skipping log migration.")

    # 5. Generate the 'results.json' (The State Machine Trigger)
    # We inject your previously reported mIoU score here
    results_payload = {
        "model_architecture": model_name,
        "initialization_params": {"in_channels": 5, "num_classes": 32},
        "phase": "hero",
        "completed_at": datetime.datetime.now().isoformat(),
        "final_test_mIoU": 0.4604, 
        "best_validation_mIoU": 0.4604,
        "epochs_trained": 300,
        "migration_note": "Manually migrated from legacy flat-folder structure."
    }
    
    with open(os.path.join(new_run_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
        
    print("[*] Generated results.json state trigger.")
    print("\n--- Migration Complete ---")
    print("You can now run `uv run train.py`. The state machine will detect the completed Hero run and automatically begin the Microtune phase.")

if __name__ == "__main__":
    main()