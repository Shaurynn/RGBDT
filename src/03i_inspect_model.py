import os
import json
import torch
from torchinfo import summary

# Import your custom 5-channel architecture
from model import TriModalYOLOSeg

DATASET_DIR = "dataset/MM5"

def get_num_classes():
    """Reads the JSON mapping to dynamically determine the YOLO head size."""
    json_path = os.path.join(DATASET_DIR, "label_mapping.json")
    if not os.path.exists(json_path):
        # Fallback default if json isn't present in the current working directory
        return 32
        
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

def main():
    # 1. Dynamically discover the required output channels
    num_classes = get_num_classes()
    
    # 2. Instantiate the model
    model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes)
    
    # 3. Define the exact input size expected by the architecture
    # Format: (Batch_Size, Channels, Height, Width)
    # Using a batch size of 1 for the architectural summary
    input_size = (1, 5, 480, 640)
    
    print("\n" + "="*70)
    print(f"Analyzing TriModalYOLOSeg Architecture ({num_classes} Output Channels)")
    print("="*70)
    
    # 4. Generate the Keras-like summary report
    # row_settings=["depth", "var_names"] provides granular nested block tracking for CSP modules
    model_stats = summary(
        model, 
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
        verbose=0
    )
    
    # Print the beautifully formatted table
    print(model_stats)
    
    # 5. Native PyTorch programmatic parameter fallback count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- Quick Programmatic Verification ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Params: {total_params - trainable_params:,}\n")

if __name__ == "__main__":
    main()