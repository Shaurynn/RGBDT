import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from dataset import TriModalSegDataset
from model import TriModalYOLOSeg

# --- 1. Custom Loss Function ---
class FocalDiceLoss(nn.Module):
    """
    Composite Loss Function:
    Focal Loss handles extreme class imbalance (98% background vs 2% anomaly).
    Dice Loss directly optimizes the overlap of the structural boundaries.
    """
    def __init__(self, num_classes, ignore_index=0, gamma=2.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        # A. Focal Loss Calculation
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # B. Dice Loss Calculation
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        smooth = 1e-6
        valid_classes = 0

        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
                
            p = probs[:, c]
            t = targets_one_hot[:, c]
            
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            
            # Dice coefficient: (2 * intersection) / (prediction + target)
            dice_c = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += dice_c.mean()
            valid_classes += 1

        dice_loss = dice_loss / max(valid_classes, 1)

        return focal_loss + (self.dice_weight * dice_loss)


# --- 2. Geometric Metric Evaluation ---
def compute_batch_miou(logits, targets, num_classes, ignore_index=0):
    """Calculates Mean Intersection over Union (mIoU) for a given batch."""
    preds = torch.argmax(logits, dim=1)
    ious = []
    
    for c in range(num_classes):
        if c == ignore_index:
            continue
            
        pred_inds = preds == c
        target_inds = targets == c
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / float(max(union, 1)))

    return sum(ious) / max(len(ious), 1) if ious else 0.0


# --- 3. Maximization Early Stopping ---
class EarlyStopping:
    """Early stops the training if mIoU doesn't improve after a given patience."""
    def __init__(self, patience=15, min_delta=0.001, verbose=True, path='weights/best_trimodal_seg.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(current_score, model)
            return

        if current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"--> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, current_score, model):
        if self.verbose:
            print(f"--> Validation mIoU increased to {current_score:.4f}. Saving optimal weights...")
        torch.save(model.state_dict(), self.path)


# --- 4. Architectural Inspection ---
def inspect_model_architecture(model, num_classes, device):
    """Generates a Keras-like summary table for the PyTorch architecture."""
    print("\n" + "="*75)
    print(f"Analyzing TriModalYOLOSeg Architecture ({num_classes} Output Channels)")
    print("="*75)
    
    input_size = (1, 5, 480, 640)
    
    # Generate torchinfo summary
    model_stats = summary(
        model, 
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
        verbose=0,
        device=device
    )
    print(model_stats)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- PyTorch Parameter Verification ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Params: {total_params - trainable_params:,}\n")
    print("="*75 + "\n")


def main():
    BATCH_SIZE = 8
    MAX_EPOCHS = 300
    PATIENCE = 20
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = "weights"
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Executing training on device: {DEVICE}")

    print("Initializing datasets...")
    train_dataset = TriModalSegDataset(csv_file="dataset/MM5/train_dataset.csv")
    eval_dataset = TriModalSegDataset(csv_file="dataset/MM5/eval_dataset.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    NUM_CLASSES = train_dataset.num_classes

    # --- Initialize & Inspect Model ---
    model = TriModalYOLOSeg(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    inspect_model_architecture(model, NUM_CLASSES, DEVICE)
    
    criterion = FocalDiceLoss(num_classes=NUM_CLASSES, ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler(DEVICE.type)
    
    checkpoint_path = os.path.join(WEIGHTS_DIR, "best_trimodal_seg.pt")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True, path=checkpoint_path)

    writer = SummaryWriter(log_dir="runs/TriModal_POC_Exp1")
    print("--- TensorBoard Live Tracking Enabled ---")
    print("Run 'tensorboard --logdir=runs' in a new terminal to view curves.\n")

    for epoch in range(MAX_EPOCHS):
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for tensors, masks in loop:
            tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(device_type=DEVICE.type):
                logits = model(tensors)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_miou_accum = 0.0
        batches = 0
        
        with torch.no_grad():
            eval_loop = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Eval ]")
            for tensors, masks in eval_loop:
                tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
                
                with autocast(device_type=DEVICE.type):
                    logits = model(tensors)
                    loss = criterion(logits, masks)
                
                val_loss += loss.item()
                val_miou_accum += compute_batch_miou(logits, masks, NUM_CLASSES, ignore_index=0)
                batches += 1
                
        avg_val_loss = val_loss / batches
        avg_val_miou = val_miou_accum / batches
        
        # Log to TensorBoard
        writer.add_scalars("Loss", {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Validation_mIoU", avg_val_miou, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}")

        # Check Early Stopping
        early_stopper(avg_val_miou, model)
        
        if early_stopper.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
            print("Validation mIoU has plateaued. Training terminated to prevent overfitting.")
            break

    writer.flush()
    writer.close()
    print(f"\nTraining complete. Best model weights saved to: {checkpoint_path}")

if __name__ == '__main__':
    main()