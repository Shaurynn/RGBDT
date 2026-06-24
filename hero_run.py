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

# --- 1. Custom Loss Function (With Empty Class Masking) ---
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, gamma=1.6627, dice_weight=0.6250):
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
                
            t = targets_one_hot[:, c]
            
            # Skip empty classes to prevent penalizing softmax noise
            if t.sum() == 0:
                continue
                
            p = probs[:, c]
            
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            
            dice_c = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += dice_c
            valid_classes += 1

        if valid_classes > 0:
            dice_loss = dice_loss / valid_classes
        else:
            dice_loss = 0.0

        return focal_loss + (self.dice_weight * dice_loss)

# --- 2. Geometric Metric Evaluation ---
def compute_batch_miou(logits, targets, num_classes, ignore_index=0):
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
    def __init__(self, patience=40, min_delta=0.001, verbose=True, path='weights/best_trimodal_seg.pt'):
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
    print("\n" + "="*75)
    print(f"Analyzing TriModalYOLOSeg Architecture ({num_classes} Output Channels)")
    print("="*75)
    
    model_stats = summary(
        model, 
        input_size=(1, 5, 480, 640),
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
    print("="*75 + "\n")

def main():
    # --- The Hero Run Configurations ---
    BATCH_SIZE = 8
    MAX_EPOCHS = 300
    PATIENCE = 40
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = "weights"
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Executing Phase 4 Production Training on: {DEVICE}")

    train_dataset = TriModalSegDataset(csv_file="dataset/MM5/train_dataset.csv", split="train")
    eval_dataset = TriModalSegDataset(csv_file="dataset/MM5/eval_dataset.csv", split="eval")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    NUM_CLASSES = train_dataset.num_classes

    model = TriModalYOLOSeg(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    inspect_model_architecture(model, NUM_CLASSES, DEVICE)
    
    # 1. Hardcoded Optuna Loss Parameters
    criterion = FocalDiceLoss(
        num_classes=NUM_CLASSES, 
        ignore_index=0, 
        gamma=1.6627,        # Optuna optimized
        dice_weight=0.6250   # Optuna optimized
    )
    
    # 2. Hardcoded Optuna Optimizer Parameters (SGD + Nesterov)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.0753,           # Optuna optimized
        weight_decay=0.0003, # Optuna optimized
        momentum=0.9685,     # Optuna optimized
        nesterov=True        # Optuna optimized
    )
    
    # 3. Hardcoded Optuna Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=MAX_EPOCHS, 
        eta_min=1e-6
    )
    
    scaler = GradScaler(DEVICE.type)
    checkpoint_path = os.path.join(WEIGHTS_DIR, "best_trimodal_seg.pt")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True, path=checkpoint_path)

    writer = SummaryWriter(log_dir="runs/TriModal_Hero_Run")
    print("--- TensorBoard Live Tracking Enabled ---")

    for epoch in range(MAX_EPOCHS):
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
        scheduler.step()
        
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
        
        writer.add_scalars("Loss", {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Validation_mIoU", avg_val_miou, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}")

        early_stopper(avg_val_miou, model)
        
        if early_stopper.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
            print("Validation mIoU has plateaued. Production training terminated.")
            break

    writer.flush()
    writer.close()
    print(f"\nTraining complete. Production weights saved to: {checkpoint_path}")

if __name__ == '__main__':
    main()