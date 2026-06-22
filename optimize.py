import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Import established modules
from dataset import TriModalSegDataset
from model import TriModalYOLOSeg

# --- Global Configurations ---
BATCH_SIZE = 8
MAX_EPOCHS = 30  # Lightweight epoch cap for rapid hyperparameter sweeps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "dataset/MM5/train_dataset.csv"
EVAL_CSV = "dataset/MM5/eval_dataset.csv"

# --- Mirroring Production Loss & Metric Logic ---
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, gamma=2.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

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
            dice_c = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += dice_c.mean()
            valid_classes += 1

        dice_loss = dice_loss / max(valid_classes, 1)
        return focal_loss + (self.dice_weight * dice_loss)

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


def objective(trial):
    print(f"\n--- Starting Trial {trial.number} ---")

    # 1. Hyperparameter Search Space Definition
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    # 2. Data & Model Setup
    train_dataset = TriModalSegDataset(csv_file=TRAIN_CSV)
    eval_dataset = TriModalSegDataset(csv_file=EVAL_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    NUM_CLASSES = train_dataset.num_classes

    model = TriModalYOLOSeg(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FocalDiceLoss(num_classes=NUM_CLASSES, ignore_index=0)
    
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scaler = GradScaler(DEVICE.type)
    best_val_miou = 0.0

    # 3. Execution Loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        for tensors, masks in train_loader:
            tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(device_type=DEVICE.type):
                logits = model(tensors)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation Evaluation
        model.eval()
        val_miou_accum = 0.0
        batches = 0
        
        with torch.no_grad():
            for tensors, masks in eval_loader:
                tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    logits = model(tensors)
                val_miou_accum += compute_batch_miou(logits, masks, NUM_CLASSES, ignore_index=0)
                batches += 1
                
        avg_val_miou = val_miou_accum / batches

        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou

        # 4. Report Metric to Optuna Pruner
        trial.report(avg_val_miou, epoch)

        # 5. Pruning Execution
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} (mIoU: {avg_val_miou:.4f})")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} completed. Best Validation mIoU: {best_val_miou:.4f}")
    return best_val_miou  # Optuna maximizes this returned score


def main():
    print(f"Initializing Hyperparameter Optimization on {DEVICE}...")

    study_name = "trimodal_yolo_sweep"
    storage_name = f"sqlite:///{study_name}.db"
    
    # CRITICAL: Direction changed to MAXIMIZE geometric overlap (mIoU)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)
    )

    print(f"Executing search. Press Ctrl+C to safely pause the study at any time.")
    
    try:
        study.optimize(objective, n_trials=30, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptimization manually paused.")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n--- Study Statistics ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")

    if complete_trials:
        print("\n--- Best Trial ---")
        trial = study.best_trial
        print(f"Highest Validation mIoU: {trial.value:.4f}")
        print("Optimal Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()