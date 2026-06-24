import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Import established modules
from dataset import TriModalSegDataset
from models import TriModalYOLOSeg

# --- Global Configurations ---
# Increased to 100 to give SGD and Schedulers room to converge
MAX_EPOCHS = 100 
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "dataset/MM5/train_dataset.csv"
EVAL_CSV = "dataset/MM5/eval_dataset.csv"

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

    # 1. EXPANDED SEARCH SPACE
    # ------------------------
    # Loss Function Tuning
    gamma = trial.suggest_float("gamma", 1.0, 5.0)
    dice_weight = trial.suggest_float("dice_weight", 0.5, 3.0)

    # Optimizer Selection
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop", "SGD"])
    
    # SGD often requires a higher starting learning rate than Adam
    lr_upper_bound = 1e-1 if optimizer_name == "SGD" else 1e-2
    lr = trial.suggest_float("lr", 1e-5, lr_upper_bound, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Learning Rate Scheduler Selection
    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealing", "ReduceLROnPlateau", "None"])

    # 2. Data & Model Setup
    train_dataset = TriModalSegDataset(csv_file=TRAIN_CSV, split="train")
    eval_dataset = TriModalSegDataset(csv_file=EVAL_CSV, split="eval")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    NUM_CLASSES = train_dataset.num_classes

    model = TriModalYOLOSeg(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Inject dynamically suggested loss parameters
    criterion = FocalDiceLoss(num_classes=NUM_CLASSES, ignore_index=0, gamma=gamma, dice_weight=dice_weight)
    
    # 3. Dynamic Optimizer Injection
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("rmsprop_momentum", 0.4, 0.95)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "SGD":
        # SGD requires heavy momentum tuning and optionally Nesterov acceleration
        sgd_momentum = trial.suggest_float("sgd_momentum", 0.8, 0.99)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=sgd_momentum, nesterov=nesterov)

    # 4. Dynamic Scheduler Injection
    if scheduler_name == "CosineAnnealing":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    elif scheduler_name == "ReduceLROnPlateau":
        patience = trial.suggest_int("plateau_patience", 3, 10)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)
    else:
        scheduler = None

    scaler = GradScaler(DEVICE.type)
    best_val_miou = 0.0

    # 5. Execution Loop
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

        # Step the Scheduler
        if scheduler is not None:
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(avg_val_miou)  # Plateaus are based on the metric
            else:
                scheduler.step()              # Cosine steps blindly per epoch

        # 6. Report Metric to Optuna Hyperband Pruner
        trial.report(avg_val_miou, epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} (mIoU: {avg_val_miou:.4f})")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} completed. Best Validation mIoU: {best_val_miou:.4f}")
    return best_val_miou


def main():
    print(f"Initializing HPO on {DEVICE}...")


    # New DB name to prevent collision with old sweeps
    study_name = "trimodal_research_sweep"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"Run 'uv run optuna-dashboard {storage_name}' to visualize the optimization process.")
    
    # HYPERBAND PRUNER (ASHA)
    # This is the gold standard for massive sweeps. It kills bottom-performing models at 
    # checkpoints (e.g., epoch 10, epoch 30), allocating full compute only to the best models.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10, 
        max_resource=MAX_EPOCHS, 
        reduction_factor=3
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner
    )

    print(f"Executing deep search. Press Ctrl+C to safely pause.")
    
    try:
        # Pushed to 100 trials to explore the massively expanded dimensions
        study.optimize(objective, n_trials=100, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptimization manually paused.")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n--- Research Sweep Statistics ---")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Pruned (Killed Early): {len(pruned_trials)}")
    print(f"Completed (Full 100 Epochs): {len(complete_trials)}")

    if complete_trials:
        print("\n--- Absolute Best Trial ---")
        trial = study.best_trial
        print(f"Highest Validation mIoU: {trial.value:.4f}")
        print("Optimal Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()