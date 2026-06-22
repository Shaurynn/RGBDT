import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # <-- UPDATED IMPORT
from tqdm import tqdm

from dataset import TriModalSegDataset
from model import TriModalYOLOSeg

BATCH_SIZE = 8
MAX_EPOCHS = 40  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "dataset/MM5/train_dataset.csv"
EVAL_CSV = "dataset/MM5/eval_dataset.csv"

def objective(trial):
    print(f"\n--- Starting Trial {trial.number} ---")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    train_dataset = TriModalSegDataset(csv_file=TRAIN_CSV)
    eval_dataset = TriModalSegDataset(csv_file=EVAL_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = train_dataset.num_classes

    model = TriModalYOLOSeg(in_channels=5, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # <-- UPDATED SCALER
    scaler = GradScaler(DEVICE.type)
    best_val_loss = float('inf')

    for epoch in range(MAX_EPOCHS):
        model.train()
        for tensors, masks in train_loader:
            tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            # <-- UPDATED AUTOCAST
            with autocast(device_type=DEVICE.type):
                logits = model(tensors)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tensors, masks in eval_loader:
                tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
                
                # <-- UPDATED AUTOCAST
                with autocast(device_type=DEVICE.type):
                    logits = model(tensors)
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(eval_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} (Val Loss: {avg_val_loss:.4f})")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} completed. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    print(f"Initializing Hyperparameter Optimization on {DEVICE}...")

    study_name = "trimodal_yolo_sweep"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
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
        print(f"Lowest Validation Loss: {trial.value:.4f}")
        print("Optimal Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()