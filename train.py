import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TriModalSegDataset
from model import TriModalYOLOSeg

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, min_delta=1e-4, verbose=True, path='weights/best_trimodal_seg.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # First epoch setup
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            return

        # If loss didn't improve by at least min_delta
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"--> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # If loss improved
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"--> Validation loss decreased to {val_loss:.4f}. Saving checkpoint...")
        torch.save(model.state_dict(), self.path)


def main():
    # --- Configuration ---
    BATCH_SIZE = 8
    MAX_EPOCHS = 300       # Set high; Early Stopping will terminate training naturally
    PATIENCE = 15          # Epochs to wait for improvement before stopping
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = "weights"
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Executing training on device: {DEVICE}")

    # --- Data Loading ---
    print("Initializing datasets...")
    train_dataset = TriModalSegDataset(csv_file="dataset/MM5/train_dataset.csv")
    eval_dataset = TriModalSegDataset(csv_file="dataset/MM5/eval_dataset.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    NUM_CLASSES = train_dataset.num_classes
    print(f"Detected {NUM_CLASSES} classes.")

    # --- Model, Loss, Optimizer ---
    model = TriModalYOLOSeg(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Initialize Early Stopping
    checkpoint_path = os.path.join(WEIGHTS_DIR, "best_trimodal_seg.pt")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True, path=checkpoint_path)

    # --- TensorBoard Instrumentation ---
    # Creates a unique log directory for this experiment
    writer = SummaryWriter(log_dir="runs/TriModal_POC_Exp1")
    print("\n--- TensorBoard Live Tracking Enabled ---")
    print("Open a new terminal and run: tensorboard --logdir=runs\n")

    # --- Training Loop ---
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch_idx, (tensors, masks) in enumerate(loop):
            tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with autocast():
                logits = model(tensors)
                loss = criterion(logits, masks)
                
            # Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Optional: Log batch-level training loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch_Loss/Train", loss.item(), global_step)
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            eval_loop = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Eval ]")
            for tensors, masks in eval_loop:
                tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
                
                with autocast():
                    logits = model(tensors)
                    loss = criterion(logits, masks)
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(eval_loader)
        
        # Log epoch-level metrics to TensorBoard
        writer.add_scalars("Epoch_Loss", {
            'Train': avg_train_loss,
            'Validation': avg_val_loss
        }, epoch + 1)
        
        # Log learning rate to track decay (if you add a scheduler later)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning_Rate", current_lr, epoch + 1)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        early_stopper(avg_val_loss, model)
        
        if early_stopper.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
            print("Validation loss has plateaued. Training terminated.")
            break

    writer.flush()
    writer.close()
    print(f"\nTraining complete. Best model weights saved to: {checkpoint_path}")

if __name__ == '__main__':
    main()