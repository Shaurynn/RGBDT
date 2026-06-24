import os
import glob
import json
import torch
import cv2
import optuna # Added HPO engine
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TriModalSegDataset
import models  

# --- 1. Custom Loss & Metrics ---
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, gamma=1.6627, dice_weight=0.6250):
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
        valid_classes = 0

        for c in range(self.num_classes):
            if c == self.ignore_index: continue
            t = targets_one_hot[:, c]
            if t.sum() == 0: continue
            
            p = probs[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            
            dice_c = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss += dice_c
            valid_classes += 1

        dice_loss = (dice_loss / valid_classes) if valid_classes > 0 else 0.0
        return focal_loss + (self.dice_weight * dice_loss)

def compute_batch_miou(logits, targets, num_classes, ignore_index=0):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for c in range(num_classes):
        if c == ignore_index: continue
        pred_inds = preds == c
        target_inds = targets == c
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0: ious.append(intersection / float(union))
    return sum(ious) / max(len(ious), 1) if ious else 0.0

# --- 2. Explainability Engine (Segmentation Grad-CAM) ---
class SemanticGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d): target_layer = module
        if target_layer is None: return

        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        class_mask = logits[:, target_class, :, :]
        loss = class_mask.sum()
        loss.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam

# --- 3. The State Manager ---
class ExperimentManager:
    def __init__(self, model_instance, base_dir="results"):
        self.model_name = model_instance.__class__.__name__
        self.model_dir = os.path.join(base_dir, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        # Added HPO to the curriculum sequence
        self.phase_sequence = ["baseline", "hpo", "hero", "microtune", "export"]
        
    def detect_state(self):
        existing_runs = sorted(glob.glob(os.path.join(self.model_dir, "*_*")))
        if not existing_runs:
            return self._create_new_run("baseline", resume_from=None)
            
        latest_run = existing_runs[-1]
        run_name = os.path.basename(latest_run)
        
        if os.path.exists(os.path.join(latest_run, "results.json")):
            current_phase = run_name.split("_")[-1].lower()
            if current_phase == self.phase_sequence[-1]:
                print(f"[*] Pipeline for {self.model_name} is fully complete.")
                return None
                
            next_phase_idx = self.phase_sequence.index(current_phase) + 1
            next_phase = self.phase_sequence[next_phase_idx]
            
            # Curriculum Learning Weight Routing
            if next_phase == "hpo":
                inherit_weights = os.path.join(latest_run, "best_model.pt") 
            elif next_phase == "hero":
                baseline_runs = [r for r in existing_runs if r.lower().endswith("_baseline")]
                inherit_weights = os.path.join(baseline_runs[-1], "best_model.pt") if baseline_runs else None
            else: 
                hero_runs = [r for r in existing_runs if r.lower().endswith("_hero")]
                inherit_weights = os.path.join(hero_runs[-1], "best_model.pt") if hero_runs else None
                
            return self._create_new_run(next_phase, resume_from=inherit_weights)
        else:
            return self._resume_run(latest_run)

    def _create_new_run(self, phase, resume_from):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.model_dir, f"{timestamp}_{phase.capitalize()}")
        
        os.makedirs(run_dir)
        os.makedirs(os.path.join(run_dir, "weights"))
        os.makedirs(os.path.join(run_dir, "logs"))
        if phase != "hpo":
            os.makedirs(os.path.join(run_dir, "explainability")) 
        
        state = {
            "run_dir": run_dir,
            "phase": phase,
            "is_resume": False,
            "inherit_weights": resume_from,
            "start_epoch": 0,
            "best_miou": 0.0,
            "patience_counter": 0
        }
        self._save_state(run_dir, state)
        return state

    def _resume_run(self, run_dir):
        state_file = os.path.join(run_dir, "state.json")
        with open(state_file, 'r') as f: state = json.load(f)
        state["is_resume"] = True
        return state
        
    def _save_state(self, run_dir, state_dict):
        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state_dict, f, indent=4)

# --- 4. Dynamic Configuration Injector ---
def build_phase_config(phase, model, max_epochs, model_dir, num_classes):
    lr, gamma, dice, opt_type = 0.0753, 1.6627, 0.6250, "SGD"
    momentum, weight_decay = 0.9685, 0.0003
    
    # Auto-Load Optuna Parameters if HPO completed previously
    if phase in ["hero", "microtune"]:
        existing_runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
        hpo_runs = [r for r in existing_runs if r.lower().endswith("_hpo")]
        if hpo_runs:
            params_path = os.path.join(hpo_runs[-1], "best_params.json")
            if os.path.exists(params_path):
                with open(params_path, "r") as f: p = json.load(f)
                lr = p.get("lr", lr)
                gamma = p.get("gamma", gamma)
                dice = p.get("dice_weight", dice)
                opt_type = p.get("optimizer", opt_type)
                momentum = p.get("sgd_momentum", momentum)
                weight_decay = p.get("weight_decay", weight_decay)

    criterion = FocalDiceLoss(num_classes=num_classes, ignore_index=0, gamma=gamma, dice_weight=dice)

    if phase == "baseline":
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "hero":
        if opt_type == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "microtune":
        if opt_type == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        else:
            opt = optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        
    return opt, sch, criterion

# --- 5. Isolated HPO Engine ---
def run_hpo_phase(run_dir, inherit_weights, ModelClass, model_kwargs, train_loader, eval_loader, num_classes, device):
    """Executes a silent, high-speed Optuna sweep to find optimal loss and optimizer parameters."""
    print("--- Initiating Optuna Hyperparameter Sweep (30 Trials) ---")
    study_db_path = os.path.join(run_dir, "optuna_study.db")
    
    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights):
            model.load_state_dict(torch.load(inherit_weights))
            
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 1.0, 5.0)
        dice = trial.suggest_float("dice_weight", 0.5, 3.0)
        opt_type = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        
        criterion = FocalDiceLoss(num_classes, gamma=gamma, dice_weight=dice)
        
        if opt_type == "AdamW":
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            momentum = trial.suggest_float("sgd_momentum", 0.8, 0.99)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
            
        scaler = GradScaler(device.type)
        best_miou = 0.0
        
        for epoch in range(30): # 30 Epoch fast-pruning limit
            model.train()
            for tensors, masks in train_loader:
                tensors, masks = tensors.to(device), masks.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    loss = criterion(model(tensors), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_miou, batches = 0.0, 0
            with torch.no_grad():
                for tensors, masks in eval_loader:
                    tensors, masks = tensors.to(device), masks.to(device)
                    with autocast(device_type=device.type):
                        logits = model(tensors)
                    val_miou += compute_batch_miou(logits, masks, num_classes)
                    batches += 1
            
            score = val_miou / batches
            if score > best_miou: best_miou = score
            
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                
        return best_miou

    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{study_db_path}", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    
    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_value

# --- 6. Edge Deployment Engine (ONNX Export) ---
def export_to_onnx(model, weights_path, run_dir, device):
    print(f"\n--- Serializing Architecture to ONNX ---")
    
    # Ensure the model is completely isolated from training gradients
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Generate a synthetic 5-channel dummy tensor matching your Jetson camera feeds
    dummy_input = torch.randn(1, 5, 480, 640, device=device)
    export_dir = os.path.join(run_dir, "deployment")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "trimodal_seg_dynamic.onnx")

    print("[*] Tracing computational graph and folding constants...")
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=14,          # Opset 14 is highly stable for TensorRT 8.x
            do_constant_folding=True,  # Fuses BatchNorm layers for faster inference
            input_names=['input_rgbdt'],
            output_names=['output_mask'],
            dynamic_axes={
                'input_rgbdt': {0: 'batch_size'}, 
                'output_mask': {0: 'batch_size'}
            }
        )
        
    print(f"[SUCCESS] ONNX graph serialized to: {onnx_path}")
    return onnx_path

# --- 7. Main Execution Engine ---
def main():
    parser = argparse.ArgumentParser(description="TriModal R&D State-Machine Pipeline")
    parser.add_argument("--model", type=str, default="TriModalYOLOSeg")
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8

    train_dataset = TriModalSegDataset(csv_file="dataset/MM5/train_dataset.csv", split="train")
    eval_dataset = TriModalSegDataset(csv_file="dataset/MM5/eval_dataset.csv", split="eval")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    NUM_CLASSES = train_dataset.num_classes

    try:
        ModelClass = getattr(models, args.model)
    except AttributeError:
        raise AttributeError(f"[!] Architecture '{args.model}' not found in models.py.")

    model_kwargs = {"in_channels": 5, "num_classes": NUM_CLASSES}
    model_kwargs.update(json.loads(args.params))

    model = ModelClass(**model_kwargs).to(DEVICE)
    manager = ExperimentManager(model_instance=model)
    state = manager.detect_state()
    
    # Catch both a single None and the (None, None) tuple
    if state is None or state == (None, None): 
        return 
        
    run_dir = state["run_dir"]
    phase = state["phase"]

    # --- HPO Branch ---
    if phase == "hpo":
        print(f"\n🚀 INITIALIZING HPO SWEEP FOR {model.__class__.__name__}")
        best_score = run_hpo_phase(run_dir, state["inherit_weights"], ModelClass, model_kwargs, train_loader, eval_loader, NUM_CLASSES, DEVICE)
        
        results_payload = {
            "model_architecture": model.__class__.__name__,
            "phase": phase,
            "completed_at": datetime.datetime.now().isoformat(),
            "best_hpo_mIoU": best_score
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results_payload, f, indent=4)
        print("\n[SUCCESS] HPO Complete. Run `python train.py` to auto-start the Hero Phase.")
        return

    # --- Export / Serialization Branch ---
    if phase == "export":
        print(f"\n🚀 INITIALIZING DEPLOYMENT EXPORT FOR {model.__class__.__name__}")
        
        # The manager automatically passes the best weights from the Microtune phase here
        best_weights_path = state["inherit_weights"] 
        
        onnx_file = export_to_onnx(model, best_weights_path, run_dir, DEVICE)
        
        results_payload = {
            "model_architecture": model.__class__.__name__,
            "phase": phase,
            "completed_at": datetime.datetime.now().isoformat(),
            "deployment_artifact": onnx_file,
            "status": "Ready for Jetson TensorRT Compilation"
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results_payload, f, indent=4)
            
        print("\n[SUCCESS] Pipeline Complete. The model is ready for hardware deployment.")
        return

    # --- Standard Training Branch ---
    MAX_EPOCHS = 150 if phase == "baseline" else (300 if phase == "hero" else 200)
    PATIENCE = 25 if phase == "baseline" else 40

    print("\n" + "="*75)
    print(f"🚀 INITIALIZING RUN: {os.path.basename(run_dir)}")
    print(f"🧠 ARCHITECTURE: {model.__class__.__name__}")
    print(f"📊 PHASE: {phase.upper()} | EPOCHS: {MAX_EPOCHS} | PATIENCE: {PATIENCE}")
    print("="*75 + "\n")

    optimizer, scheduler, criterion = build_phase_config(phase, model, MAX_EPOCHS, manager.model_dir, NUM_CLASSES)
    scaler = GradScaler(DEVICE.type)

    if state["is_resume"]:
        checkpoint = torch.load(os.path.join(run_dir, "latest_checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        print(f"[*] Resuming interrupted {phase} run from Epoch {state['start_epoch']}")
        
    elif state["inherit_weights"] is not None:
        model.load_state_dict(torch.load(state["inherit_weights"]))
        print(f"[*] Curriculum Learning: Inheriting weights from previous phase.")

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    for epoch in range(state["start_epoch"], MAX_EPOCHS):
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
        val_loss, val_miou_accum, batches = 0.0, 0.0, 0
        
        with torch.no_grad():
            for tensors, masks in eval_loader:
                tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    logits = model(tensors)
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                val_miou_accum += compute_batch_miou(logits, masks, NUM_CLASSES, ignore_index=0)
                batches += 1
                
        avg_val_loss = val_loss / batches
        avg_val_miou = val_miou_accum / batches
        
        # --- CRITICAL TENSORBOARD FIX ---
        writer.add_scalars("Loss", {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Validation_mIoU", avg_val_miou, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        writer.flush() # Forces immediate update to the live local web server
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}")

        state["start_epoch"] = epoch + 1
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
        }, os.path.join(run_dir, "latest_checkpoint.pt"))

        if avg_val_miou > state["best_miou"]:
            state["best_miou"] = avg_val_miou
            state["patience_counter"] = 0
            best_weights = model.state_dict()
            torch.save(best_weights, os.path.join(run_dir, "best_model.pt"))
            torch.save(best_weights, os.path.join(run_dir, "weights", f"epoch_{epoch+1}_mIoU_{avg_val_miou:.3f}.pt"))
        else:
            state["patience_counter"] += 1

        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state, f, indent=4)

        if state["patience_counter"] >= PATIENCE:
            print(f"\n[!] Early Stopping triggered. Phase complete.")
            break

    writer.close()

    print("\n--- Generating Final Diagnostic & Grad-CAM Report ---")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    model.eval()
    
    grad_cam = SemanticGradCAM(model)
    final_val_miou, batches = 0.0, 0
    
    for i, (tensors, masks) in enumerate(tqdm(eval_loader, desc="Final Test Pass")):
        tensors, masks = tensors.to(DEVICE), masks.to(DEVICE)
        
        with autocast(device_type=DEVICE.type):
            logits = model(tensors)
            
        final_val_miou += compute_batch_miou(logits, masks, NUM_CLASSES, ignore_index=0)
        batches += 1
        
        if i < 5:
            predictions = torch.argmax(logits, dim=1)
            for b in range(tensors.size(0)):
                unique_classes = torch.unique(predictions[b])
                for cls in unique_classes:
                    if cls == 0: continue
                    input_tensor = tensors[b].unsqueeze(0)
                    heatmap = grad_cam.generate_heatmap(input_tensor, cls.item())
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    explain_path = os.path.join(run_dir, "explainability", f"batch{i}_img{b}_class{cls.item()}_cam.png")
                    cv2.imwrite(explain_path, heatmap_colored)

    final_score = final_val_miou / batches
    
    results_payload = {
        "model_architecture": model.__class__.__name__,
        "initialization_params": model_kwargs,
        "phase": phase,
        "completed_at": datetime.datetime.now().isoformat(),
        "final_test_mIoU": final_score,
        "best_validation_mIoU": state["best_miou"],
        "epochs_trained": state["start_epoch"]
    }
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Phase {phase.upper()} recorded with final mIoU: {final_score:.4f}")
    print("Run `python train.py` again to automatically begin the next phase.\n")

if __name__ == '__main__':
    main()