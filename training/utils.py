# Helper IO, plotting, and inference

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def get_model_name(name, batch_size, learning_rate, epoch, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = f"model_{name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}.pt"
    return os.path.join(checkpoint_dir, filename)

def load_model_checkpoint(checkpoint_path, model, optimizer=None, device="cpu", multi_task_loss=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load the learned uncertainty weights if provided and available
    if multi_task_loss is not None and "loss_state_dict" in checkpoint and checkpoint["loss_state_dict"] is not None:
        multi_task_loss.load_state_dict(checkpoint["loss_state_dict"])
        print("Successfully loaded MultiTaskLoss uncertainty weights.")
        
    start_epoch = checkpoint.get("epoch", 0)
    loaded_history = checkpoint.get("history", None)
    
    print(f"Successfully loaded checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Resuming training from epoch {start_epoch}...")
    
    # We don't need to return multi_task_loss because it is updated in-place
    return model, optimizer, start_epoch, loaded_history

def plot_training_curve(path, is_multitask=True, target_task="variant", save_path_prefix=None, max_epoch=None):
    """
    Plot training curves for F1-Scores and Loss.
    Reads from CSVs and saves the output as PNG files.
    """
    # 1. Load Loss (Always present)
    train_loss = np.atleast_1d(np.loadtxt(f"{path}_train_loss.csv"))
    val_loss = np.atleast_1d(np.loadtxt(f"{path}_val_loss.csv"))
    
    if max_epoch is not None:
        train_loss = train_loss[:max_epoch]
        val_loss = val_loss[:max_epoch]
        
    n_epochs = len(train_loss)

    # 2. Load F1 Scores based on task
    if is_multitask:
        train_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_var_f1.csv"))[:max_epoch]
        val_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_var_f1.csv"))[:max_epoch]
        train_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_air_f1.csv"))[:max_epoch]
        val_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_air_f1.csv"))[:max_epoch]
    else:
        if target_task == "airline":
            train_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_air_f1.csv"))[:max_epoch]
            val_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_air_f1.csv"))[:max_epoch]
            f1_title = "Airline Weighted F1-Score"
        else:
            train_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_var_f1.csv"))[:max_epoch]
            val_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_var_f1.csv"))[:max_epoch]
            f1_title = "Variant Weighted F1-Score"

    # 3. Load iteration metrics if they exist
    try:
        iter_steps = np.atleast_1d(np.loadtxt(f"{path}_iter_steps.csv"))
        iter_train_loss = np.atleast_1d(np.loadtxt(f"{path}_iter_train_loss.csv"))
        iter_val_loss = np.atleast_1d(np.loadtxt(f"{path}_iter_val_loss.csv"))
        
        if is_multitask:
            iter_train_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_var_f1.csv"))
            iter_val_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_var_f1.csv"))
            iter_train_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_air_f1.csv"))
            iter_val_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_air_f1.csv"))
        else:
            if target_task == "airline":
                iter_train_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_air_f1.csv"))
                iter_val_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_air_f1.csv"))
            else:
                iter_train_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_var_f1.csv"))
                iter_val_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_var_f1.csv"))
        has_iters = True
    except OSError:
        has_iters = False

    # --- Plot 1: Epoch-Level Metrics ---
    plt.figure(figsize=(15 if is_multitask else 10, 4))
    plt.suptitle("Epoch-Level Metrics (Train vs Validation)", fontsize=14, y=1.05)

    if is_multitask:
        plt.subplot(1, 3, 1)
        plt.title("Variant Weighted F1-Score")
        plt.plot(range(1, n_epochs+1), train_var_f1, label="Train")
        plt.plot(range(1, n_epochs+1), val_var_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.legend(loc='best')
        plt.locator_params(axis='x', integer=True)

        plt.subplot(1, 3, 2)
        plt.title("Airline Weighted F1-Score")
        plt.plot(range(1, n_epochs+1), train_air_f1, label="Train")
        plt.plot(range(1, n_epochs+1), val_air_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.legend(loc='best')
        plt.locator_params(axis='x', integer=True)
        
        plt.subplot(1, 3, 3)
    else:
        plt.subplot(1, 2, 1)
        plt.title(f1_title)
        plt.plot(range(1, n_epochs+1), train_f1, label="Train")
        plt.plot(range(1, n_epochs+1), val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.legend(loc='best')
        plt.locator_params(axis='x', integer=True)
        
        plt.subplot(1, 2, 2)

    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n_epochs+1), train_loss, label="Train")
    plt.plot(range(1, n_epochs+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.locator_params(axis='x', integer=True)

    plt.tight_layout()
    
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_epoch_metrics.png", bbox_inches='tight')
        print(f"Saved epoch metrics graph to: {save_path_prefix}_epoch_metrics.png")
        plt.close()
    else:
        plt.show()

    # --- Plot 2: Iteration-Level Metrics ---
    if has_iters and len(iter_steps) > 0:
        plt.figure(figsize=(15 if is_multitask else 10, 4))
        plt.suptitle("Iteration-Level Metrics (Train vs Validation)", fontsize=14, y=1.05)

        if is_multitask:
            plt.subplot(1, 3, 1)
            plt.title("Variant Weighted F1-Score")
            plt.plot(iter_steps, iter_train_var_f1, label="Train", color='blue')
            plt.plot(iter_steps, iter_val_var_f1, label="Validation", color='red')
            plt.xlabel("Iteration (Batches)")
            plt.ylabel("F1-Score")
            plt.legend(loc='best')

            plt.subplot(1, 3, 2)
            plt.title("Airline Weighted F1-Score")
            plt.plot(iter_steps, iter_train_air_f1, label="Train", color='blue')
            plt.plot(iter_steps, iter_val_air_f1, label="Validation", color='red')
            plt.xlabel("Iteration (Batches)")
            plt.ylabel("F1-Score")
            plt.legend(loc='best')
            
            plt.subplot(1, 3, 3)
        else:
            plt.subplot(1, 2, 1)
            plt.title(f1_title)
            plt.plot(iter_steps, iter_train_f1, label="Train", color='blue')
            plt.plot(iter_steps, iter_val_f1, label="Validation", color='red')
            plt.xlabel("Iteration (Batches)")
            plt.ylabel("F1-Score")
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)

        plt.title("Train vs Validation Loss")
        plt.plot(iter_steps, iter_train_loss, label="Train Loss", color='blue')
        plt.plot(iter_steps, iter_val_loss, label="Val Loss", color='red')
        plt.xlabel("Iteration (Batches)")
        plt.ylabel("Loss")
        plt.legend(loc='best')

        plt.tight_layout()
        
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_iter_metrics.png", bbox_inches='tight')
            print(f"Saved iteration metrics graph to: {save_path_prefix}_iter_metrics.png")
            plt.close()
        else:
            plt.show()

def pad_to_square(img, fill_color=(255, 255, 255)):
    w, h = img.size
    if w == h: return img
    size = max(w, h)
    new_img = Image.new('RGB', (size, size), fill_color)
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_img

def predict_image(image_path, model, device, variant_idx_to_name, airline_idx_to_name=None, is_multitask=True, image_size=224):
    model.eval()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    img = pad_to_square(img)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_multitask:
            var_outputs, air_outputs = model(img_tensor)
            var_idx = torch.argmax(var_outputs, dim=1).item()
            air_idx = torch.argmax(air_outputs, dim=1).item()
            var_pred = variant_idx_to_name[var_idx]
            air_pred = airline_idx_to_name[air_idx]
            return var_pred, air_pred
        else:
            var_outputs = model(img_tensor)
            var_idx = torch.argmax(var_outputs, dim=1).item()
            var_pred = variant_idx_to_name[var_idx]
            return var_pred