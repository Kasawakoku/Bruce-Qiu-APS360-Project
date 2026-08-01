# Core training and evaluation loops

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import get_model_name

def evaluate(net, loader, criterion, device, is_multitask=True):
    total_loss = 0.0
    all_var_preds = []
    all_var_labels = []
    if is_multitask:
        all_air_preds = []
        all_air_labels = []
        
    net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, variant_labels, airline_labels = data
            inputs = inputs.to(device)
            variant_labels = variant_labels.to(device)
            airline_labels = airline_labels.to(device)

            if is_multitask:
                var_outputs, air_outputs = net(inputs)
                loss = criterion(var_outputs, variant_labels) + criterion(air_outputs, airline_labels)
                _, var_preds = torch.max(var_outputs.data, 1)
                _, air_preds = torch.max(air_outputs.data, 1)
                all_air_preds.extend(air_preds.cpu().numpy())
                all_air_labels.extend(airline_labels.cpu().numpy())
            else:
                var_outputs = net(inputs)
                loss = criterion(var_outputs, variant_labels)
                _, var_preds = torch.max(var_outputs.data, 1)

            total_loss += loss.item()
            all_var_preds.extend(var_preds.cpu().numpy())
            all_var_labels.extend(variant_labels.cpu().numpy())

    var_f1 = f1_score(all_var_labels, all_var_preds, average='weighted', zero_division=0)
    avg_loss = float(total_loss) / len(loader)

    if is_multitask:
        air_f1 = f1_score(all_air_labels, all_air_preds, average='weighted', zero_division=0)
        return var_f1, air_f1, avg_loss
    else:
        return var_f1, avg_loss

def train_net(net, train_loader, val_loader, batch_size=64, learning_rate=0.01, num_epochs=30, checkpoint_freq=1, 
              is_multitask=True, checkpoint_dir="checkpoints", optimizer=None, start_epoch=0, 
              track_iterations=True, record_freq=100, loaded_history=None, custom_model_name=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    net.to(device)
    torch.manual_seed(1000)

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_var_f1, train_loss = np.zeros(num_epochs), np.zeros(num_epochs)
    val_var_f1, val_loss = np.zeros(num_epochs), np.zeros(num_epochs)

    if is_multitask:
        train_air_f1 = np.zeros(num_epochs)
        val_air_f1 = np.zeros(num_epochs)

    global_step = 0
    iter_steps = []
    iter_train_loss, iter_val_loss = [], []
    iter_train_var_f1, iter_val_var_f1 = [], []
    if is_multitask:
        iter_train_air_f1, iter_val_air_f1 = [], []

    if loaded_history is not None:
        print("Restoring training history from checkpoint for seamless graphs...")
        limit = min(start_epoch, len(loaded_history.get('train_loss', [])))
        train_var_f1[:limit] = loaded_history['train_var_f1'][:limit]
        val_var_f1[:limit] = loaded_history['val_var_f1'][:limit]
        train_loss[:limit] = loaded_history['train_loss'][:limit]
        val_loss[:limit] = loaded_history['val_loss'][:limit]
        if is_multitask and loaded_history.get('train_air_f1') is not None:
            train_air_f1[:limit] = loaded_history['train_air_f1'][:limit]
            val_air_f1[:limit] = loaded_history['val_air_f1'][:limit]
            
        if track_iterations:
            global_step = loaded_history.get('global_step', 0)
            iter_steps = loaded_history.get('iter_steps', [])
            iter_train_loss = loaded_history.get('iter_train_loss', [])
            iter_train_var_f1 = loaded_history.get('iter_train_var_f1', [])
            iter_val_loss = loaded_history.get('iter_val_loss', [])
            iter_val_var_f1 = loaded_history.get('iter_val_var_f1', [])
            if is_multitask:
                iter_train_air_f1 = loaded_history.get('iter_train_air_f1', [])
                iter_val_air_f1 = loaded_history.get('iter_val_air_f1', [])

    start_time = time.time()
    print("Start training...")

    for epoch in range(start_epoch, num_epochs): 
        net.train()
        total_train_loss = 0.0
        all_var_preds, all_var_labels = [], []
        if is_multitask:
            all_air_preds, all_air_labels = [], []

        window_loss = 0.0
        window_var_preds, window_var_labels = [], []
        if is_multitask:
            window_air_preds, window_air_labels = [], []

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, data in enumerate(progress):
            if track_iterations: global_step += 1
            
            inputs, variant_labels, airline_labels = data
            inputs, variant_labels, airline_labels = inputs.to(device), variant_labels.to(device), airline_labels.to(device)

            optimizer.zero_grad()

            if is_multitask:
                var_outputs, air_outputs = net(inputs)
                loss = criterion(var_outputs, variant_labels) + criterion(air_outputs, airline_labels)
                _, var_preds = torch.max(var_outputs.data, 1)
                _, air_preds = torch.max(air_outputs.data, 1)
                all_air_preds.extend(air_preds.cpu().numpy())
                all_air_labels.extend(airline_labels.cpu().numpy())
            else:
                var_outputs = net(inputs)
                loss = criterion(var_outputs, variant_labels)
                _, var_preds = torch.max(var_outputs.data, 1)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            all_var_preds.extend(var_preds.cpu().numpy())
            all_var_labels.extend(variant_labels.cpu().numpy())

            if track_iterations:
                window_loss += loss.item()
                window_var_preds.extend(var_preds.cpu().numpy())
                window_var_labels.extend(variant_labels.cpu().numpy())
                if is_multitask:
                    window_air_preds.extend(air_preds.cpu().numpy())
                    window_air_labels.extend(airline_labels.cpu().numpy())

                if global_step % record_freq == 0 or (i + 1) == len(train_loader):
                    iter_steps.append(global_step)
                    iter_train_loss.append(window_loss / ((global_step % record_freq) or record_freq))
                    iter_train_var_f1.append(f1_score(window_var_labels, window_var_preds, average='weighted', zero_division=0))
                    
                    if is_multitask:
                        iter_train_air_f1.append(f1_score(window_air_labels, window_air_preds, average='weighted', zero_division=0))
                        v_var, v_air, v_loss = evaluate(net, val_loader, criterion, device, is_multitask)
                        iter_val_var_f1.append(v_var); iter_val_air_f1.append(v_air); iter_val_loss.append(v_loss)
                    else:
                        v_var, v_loss = evaluate(net, val_loader, criterion, device, is_multitask)
                        iter_val_var_f1.append(v_var); iter_val_loss.append(v_loss)
                        
                    net.train()
                    window_loss = 0.0
                    window_var_preds, window_var_labels = [], []
                    if is_multitask: window_air_preds, window_air_labels = [], []

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_var_f1[epoch] = f1_score(all_var_labels, all_var_preds, average='weighted', zero_division=0)
        train_loss[epoch] = float(total_train_loss) / len(train_loader)
        if is_multitask:
            train_air_f1[epoch] = f1_score(all_air_labels, all_air_preds, average='weighted', zero_division=0)

        if is_multitask:
            val_var_f1[epoch], val_air_f1[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device, is_multitask)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f} | Train Var F1: {train_var_f1[epoch]:.4f} | Train Air F1: {train_air_f1[epoch]:.4f}")
            print(f"          Val Loss: {val_loss[epoch]:.4f} | Val Var F1: {val_var_f1[epoch]:.4f} | Val Air F1: {val_air_f1[epoch]:.4f}")
        else:
            val_var_f1[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device, is_multitask)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f} | Train Var F1: {train_var_f1[epoch]:.4f}")
            print(f"          Val Loss: {val_loss[epoch]:.4f} | Val Var F1: {val_var_f1[epoch]:.4f}")

        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            model_name = custom_model_name if custom_model_name else getattr(net, 'name', net.__class__.__name__)
            model_path = get_model_name(model_name, batch_size, learning_rate, epoch + 1, checkpoint_dir)
            
            history_dict = {
                "train_var_f1": train_var_f1, "val_var_f1": val_var_f1,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_air_f1": train_air_f1 if is_multitask else None,
                "val_air_f1": val_air_f1 if is_multitask else None,
            }
            if track_iterations:
                history_dict.update({
                    "iter_steps": iter_steps, 
                    "iter_train_loss": iter_train_loss, "iter_val_loss": iter_val_loss,
                    "iter_train_var_f1": iter_train_var_f1, "iter_val_var_f1": iter_val_var_f1,
                    "iter_train_air_f1": iter_train_air_f1 if is_multitask else None,
                    "iter_val_air_f1": iter_val_air_f1 if is_multitask else None,
                    "global_step": global_step
                })
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history_dict,

                "metadata": {
                    "model_name": model_name,
                    "architecture_class": net.__class__.__name__,
                    "is_multitask": is_multitask,
                    "hyperparameters": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs_total": num_epochs,
                        "optimizer": optimizer.__class__.__name__
                    }
                }
            }, model_path)

    print('Finished Training')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))

    model_name = getattr(net, 'name', 'model')
    model_base_path = get_model_name(model_name, batch_size, learning_rate, "final", checkpoint_dir).replace('.pt', '')
    
    np.savetxt(f"{model_base_path}_train_var_f1.csv", train_var_f1)
    np.savetxt(f"{model_base_path}_val_var_f1.csv", val_var_f1)
    np.savetxt(f"{model_base_path}_train_loss.csv", train_loss)
    np.savetxt(f"{model_base_path}_val_loss.csv", val_loss)
    if is_multitask:
        np.savetxt(f"{model_base_path}_train_air_f1.csv", train_air_f1)
        np.savetxt(f"{model_base_path}_val_air_f1.csv", val_air_f1)
    
    if track_iterations and len(iter_steps) > 0:
        np.savetxt(f"{model_base_path}_iter_steps.csv", iter_steps)
        np.savetxt(f"{model_base_path}_iter_train_loss.csv", iter_train_loss)
        np.savetxt(f"{model_base_path}_iter_val_loss.csv", iter_val_loss)
        np.savetxt(f"{model_base_path}_iter_train_var_f1.csv", iter_train_var_f1)
        np.savetxt(f"{model_base_path}_iter_val_var_f1.csv", iter_val_var_f1)
        if is_multitask:
            np.savetxt(f"{model_base_path}_iter_train_air_f1.csv", iter_train_air_f1)
            np.savetxt(f"{model_base_path}_iter_val_air_f1.csv", iter_val_air_f1)

    return model_base_path