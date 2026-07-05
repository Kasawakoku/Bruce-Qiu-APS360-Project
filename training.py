# Include packages

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import ( 
    efficientnet_b3, EfficientNet_B3_Weights, 
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)
# Other packages
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import natsort
from natsort import natsorted
import random
import os
from sklearn.model_selection import train_test_split


# Neural Net Modules

# Baseline Models for Single Task Classification (Variant Classification)

class BaselineEfficientNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineEfficientNet, self).__init__()
        # Load the off-the-shelf backbone
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Replace the final classifier head for the single task
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_variant_classes)
        )

    def forward(self, x):
        return self.model(x)


class BaselineResNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineResNet, self).__init__()
        # 1. Load the model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. Extract the input features of the final layer
        # Note: For ResNet, the final layer is called 'fc', not 'classifier'
        in_features = self.model.fc.in_features 
        
        # 3. Replace the final layer
        self.model.fc = nn.Linear(in_features, num_variant_classes)

    def forward(self, x):
        return self.model(x)


class BaselineMobileNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineMobileNet, self).__init__()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        # MobileNet uses a 'classifier' sequential block like EfficientNet
        # The linear layer is at index 3
        in_features = self.model.classifier[3].in_features
        
        self.model.classifier[3] = nn.Linear(in_features, num_variant_classes)

    def forward(self, x):
        return self.model(x)
    
class BaselineConvNeXt(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineConvNeXt, self).__init__()
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # ConvNeXt uses a 'classifier' block, the linear layer is at index 2
        in_features = self.model.classifier[2].in_features
        
        self.model.classifier[2] = nn.Linear(in_features, num_variant_classes)

    def forward(self, x):
        return self.model(x)
    
# Primary Model for Multi-Task Classification (Variant + Airline Classification)

class DualBranchNet(nn.Module):
    def __init__(self, num_variant_classes, num_airline_classes):
        super(DualBranchNet, self).__init__()
        
        # 1. Shared Convolutional Backbone (Remove default classifier)
        base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = base_model.features
        
        # 1536 is the output channel size for EfficientNet-B3
        feature_dim = 1536 
        
        # 2. Structural Head (Global Average Pooling for Variant)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.structural_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_variant_classes)
        )
        
        # 3. Branding Head (Adaptive Max Pooling for Airline Livery)
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.branding_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_airline_classes)
        )

    def forward(self, x):
        # Shared feature extraction
        shared_features = self.features(x)
        
        # Branch 1: Variant Prediction
        v = self.gap(shared_features).view(shared_features.size(0), -1)
        variant_pred = self.structural_mlp(v)
        
        # Branch 2: Airline Prediction
        a = self.amp(shared_features).view(shared_features.size(0), -1)
        airline_pred = self.branding_mlp(a)
        
        return variant_pred, airline_pred







def get_model_name(name, batch_size, learning_rate, epoch):
  # Generate name for model containing all hyperparameters
  path = "/content/checkpoints/model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size,learning_rate, epoch)
  return path

def evaluate(net, loader, criterion, device):
  # Evaluate network on validation set
  total_loss = 0.0
  total_err = 0.0
  total_epoch = 0

  net.eval() # set to evaluation mode
  with torch.no_grad(): # reduce memory and runtime length
    for i, data in enumerate(loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device) # need to do this for multi-class

      outputs = net(inputs)
      loss = criterion(outputs, labels)

      # Find class with highest score
      _, predicted = torch.max(outputs.data, 1)
      corr = (predicted != labels)

      # Get total error, loss, epoch
      total_err += int(corr.sum())
      total_loss += loss.item()
      total_epoch += len(labels)


  err = float(total_err) / total_epoch
  loss = float(total_loss) / (i + 1)
  return err, loss

def plot_training_curve(path):
  # Plot training curve for a model run
  train_err = np.loadtxt("{}_train_err.csv".format(path))
  val_err = np.loadtxt("{}_val_err.csv".format(path))
  train_loss = np.loadtxt("{}_train_loss.csv".format(path))
  val_loss = np.loadtxt("{}_val_loss.csv".format(path))

  n = len(train_err) # num of epochs

  plt.figure(figsize=(10, 4))

  # Plot Error
  plt.subplot(1, 2, 1)
  plt.title("Train vs Validation Error")
  plt.plot(range(1,n+1), train_err, label="Train")
  plt.plot(range(1,n+1), val_err, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.legend(loc='best')

  # Plot Loss
  plt.subplot(1, 2, 2)
  plt.title("Train vs Validation Loss")
  plt.plot(range(1,n+1), train_loss, label="Train")
  plt.plot(range(1,n+1), val_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')

  plt.tight_layout()
  plt.show()

def train_net(net, train_loader, val_loader, batch_size=64, learning_rate=0.01, num_epochs=30, checkpoint_freq=5):
  # Trains multi-class PyTorch model

  # Setup Device (Use GPU if available)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on device: {device}")
  net.to(device)

  # Fixed PyTorch random seed for reproducible result
  torch.manual_seed(1000)

  # multi-class loss and Adam Optimizer
  # Use cross entropy loss for multi-class loss (explain...)
  # Use Adam for optimizer (explain...)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  # Arrays to store metrics
  train_err = np.zeros(num_epochs)
  train_loss = np.zeros(num_epochs)
  val_err = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)

  start_time = time.time()

  for epoch in range(num_epochs):
    # Training
    net.train() # Training mode
    total_train_loss = 0.0
    total_train_err = 0.0
    total_epoch = 0

    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device) # need to do this for multi-class

      # Zero
      optimizer.zero_grad()

      # Forward
      outputs = net(inputs)
      loss = criterion(outputs, labels)

      # Backward
      loss.backward()

      # Optimize
      optimizer.step()

      # Calculate error
      _, predicted = torch.max(outputs, 1)
      corr = (predicted != labels)

      total_train_err += int(corr.sum())
      total_train_loss += loss.item()
      total_epoch += len(labels)

    train_err[epoch] = float(total_train_err) / total_epoch
    train_loss[epoch] = float(total_train_loss) / (i+1)

    # Evaluate on validation set
    val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device)
    print(("Epoch {}: Train err: {}, Train loss: {} |"+
                "Validation err: {}, Validation loss: {}").format(
                    epoch + 1,
                    train_err[epoch],
                    train_loss[epoch],
                    val_err[epoch],
                    val_loss[epoch]))

  # Checkpointing
    if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
      model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
      torch.save(net.state_dict(), model_path)

  print('Finished Training')
  end_time = time.time()
  print("Total time elapsed: {:.2f} seconds".format(end_time - start_time))

  # Write the train/test loss/err into CSV file for plotting later
  model_base_path = get_model_name(net.name, batch_size, learning_rate, "final")
  np.savetxt("{}_train_err.csv".format(model_base_path), train_err)
  np.savetxt("{}_train_loss.csv".format(model_base_path), train_loss)
  np.savetxt("{}_val_err.csv".format(model_base_path), val_err)
  np.savetxt("{}_val_loss.csv".format(model_base_path), val_loss)

  return model_base_path