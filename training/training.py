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
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import natsort
from natsort import natsorted
import random
import os
from sklearn.model_selection import train_test_split



# Data Loading

'''
def split_intersection_dataset(
    csv_path, 
    val_size=0.10, 
    test_size=0.10, 
    random_state=42, 
    min_stratify_count=2
):
    """
    Splits the intersection dataset into train, val, and test sets using
    stratification on a combined (airline + variant) key.
    
    Parameters:
    -----------
    csv_path : str
        Path to the intersection metadata CSV file.
    val_size : float
        Proportion of the dataset to include in the validation split.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Controls the shuffling applied to the data before the split.
    min_stratify_count : int
        Minimum occurrences of a combined class to be stratified. Pairs 
        with counts below this will be grouped together into a fallback class.
    """
    # 1. Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # 2. Create a composite key for stratification
    # We use both airline and variant to capture the multi-task nature
    df['stratify_key'] = df['airline'].astype(str) + "_" + df['aircraft_variant'].astype(str)
    
    # 3. Handle rare combinations that would break stratification
    counts = df['stratify_key'].value_counts()
    rare_classes = counts[counts < min_stratify_count].index
    
    # Group rare combinations into a single fallback category for splitting purposes
    df['split_key'] = df['stratify_key'].apply(lambda x: 'rare_combination' if x in rare_classes else x)
    
    # 4. First Split: Isolate the Test Set
    # To get a true test_size proportion of the total, adjust the remainder budget
    remaining_size = 1.0 - test_size
    relative_val_size = val_size / remaining_size
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['split_key'],
        random_state=random_state
    )
    
    # 5. Second Split: Separate Train and Validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df['split_key'],
        random_state=random_state
    )
    
    # Clean up the temporary helper columns before saving
    for target_df in [train_df, val_df, test_df]:
        target_df.drop(columns=['stratify_key', 'split_key'], inplace=True, errors='ignore')
        
    print("\n--- Split Results ---")
    print(f"Train set size:      {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set size:       {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

# Example Usage (uncomment to run when ready):
# train_df, val_df, test_df = split_intersection_dataset(
#     csv_path='path_to_your_intersection.csv', 
#     val_size=0.10, 
#     test_size=0.10
# )
# 
# # Save the splits to new CSVs which your PyTorch Dataset can read directly
# train_df.to_csv('train_metadata.csv', index=False)
# val_df.to_csv('val_metadata.csv', index=False)
# test_df.to_csv('test_metadata.csv', index=False)
'''
'''
def generate_splits_with_union(
    master_csv_path, 
    intersection_csv_path, 
    valid_airlines_path, 
    valid_variants_path,
    val_size=0.10, 
    test_size=0.10, 
    random_state=42
):
    """
    Splits the clean intersection data for pristine validation/testing,
    then combines the training remainder with any valid single-task images
    from the messy union dataset, mapping missing labels to -100.
    """
    # 1. Load your locked-down trimmed classes to determine what is "valid"
    valid_airlines_df = pd.read_csv(valid_airlines_path)
    valid_variants_df = pd.read_csv(valid_variants_path)
    
    # Create sets for O(1) lookups
    valid_airlines = set(valid_airlines_df['Airline'].dropna().unique())
    valid_variants = set(valid_variants_df['Variant'].dropna().unique())
    
    # 2. Load Dataframes
    df_master = pd.read_csv(master_csv_path)
    df_inter = pd.read_csv(intersection_csv_path)
    
    # 3. Stratify and split the clean intersection set first
    # Create a temporary joint column just to ensure balanced splits
    df_inter['split_key'] = df_inter['airline'].astype(str) + "_" + df_inter['aircraft_variant'].astype(str)
    
    # Smooth rare combinations that appear only once in the intersection set
    counts = df_inter['split_key'].value_counts()
    rare_classes = counts[counts < 2].index
    df_inter['split_key'] = df_inter['split_key'].apply(lambda x: 'rare' if x in rare_classes else x)
    
    # Calculate relative validation size for the second split
    remaining_size = 1.0 - test_size
    relative_val_size = val_size / remaining_size
    
    # Perform splits
    inter_train_val, test_df = train_test_split(
        df_inter, test_size=test_size, stratify=df_inter['split_key'], random_state=random_state
    )
    inter_train_df, val_df = train_test_split(
        inter_train_val, test_size=relative_val_size, stratify=inter_train_val['split_key'], random_state=random_state
    )
    
    # 4. Isolate the "Union-Only" entries from the master dataframe
    # We want images that are NOT in the intersection set, but possess at least ONE valid label
    allocated_photo_ids = set(df_inter['photo_id'])
    df_leftovers = df_master[~df_master['photo_id'].isin(allocated_photo_ids)].copy()
    
    # Check validity against your trimmed cutoff criteria
    df_leftovers['has_valid_airline'] = df_leftovers['airline'].isin(valid_airlines)
    df_leftovers['has_valid_variant'] = df_leftovers['aircraft_variant'].isin(valid_variants)
    
    # Keep only rows that have at least one useful valid attribute
    union_train_reinforcements = df_leftovers[df_leftovers['has_valid_airline'] | df_leftovers['has_valid_variant']].copy()
    
    # Mask invalid/dropped categorical values right in the dataframe so your Dataset handles them seamlessly
    union_train_reinforcements.loc[~union_train_reinforcements['has_valid_airline'], 'airline'] = '-100'
    union_train_reinforcements.loc[~union_train_reinforcements['has_valid_variant'], 'aircraft_variant'] = '-100'
    
    # 5. Merge the clean training slice with the masked union leftovers
    # Keep only the essential columns to match schemas
    keep_cols = ['photo_id', 'airline', 'aircraft_variant', 'image_filename']
    
    final_train_df = pd.concat([
        inter_train_df[keep_cols], 
        union_train_reinforcements[keep_cols]
    ], ignore_index=True)
    
    # Clean up evaluation dataframes
    final_val_df = val_df[keep_cols].copy()
    final_test_df = test_df[keep_cols].copy()
    
    print("--- Multi-Task Split Summary ---")
    print(f"Total Master Scraped Images:    {len(df_master)}")
    print(f"Clean Intersection Sub-budget:  {len(df_inter)}")
    print(f"Pristine Validation Set Size:   {len(final_val_df)}")
    print(f"Pristine Test Set Size:         {len(final_test_df)}")
    print(f"Final Augmented Training Size:  {len(final_train_df)}")
    print(f" -> From clean pairs:           {len(inter_train_df)}")
    print(f" -> From single-task union:     {len(union_train_reinforcements)}")
    
    return final_train_df, final_val_df, final_test_df

# To use this when ready, execute:
# train_df, val_df, test_df = generate_splits_with_union(
#     master_csv_path='master_metadata.csv',
#     intersection_csv_path='intersection_metadata.csv',
#     valid_airlines_path='valid_airlines.csv',
#     valid_variants_path='valid_variants.csv'
# )
'''




def generate_simplified_splits(
    intersection_csv_path, 
    union_only_csv_path, # Your "union-intersection" CSV
    valid_airlines_path, 
    valid_variants_path,
    val_size=0.10, 
    test_size=0.10, 
    random_state=42
):
    # 1. Load your pre-processed datasets and valid class lists
    df_inter = pd.read_csv(intersection_csv_path)
    df_union_only = pd.read_csv(union_only_csv_path)
    
    valid_airlines = set(pd.read_csv(valid_airlines_path)['Airline'].dropna().unique())
    valid_variants = set(pd.read_csv(valid_variants_path)['Variant'].dropna().unique())
    
    # 2. FAILSAFE LOGIC: Smooth rare combinations in the intersection set
    # This prevents scikit-learn from crashing on singletons (like your Boeing 737-400)
    df_inter['split_key'] = df_inter['airline'].astype(str) + "_" + df_inter['aircraft_variant'].astype(str)
    counts = df_inter['split_key'].value_counts()
    rare_classes = counts[counts < 2].index
    df_inter['split_key'] = df_inter['split_key'].apply(lambda x: 'rare' if x in rare_classes else x)
    
    # Calculate relative validation size for the second split
    remaining_size = 1.0 - test_size
    relative_val_size = val_size / remaining_size
    
    # 3. Perform splits on the pristine intersection data
    inter_train_val, test_df = train_test_split(
        df_inter, test_size=test_size, stratify=df_inter['split_key'], random_state=random_state
    )
    inter_train_df, val_df = train_test_split(
        inter_train_val, test_size=relative_val_size, stratify=inter_train_val['split_key'], random_state=random_state
    )
    
    # 4. Apply the ignore_index mask to the union-only data
    # If an airline or variant isn't in your locked valid lists, map it to '-100'
    df_union_only.loc[~df_union_only['airline'].isin(valid_airlines), 'airline'] = '-100'
    df_union_only.loc[~df_union_only['aircraft_variant'].isin(valid_variants), 'aircraft_variant'] = '-100'
    
    # 5. Merge the clean training slice with the masked union leftovers
    keep_cols = ['photo_id', 'airline', 'aircraft_variant', 'image_filename']
    
    final_train_df = pd.concat([
        inter_train_df[keep_cols], 
        df_union_only[keep_cols]
    ], ignore_index=True)
    
    final_val_df = val_df[keep_cols].copy()
    final_test_df = test_df[keep_cols].copy()
    
    print("--- Simplified Multi-Task Split Summary ---")
    print(f"Pristine Validation Set Size:   {len(final_val_df)}")
    print(f"Pristine Test Set Size:         {len(final_test_df)}")
    print(f"Final Augmented Training Size:  {len(final_train_df)}")
    print(f" -> From clean pairs:           {len(inter_train_df)}")
    print(f" -> From single-task union:     {len(df_union_only)}")
    
    return final_train_df, final_val_df, final_test_df

# Example Execution:
# train_df, val_df, test_df = generate_simplified_splits(
#     'intersection.csv', 'union_intersection.csv', 'valid_airlines.csv', 'valid_variants.csv'
# )



def generate_parameterized_splits(
    intersection_csv_path, 
    union_only_csv_path, 
    valid_airlines_path, 
    valid_variants_path,
    val_pct=0.10, 
    test_pct=0.10, 
    min_count=5,          # Parameterized threshold for the rare group
    random_state=42
):
    print("Loading data...")
    df_inter = pd.read_csv(intersection_csv_path)
    df_union_only = pd.read_csv(union_only_csv_path)
    
    valid_airlines = set(pd.read_csv(valid_airlines_path)['Airline'].dropna().unique())
    valid_variants = set(pd.read_csv(valid_variants_path)['Variant'].dropna().unique())
    
    # 1. Create the joint class key
    df_inter['split_key'] = df_inter['airline'].astype(str) + "_" + df_inter['aircraft_variant'].astype(str)
    
    # 2. IDENTIFY AND GROUP RARE CLASSES
    counts = df_inter['split_key'].value_counts()
    rare_classes = counts[counts < min_count].index
    
    # Reassign the split_key to a unified bucket for anything below min_count
    df_inter['split_key'] = df_inter['split_key'].apply(
        lambda x: 'RARE_GROUP' if x in rare_classes else x
    )
    
    # 3. Custom Groupby Split Logic
    def split_group(group):
        n_total = len(group)
        
        # Calculate Val and Test sizes (rounding up)
        n_val = math.ceil(n_total * val_pct)
        n_test = math.ceil(n_total * test_pct)
        
        # Failsafe: Ensure at least 1 image remains for Training 
        # (in case a user sets min_count=2, meaning a group of 2 might get fully consumed)
        if n_val + n_test >= n_total and n_total >= 3:
            n_val = max(1, math.floor(n_total * val_pct))
            n_test = max(1, math.floor(n_total * test_pct))
            if n_val + n_test >= n_total:
                n_val, n_test = 1, 1
                
        # If the total group size is somehow 1 or 2 (e.g. all rare items combined only total 2)
        if n_total < 3:
            n_val, n_test = 0, 0
            
        # Shuffle rows to randomize selection
        shuffled = group.sample(frac=1, random_state=random_state).copy()
        
        # Assign splits
        shuffled['split_label'] = 'train'
        split_col_idx = shuffled.columns.get_loc('split_label')
        
        if n_val > 0:
            shuffled.iloc[0 : n_val, split_col_idx] = 'val'
        if n_test > 0:
            shuffled.iloc[n_val : n_val + n_test, split_col_idx] = 'test'
        
        return shuffled

    print("Splitting intersection dataset...")
    split_df = df_inter.groupby('split_key', group_keys=False).apply(split_group)
    
    # 4. Separate the carved splits
    inter_train_df = split_df[split_df['split_label'] == 'train'].copy()
    final_val_df = split_df[split_df['split_label'] == 'val'].copy()
    final_test_df = split_df[split_df['split_label'] == 'test'].copy()
    
    print("Masking union dataset labels...")
    # 5. Apply the ignore_index (-100) mask to the union-only data
    df_union_only.loc[~df_union_only['airline'].isin(valid_airlines), 'airline'] = '-100'
    df_union_only.loc[~df_union_only['aircraft_variant'].isin(valid_variants), 'aircraft_variant'] = '-100'
    
    # 6. Merge and Clean
    keep_cols = ['photo_id', 'airline', 'aircraft_variant', 'image_filename']
    
    final_train_df = pd.concat([
        inter_train_df[keep_cols], 
        df_union_only[keep_cols]
    ], ignore_index=True)
    
    final_val_df = final_val_df[keep_cols]
    final_test_df = final_test_df[keep_cols]
    
    print("\n--- Parameterized Multi-Task Split Summary ---")
    print(f"Pristine Validation Set Size:   {len(final_val_df)}")
    print(f"Pristine Test Set Size:         {len(final_test_df)}")
    print(f"Final Augmented Training Size:  {len(final_train_df)}")
    print(f" -> From clean pairs:           {len(inter_train_df)}")
    print(f" -> From masked union labels:   {len(df_union_only)}")
    
    return final_train_df, final_val_df, final_test_df




train_df, val_df, test_df = generate_parameterized_splits(
    '../data/metadata/airliners_metadata_trimmed_intersection.csv',
    '../data/metadata/airliners_metadata_trimmed_symmetric_diff.csv',
    '../data/metadata/counts_airlines_merged_trimmed.csv',
    '../data/metadata/counts_variants_trimmed.csv',
    0.25,
    0.25,
    5,
    42
)



'''

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

# Need to change...
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

# Need to change...
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


'''