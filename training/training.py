# Include packages

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
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
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings






# PARAMETERS
BATCH_SIZE = 16 # do not go above 16 for cpu
IMAGE_SIZE = 300

# File paths
image_folder_path = r"D:\Bruce-Qiu-APS360-Project\Data\airliners_images"


airline_csv_path = r"D:\Bruce-Qiu-APS360-Project\Data\metadata\counts_airlines_merged_trimmed.csv"
variant_csv_path = r"D:\Bruce-Qiu-APS360-Project\Data\metadata\counts_variants_trimmed.csv"

checkpoints_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_path"



# Neural Net Modules

    
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

# CNN baseline for variants 

class BaselineVariantCNN(nn.Module):
    """
    From-scratch Baseline CNN for Structural Variant Classification.
    Uses Global Average Pooling to capture overall structural shapes.
    """
    def __init__(self, num_variant_classes):
        super(BaselineVariantCNN, self).__init__()
        self.name = "Baseline_Variant_Scratch"
        
        # Lightweight Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 300 -> 150
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 150 -> 75
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 75 -> 37
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 37 -> 18
        )
        
        # Structural Head: Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_variant_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# CNN baseline for airlines (unused) 

class BaselineAirlineCNN(nn.Module):
    """
    From-scratch Baseline CNN for Airline Livery Classification.
    Uses Adaptive Max Pooling to capture sharp, localized branding features.
    """
    def __init__(self, num_airline_classes):
        super(BaselineAirlineCNN, self).__init__()
        self.name = "Baseline_Airline_Scratch"
        
        # Lightweight Convolutional Feature Extractor (Identical Architecture)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Branding Head: Adaptive Max Pooling
        self.pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_airline_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Models for Future Ablation study

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


# Data Loading

class PadToSquare:
    """
    Custom PyTorch Transform that pads a rectangular image to make it a square,
    maintaining the original aspect ratio.
    """

    # In the future, may implement to remove any images with extreme dimensions


    def __init__(self, fill=255): # fill=255 is white space, fill=0 is black
        self.fill = fill

    def __call__(self, img):
        # img is a PIL Image
        w, h = img.size
        max_dim = max(w, h)
        
        # Calculate padding for all 4 sides to center the image
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

# Transform pipelines
# Use 224 x 224 initially as proof of concept
train_transforms = transforms.Compose([
    PadToSquare(fill=255),                  # 1. Add white space to make it a perfect square
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),          # 2. Safely shrink the square down to 300x300
    transforms.RandomHorizontalFlip(),      # 3. Augment data
    transforms.ToTensor(),                  # 4. Convert to tensor
    transforms.Normalize(                   # 5. Normalize colors
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([ # Shared transform for eval and test
    PadToSquare(fill=255),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




class AirlinerDataset(Dataset):
    def __init__(self, dataframe, image_dir, airline_to_idx, variant_to_idx, transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): The loaded split dataframe (Train, Val, or Test).
            image_dir (str): The root folder path where all your scraped images are stored.
            airline_to_idx (dict): Dictionary mapping valid airline names to integers.
            variant_to_idx (dict): Dictionary mapping valid variant names to integers.
            transform (callable, optional): PyTorch transforms to standardize the image.
        """
        self.dataframe = dataframe

        '''
        existing = self.dataframe["image_filename"].apply(
            lambda x: os.path.exists(os.path.join(image_dir, str(x)))
        )

        missing = (~existing).sum()
        if missing:
            print(f"Removed {missing} missing images.")

        self.dataframe = self.dataframe[existing].reset_index(drop=True)
        '''


        self.image_dir = image_dir
        self.transform = transform
        self.airline_to_idx = airline_to_idx
        self.variant_to_idx = variant_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. Get the row
        row = self.dataframe.iloc[idx]
        
        # 2. Construct the full image path and load it
        img_name = str(row['image_filename'])
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"Skipping {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        
        
        # 3. Apply transformations (Resizing, normalizing, converting to tensor)
        if self.transform:
            image = self.transform(image)
            
        # 4. Handle Labels (Map to integer, defaulting to 'OTHERS')
        airline_str = str(row['airline'])
        variant_str = str(row['aircraft_variant'])
        
        # .get() looks up the string, and if it's missing, it returns the index for 'OTHERS'
        airline_label = self.airline_to_idx.get(airline_str, self.airline_to_idx['OTHERS'])
        variant_label = self.variant_to_idx.get(variant_str, self.variant_to_idx['OTHERS'])
        
        # Return the processed image and both labels as tensors
        return image, torch.tensor(variant_label, dtype=torch.long), torch.tensor(airline_label, dtype=torch.long)



def load_split_dataframes(train_csv_path, val_csv_path, test_csv_path):
    """
    Utility function to load the generated split CSVs back into pandas DataFrames.
    Useful for feeding directly into the PyTorch custom Dataset class.
    """
    print("Loading generated splits from disk...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Fill NaNs as 'OTHERS' string just in case pandas parsed empty columns as floats
    train_df.fillna('OTHERS', inplace=True)
    val_df.fillna('OTHERS', inplace=True)
    test_df.fillna('OTHERS', inplace=True)
    
    return train_df, val_df, test_df


def build_mapping_from_csv(csv_path, column_name):
    """Reads a CSV and builds a dictionary mapping strings to integers."""
    df = pd.read_csv(csv_path)
    
    # Extract unique names and drop any empty rows
    class_list = df[column_name].dropna().unique().tolist()
    
    # Ensure 'OTHERS' is a valid class in our list
    if 'OTHERS' not in class_list:
        class_list.append('OTHERS')
        
    # Create the dictionary: {'AJet': 0, 'Aeroflot': 1, ..., 'OTHERS': N}
    return {name: idx for idx, name in enumerate(class_list)}

# Training loop

def get_model_name(name, batch_size, learning_rate, epoch, checkpoint_dir="checkpoints"):
    """
    Generate a path for saving the model checkpoints.
    Allows specifying a custom directory (e.g., 'D:\\Bruce-Qiu-APS360-Project\\checkpoints')
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    # Use os.path.join for safe Windows/Mac/Linux path formatting
    filename = f"model_{name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}.pt"
    path = os.path.join(checkpoint_dir, filename)
    return path

def load_model_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Utility function to safely load a model, optimizer state, and training history.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Restore model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Restore optimizer state if provided and available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0)
    loaded_history = checkpoint.get("history", None)
    
    print(f"Successfully loaded checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Resuming training from epoch {start_epoch}...")
    
    return model, optimizer, start_epoch, loaded_history


def evaluate(net, loader, criterion, device, is_multitask=True):
    """
    Evaluate network on validation or test set.
    Computes loss and weighted F1-Scores for either multi-task or single-task models.
    """
    total_loss = 0.0
    
    # Store all predictions and true labels for F1 calculation
    
    all_var_preds = []
    all_var_labels = []
    
    if is_multitask:
        all_air_preds = []
        all_air_labels = []
    

    net.eval() # set to evaluation mode
    with torch.no_grad(): # reduce memory and runtime length
        for data in loader:
            # Unpack the 3 items returned by your custom AirlinerDataset
            inputs, variant_labels, airline_labels = data
            inputs = inputs.to(device)
            variant_labels = variant_labels.to(device)
            airline_labels = airline_labels.to(device)

            if is_multitask:
                # Primary Model: Returns two branches
                var_outputs, air_outputs = net(inputs)
                
                # Combined Multi-Task Loss
                # Right now just equal weights, can make updatable weights for future update
                # Try: Uncertainty-Based Adaptive Weighting (Kendall et al 2018)
                loss = criterion(var_outputs, variant_labels) + criterion(air_outputs, airline_labels)
                
                # Get predictions
                _, var_preds = torch.max(var_outputs.data, 1)
                _, air_preds = torch.max(air_outputs.data, 1)
                
                # Append for F1 computation
                all_air_preds.extend(air_preds.cpu().numpy())
                all_air_labels.extend(airline_labels.cpu().numpy())
                
            else:
                # Baseline Model: Returns single branch (Variant only)
                var_outputs = net(inputs)
                loss = criterion(var_outputs, variant_labels)
                
                # Get predictions
                _, var_preds = torch.max(var_outputs.data, 1)

            total_loss += loss.item()
            all_var_preds.extend(var_preds.cpu().numpy())
            all_var_labels.extend(variant_labels.cpu().numpy())

    # Compute Weighted F1-Scores (handles imbalanced classes)
    var_f1 = f1_score(all_var_labels, all_var_preds, average='weighted')
    avg_loss = float(total_loss) / len(loader)

    if is_multitask:
        air_f1 = f1_score(all_air_labels, all_air_preds, average='weighted')
        return var_f1, air_f1, avg_loss
    else:
        return var_f1, avg_loss


def plot_training_curve(path, is_multitask=True):
    """
    Plot training curves for F1-Scores and Loss.
    Dynamically adjusts layout based on whether it was a baseline or primary model run.
    Plots both Epoch-level (Train vs Val) and Iteration-level (Train vs Val) charts.
    """
    # Load Epoch metrics
    train_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_var_f1.csv"))
    val_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_var_f1.csv"))
    
    train_loss = np.atleast_1d(np.loadtxt(f"{path}_train_loss.csv"))
    val_loss = np.atleast_1d(np.loadtxt(f"{path}_val_loss.csv"))
    
    if is_multitask:
        # Load Airline F1 metrics if primary model
        train_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_train_air_f1.csv"))
        val_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_val_air_f1.csv"))

    # Load Iteration metrics (if they exist)
    try:
        iter_steps = np.atleast_1d(np.loadtxt(f"{path}_iter_steps.csv"))
        iter_train_loss = np.atleast_1d(np.loadtxt(f"{path}_iter_train_loss.csv"))
        iter_train_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_var_f1.csv"))
        
        # New Iteration validation metrics
        iter_val_loss = np.atleast_1d(np.loadtxt(f"{path}_iter_val_loss.csv"))
        iter_val_var_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_var_f1.csv"))
        
        if is_multitask:
            iter_train_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_train_air_f1.csv"))
            iter_val_air_f1 = np.atleast_1d(np.loadtxt(f"{path}_iter_val_air_f1.csv"))
        has_iters = True
    except OSError:
        has_iters = False

    # --- Plot 1: Epoch-Level Metrics ---
    n_epochs = len(train_var_f1)

    # Make the figure wider if we have 3 plots (Multi-task) instead of 2 (Baseline)
    plt.figure(figsize=(15 if is_multitask else 10, 4))
    plt.suptitle("Epoch-Level Metrics (Train vs Validation)", fontsize=14, y=1.05)

    # Plot Variant F1-Score
    plt.subplot(1, 3 if is_multitask else 2, 1)
    plt.title("Variant Weighted F1-Score")
    plt.plot(range(1, n_epochs+1), train_var_f1, label="Train")
    plt.plot(range(1, n_epochs+1), val_var_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.legend(loc='best')
    plt.locator_params(axis='x', integer=True)

    # Plot Airline F1-Score (Only if Multi-Task)
    if is_multitask:
        plt.subplot(1, 3, 2)
        plt.title("Airline Weighted F1-Score")
        plt.plot(range(1, n_epochs+1), train_air_f1, label="Train")
        plt.plot(range(1, n_epochs+1), val_air_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.legend(loc='best')
        plt.locator_params(axis='x', integer=True)

    # Plot Total Loss
    plt.subplot(1, 3 if is_multitask else 2, 3 if is_multitask else 2)
    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n_epochs+1), train_loss, label="Train")
    plt.plot(range(1, n_epochs+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.locator_params(axis='x', integer=True)

    plt.tight_layout()
    plt.show()

    # --- Plot 2: Iteration-Level Metrics ---
    if has_iters and len(iter_steps) > 0:
        plt.figure(figsize=(15 if is_multitask else 10, 4))
        plt.suptitle("Iteration-Level Metrics (Train vs Validation)", fontsize=14, y=1.05)

        # Plot Variant F1 (Iter)
        plt.subplot(1, 3 if is_multitask else 2, 1)
        plt.title("Variant Weighted F1-Score")
        plt.plot(iter_steps, iter_train_var_f1, label="Train", color='blue')
        plt.plot(iter_steps, iter_val_var_f1, label="Validation", color='red')
        plt.xlabel("Iteration (Batches)")
        plt.ylabel("F1-Score")
        plt.legend(loc='best')

        if is_multitask:
            plt.subplot(1, 3, 2)
            plt.title("Airline Weighted F1-Score")
            plt.plot(iter_steps, iter_train_air_f1, label="Train", color='blue')
            plt.plot(iter_steps, iter_val_air_f1, label="Validation", color='red')
            plt.xlabel("Iteration (Batches)")
            plt.ylabel("F1-Score")
            plt.legend(loc='best')

        plt.subplot(1, 3 if is_multitask else 2, 3 if is_multitask else 2)
        plt.title("Train vs Validation Loss")
        plt.plot(iter_steps, iter_train_loss, label="Train Loss", color='blue')
        plt.plot(iter_steps, iter_val_loss, label="Val Loss", color='red')
        plt.xlabel("Iteration (Batches)")
        plt.ylabel("Loss")
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()


def train_net(net, train_loader, val_loader, batch_size=64, learning_rate=0.01, num_epochs=30, checkpoint_freq=1, 
              is_multitask=True, checkpoint_dir="checkpoints", optimizer=None, start_epoch=0, 
              track_iterations=True, record_freq=100, loaded_history=None):
    """
    Trains the neural network. 
    Set track_iterations=False to disable high-granularity tracking for faster training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    net.to(device)
    torch.manual_seed(1000)

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Arrays to store epoch metrics
    train_var_f1 = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_var_f1 = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    if is_multitask:
        train_air_f1 = np.zeros(num_epochs)
        val_air_f1 = np.zeros(num_epochs)

    # Arrays to store iteration metrics
    global_step = 0
    iter_steps = []
    iter_train_loss, iter_val_loss = [], []
    iter_train_var_f1, iter_val_var_f1 = [], []
    if is_multitask:
        iter_train_air_f1, iter_val_air_f1 = [], []

    # ============================================================
    # RESUME HISTORY LOGIC
    # ============================================================
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

    from tqdm import tqdm

    for epoch in range(start_epoch, num_epochs): 
        print(f"Epoch {epoch + 1}...")
        net.train()
        total_train_loss = 0.0
        
        all_var_preds, all_var_labels = [], []
        if is_multitask:
            all_air_preds, all_air_labels = [], []

        # Variables for window tracking
        window_loss = 0.0
        window_var_preds, window_var_labels = [], []
        if is_multitask:
            window_air_preds, window_air_labels = [], []

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, data in enumerate(progress):
            if track_iterations:
                global_step += 1
            
            inputs, variant_labels, airline_labels = data
            inputs = inputs.to(device)
            variant_labels = variant_labels.to(device)
            airline_labels = airline_labels.to(device)

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
                    # Use logical OR inline so smaller windows at the end of an epoch aren't artificially shrunk
                    iter_train_loss.append(window_loss / ((global_step % record_freq) or record_freq))
                    iter_train_var_f1.append(f1_score(window_var_labels, window_var_preds, average='weighted', zero_division=0))
                    
                    if is_multitask:
                        iter_train_air_f1.append(f1_score(window_air_labels, window_air_preds, average='weighted', zero_division=0))
                        v_var, v_air, v_loss = evaluate(net, val_loader, criterion, device, is_multitask)
                        iter_val_var_f1.append(v_var)
                        iter_val_air_f1.append(v_air)
                        iter_val_loss.append(v_loss)
                    else:
                        v_var, v_loss = evaluate(net, val_loader, criterion, device, is_multitask)
                        iter_val_var_f1.append(v_var)
                        iter_val_loss.append(v_loss)
                        
                    net.train()
                    
                    # Reset window
                    window_loss = 0.0
                    window_var_preds, window_var_labels = [], []
                    if is_multitask:
                        window_air_preds, window_air_labels = [], []

            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate epoch metrics
        train_var_f1[epoch] = f1_score(all_var_labels, all_var_preds, average='weighted', zero_division=0)
        train_loss[epoch] = float(total_train_loss) / len(train_loader)
        if is_multitask:
            train_air_f1[epoch] = f1_score(all_air_labels, all_air_preds, average='weighted', zero_division=0)

        # Evaluate on validation set
        if is_multitask:
            val_var_f1[epoch], val_air_f1[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device, is_multitask)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f} | Train Var F1: {train_var_f1[epoch]:.4f} | Train Air F1: {train_air_f1[epoch]:.4f}")
            print(f"          Val Loss: {val_loss[epoch]:.4f} | Val Var F1: {val_var_f1[epoch]:.4f} | Val Air F1: {val_air_f1[epoch]:.4f}")
        else:
            val_var_f1[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device, is_multitask)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f} | Train Var F1: {train_var_f1[epoch]:.4f}")
            print(f"          Val Loss: {val_loss[epoch]:.4f} | Val Var F1: {val_var_f1[epoch]:.4f}")

        # Checkpointing
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            model_name = getattr(net, 'name', 'model')
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
                "history": history_dict
            }, model_path)

    print('Finished Training')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))

    model_name = getattr(net, 'name', 'model')
    model_base_path = get_model_name(model_name, batch_size, learning_rate, "final", checkpoint_dir).replace('.pt', '')
    
    # Save Epoch CSVs
    np.savetxt(f"{model_base_path}_train_var_f1.csv", train_var_f1)
    np.savetxt(f"{model_base_path}_val_var_f1.csv", val_var_f1)
    np.savetxt(f"{model_base_path}_train_loss.csv", train_loss)
    np.savetxt(f"{model_base_path}_val_loss.csv", val_loss)
    if is_multitask:
        np.savetxt(f"{model_base_path}_train_air_f1.csv", train_air_f1)
        np.savetxt(f"{model_base_path}_val_air_f1.csv", val_air_f1)
    
    # Save Iteration CSVs
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


# Manual testing on singular images

def pad_to_square(img, fill_color=(255, 255, 255)):
    """
    Pads a PIL image with white pixels to make it a perfect square, 
    preserving the original aspect ratio (matching your training loader).
    """
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_img = Image.new('RGB', (size, size), fill_color)
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_img


def predict_image(image_path, model, device, variant_idx_to_name, airline_idx_to_name=None, is_multitask=True):
    """
    Loads a single image from disk, preprocesses it identically to the training set,
    runs it through the model, and returns the human-readable class names.
    """
    model.eval() # Ensure dropout and batchnorm are in inference mode
    
    # 1. Load and Pad
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    img = pad_to_square(img)
    
    # 2. Transform (Resize, ToTensor, Normalize)
    # NOTE: If you used specific normalization values during training (like ImageNet means), 
    # make sure they match here!
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Uncomment if used in training
    ])
    
    # Add batch dimension: [Channels, H, W] -> [1, Channels, H, W]
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 3. Predict
    with torch.no_grad():
        if is_multitask:
            var_outputs, air_outputs = model(img_tensor)
            
            # Get the index of the highest probability
            var_idx = torch.argmax(var_outputs, dim=1).item()
            air_idx = torch.argmax(air_outputs, dim=1).item()
            
            # Map back to string labels
            var_pred = variant_idx_to_name[var_idx]
            air_pred = airline_idx_to_name[air_idx]
            
            return var_pred, air_pred
        else:
            var_outputs = model(img_tensor)
            var_idx = torch.argmax(var_outputs, dim=1).item()
            var_pred = variant_idx_to_name[var_idx]
            
            return var_pred

# Final Execution
if __name__ == "__main__":
    
    # Setup
    
    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split_dataframes(
        r'D:\Bruce-Qiu-APS360-Project\Data\metadata\train\train_metadata.csv', 
        r'D:\Bruce-Qiu-APS360-Project\Data\metadata\val\val_metadata.csv', 
        r'D:\Bruce-Qiu-APS360-Project\Data\metadata\test\test_metadata.csv'
    )
    print("Load split data frame complete")

    # Automatically build the mappings (Ensure the column names exactly match your CSVs)
    airline_mapping = build_mapping_from_csv(airline_csv_path, column_name='Airline')
    variant_mapping = build_mapping_from_csv(variant_csv_path, column_name='Variant')

    # (Optional) Print to verify the mappings and total class counts
    print(f"Total Airline Classes (including OTHERS): {len(airline_mapping)}")
    print(f"Total Variant Classes (including OTHERS): {len(variant_mapping)}")

    # 1. Instantiate the Datasets
    train_dataset = AirlinerDataset(train_df, image_folder_path, airline_mapping, variant_mapping, transform=train_transforms)
    val_dataset = AirlinerDataset(val_df, image_folder_path, airline_mapping, variant_mapping, transform=eval_transforms)
    test_dataset = AirlinerDataset(test_df, image_folder_path, airline_mapping, variant_mapping, transform=eval_transforms)

    # 2. Wrap them in DataLoaders (this handles batching and shuffling)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Loaders complete")

    
    # Example constants (replace with the lengths of your mapping dictionaries)
    NUM_VARIANT_CLASSES = len(variant_mapping)
    NUM_AIRLINE_CLASSES = len(airline_mapping)
    
    
    

    # ---------------------------------------------------------
    # RUNNING THE BASELINE MODEL
    # ---------------------------------------------------------
    
    
    '''
    baseline_variant_model = BaselineVariantCNN(
        NUM_VARIANT_CLASSES
    )

    checkpoint_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_variant_baseline\model_Baseline_Variant_Scratch_bs16_lr0.001_epoch30.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_variant_model = BaselineVariantCNN(num_variant_classes=NUM_VARIANT_CLASSES).to(device)
    optimizer = optim.Adam(baseline_variant_model.parameters(), lr=0.001)

    baseline_variant_model, optimizer, start_epoch, loaded_history = load_model_checkpoint(
        checkpoint_path=checkpoint_path, 
        model=baseline_variant_model, 
        optimizer=optimizer, 
        device=device
    )

    # Baseline
    saved_multi_path = train_net(
        #net=baseline_variant_model,
        net=baseline_variant_model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        num_epochs=35,
        
        is_multitask=False,
        checkpoint_freq = 1,
        checkpoint_dir="checkpoints_variant_baseline",
        start_epoch=start_epoch,
        loaded_history=loaded_history,
        track_iterations=False,
        record_freq=100
        
        
    )

    plot_training_curve(saved_multi_path, is_multitask=False)
    
    '''

    # ---------------------------------------------------------
    # RUNNING THE PRIMARY MODEL
    # ---------------------------------------------------------
    '''
    print("Initializing Dual-Branch Model...")
    primary_model = DualBranchNet(
        num_variant_classes=NUM_VARIANT_CLASSES, 
        num_airline_classes=NUM_AIRLINE_CLASSES
    )

    
    
    
    checkpoint_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_primary_2\model_model_bs16_lr0.001_epoch1.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary_model = DualBranchNet(
        num_variant_classes=NUM_VARIANT_CLASSES, 
        num_airline_classes=NUM_AIRLINE_CLASSES
    ).to(device)
    optimizer = optim.Adam(primary_model.parameters(), lr=0.001)


    primary_model, optimizer, start_epoch, loaded_history = load_model_checkpoint(
        checkpoint_path=checkpoint_path, 
        model=primary_model, 
        optimizer=optimizer, 
        device=device
    )
    

    
    saved_multi_path = train_net(
        #net=baseline_variant_model,
        net=primary_model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        num_epochs=3,
        
        is_multitask=True,
        checkpoint_freq = 1,
        checkpoint_dir="checkpoints_primary_2",
        start_epoch=start_epoch,
        loaded_history=loaded_history,
        track_iterations=True,
        record_freq=100
        
        
    )

    plot_training_curve(saved_multi_path, is_multitask=True)
    '''

    # ---------------------------------------------------------
    # SANITY CHECK WITH SMALL DATASET
    # ---------------------------------------------------------

    '''
    # Sanity Check code
    N = 16

    small_train = Subset(train_dataset, range(N))

    small_loader = DataLoader(
        small_train,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    

    primary_model = DualBranchNet(
        NUM_VARIANT_CLASSES,
        NUM_AIRLINE_CLASSES
    )

    baseline_variant_model = BaselineVariantCNN(
        NUM_VARIANT_CLASSES
    )


    checkpoint_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_variant_baseline_sanity\model_Baseline_Variant_Scratch_bs8_lr0.001_epoch20.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Instantiate your model and optimizer first
    # (Assuming NUM_VARIANT_CLASSES is already defined in your script)
    model = BaselineVariantCNN(num_variant_classes=NUM_VARIANT_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Load the checkpoint dictionary from the file path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 4. Restore the weights, optimizer state, and extract the history
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]

    # Use .get() so it defaults to None safely if you load an older checkpoint 
    # that didn't have history saved in it!
    loaded_history = checkpoint.get("history", None)

    saved_multi_path = train_net(
        #net=baseline_variant_model,
        net=model,
        train_loader=small_loader,
        val_loader=small_loader,
        batch_size=8,
        learning_rate=1e-3,
        num_epochs=30,
        start_epoch=start_epoch,
        is_multitask=False,
        checkpoint_freq = 5,
        checkpoint_dir="checkpoints_variant_baseline_sanity_2",
        loaded_history=loaded_history,
        record_freq=2
        
    )

    plot_training_curve(saved_multi_path, is_multitask=False)
    '''

    # ---------------------------------------------------------
    # MANUAL TESTING
    # ---------------------------------------------------------
    

    # Automatically build the mappings (Ensure the column names exactly match your CSVs)
    airline_mapping = build_mapping_from_csv(airline_csv_path, column_name='Airline')
    variant_mapping = build_mapping_from_csv(variant_csv_path, column_name='Variant')
    # Example constants (replace with the lengths of your mapping dictionaries)
    NUM_VARIANT_CLASSES = len(variant_mapping)
    NUM_AIRLINE_CLASSES = len(airline_mapping)
    
    checkpoint_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_variant_baseline\model_Baseline_Variant_Scratch_bs16_lr0.001_epoch35.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_variant_model = BaselineVariantCNN(num_variant_classes=NUM_VARIANT_CLASSES).to(device)
    optimizer = optim.Adam(baseline_variant_model.parameters(), lr=0.001)

    # Now, just call the helper function!
    baseline_variant_model, optimizer, start_epoch, loaded_history = load_model_checkpoint(
        checkpoint_path=checkpoint_path, 
        model=baseline_variant_model, 
        optimizer=optimizer, 
        device=device
    )

    checkpoint_path = r"D:\Bruce-Qiu-APS360-Project\training\checkpoints_primary_2\model_model_bs16_lr0.001_epoch3.pt"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary_model = DualBranchNet(
        num_variant_classes=NUM_VARIANT_CLASSES, 
        num_airline_classes=NUM_AIRLINE_CLASSES
    ).to(device)
    optimizer = optim.Adam(primary_model.parameters(), lr=0.001)

    primary_model, optimizer, start_epoch, loaded_history = load_model_checkpoint(
        checkpoint_path=checkpoint_path, 
        model=primary_model, 
        optimizer=optimizer, 
        device=device
    )

    variant_idx_to_name = {v: k for k, v in variant_mapping.items()}
    airline_idx_to_name = {v: k for k, v in airline_mapping.items()}


    raw_image_path = r"D:\Bruce-Qiu-APS360-Project\Data\test_images\_DSC1428.JPG"

    predicted_variant = predict_image(
        image_path=raw_image_path,
        model=baseline_variant_model,
        device=device,
        variant_idx_to_name=variant_idx_to_name,
        airline_idx_to_name=airline_idx_to_name,
        is_multitask=False
    )

    print(f"Primary predicted:")
    print(f"Variant: {predicted_variant}")
    print("\n")

    predicted_variant, predicted_airline = predict_image(
        image_path=raw_image_path,
        model=primary_model,
        device=device,
        variant_idx_to_name=variant_idx_to_name,
        airline_idx_to_name=airline_idx_to_name,
        is_multitask=True
    )

    print(f"Primary predicted:")
    print(f"Variant: {predicted_variant}")
    print(f"Airline: {predicted_airline}")