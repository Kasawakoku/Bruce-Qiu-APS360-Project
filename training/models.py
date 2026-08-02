# Model/Network Architecture Definitions

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights
)

# ---------------------------------------------------------
# PRIMARY MODEL
# ---------------------------------------------------------
class DualBranchEfficientNet(nn.Module):
    def __init__(self, num_variant_classes, num_airline_classes):
        super(DualBranchEfficientNet, self).__init__()
        self.name = "Primary_DualBranch"
        
        # 1. Shared Convolutional Backbone
        base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = base_model.features
        feature_dim = 1536 
        
        # 2. Structural Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.structural_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_variant_classes)
        )
        
        # 3. Branding Head
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.branding_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_airline_classes)
        )

    def forward(self, x):
        shared_features = self.features(x)
        
        v = self.gap(shared_features).view(shared_features.size(0), -1)
        variant_pred = self.structural_mlp(v)
        
        a = self.amp(shared_features).view(shared_features.size(0), -1)
        airline_pred = self.branding_mlp(a)
        
        return variant_pred, airline_pred

from torchvision.models import (
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights
)



# ---------------------------------------------------------
# ABLATIONS
# ---------------------------------------------------------

class DualBranchResNet(nn.Module):
    def __init__(self, num_variant_classes, num_airline_classes):
        super(DualBranchResNet, self).__init__()
        self.name = "Primary_ResNet50"
        
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Strip the final fully connected layer and pooling
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        feature_dim = 2048
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.amp = nn.AdaptiveMaxPool2d(1)
        
        self.structural_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_variant_classes)
        )
        self.branding_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_airline_classes)
        )

    def forward(self, x):
        shared_features = self.features(x)
        
        v = self.gap(shared_features).view(shared_features.size(0), -1)
        variant_pred = self.structural_mlp(v)
        
        a = self.amp(shared_features).view(shared_features.size(0), -1)
        airline_pred = self.branding_mlp(a)
        
        return variant_pred, airline_pred


class DualBranchConvNeXt(nn.Module):
    def __init__(self, num_variant_classes, num_airline_classes):
        super(DualBranchConvNeXt, self).__init__()
        self.name = "Primary_ConvNeXt"
        
        base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = base_model.features
        feature_dim = 768
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.amp = nn.AdaptiveMaxPool2d(1)
        
        self.structural_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_variant_classes)
        )
        self.branding_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_airline_classes)
        )

    def forward(self, x):
        shared_features = self.features(x)
        
        v = self.gap(shared_features).view(shared_features.size(0), -1)
        variant_pred = self.structural_mlp(v)
        
        a = self.amp(shared_features).view(shared_features.size(0), -1)
        airline_pred = self.branding_mlp(a)
        
        return variant_pred, airline_pred

class DualBranchViT(nn.Module):
    def __init__(self, num_variant_classes, num_airline_classes):
        super(DualBranchViT, self).__init__()
        self.name = "Primary_ViT"
        
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        feature_dim = 768 # Hidden dim for ViT Base
        
        # Replace the original classification head with an Identity layer
        # This forces the model to output the raw (B, 768) CLS token embedding
        self.base_model.heads = nn.Identity()
        
        # No GAP or GMP pooling needed, the CLS token is already a 1D vector per image
        self.structural_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_variant_classes)
        )
        self.branding_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_airline_classes)
        )

    def forward(self, x):
        # Outputs shape: (Batch, 768)
        shared_features = self.base_model(x)
        
        variant_pred = self.structural_mlp(shared_features)
        airline_pred = self.branding_mlp(shared_features)
        
        return variant_pred, airline_pred

#---------------------------------------------------------
# SINGLE TASK MODELS
#---------------------------------------------------------

class SingleTaskNet(nn.Module):
    def __init__(self, backbone_type, num_classes, task_name="Variant"):
        super(SingleTaskNet, self).__init__()
        self.name = f"SingleTask_{backbone_type}_{task_name}"
        
        if backbone_type == "efficientnet":
            base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            self.features = base_model.features
            feature_dim = 1536
        elif backbone_type == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError("Unsupported backbone for SingleTaskNet template")
            
        if task_name == "Variant":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
            
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.mlp(x)
        return x

# ---------------------------------------------------------
# BASELINES
# ---------------------------------------------------------
class BaselineVariantCNN(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineVariantCNN, self).__init__()
        self.name = "Baseline_Variant_Scratch"
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
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

class BaselineAirlineCNN(nn.Module):
    def __init__(self, num_airline_classes):
        super(BaselineAirlineCNN, self).__init__()
        self.name = "Baseline_Airline_Scratch"
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveMaxPool2d(1)
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

# ---------------------------------------------------------
# ABLATION MODELS
# ---------------------------------------------------------
class BaselineEfficientNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineEfficientNet, self).__init__()
        self.name = "EfficientNet_Ablation"
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_variant_classes)
        )
    def forward(self, x): return self.model(x)

class BaselineResNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineResNet, self).__init__()
        self.name = "ResNet_Ablation"
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features 
        self.model.fc = nn.Linear(in_features, num_variant_classes)
    def forward(self, x): return self.model(x)

class BaselineMobileNet(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineMobileNet, self).__init__()
        self.name = "MobileNet_Ablation"
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_variant_classes)
    def forward(self, x): return self.model(x)
    
class BaselineConvNeXt(nn.Module):
    def __init__(self, num_variant_classes):
        super(BaselineConvNeXt, self).__init__()
        self.name = "ConvNeXt_Ablation"
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_variant_classes)
    def forward(self, x): return self.model(x)

# ---------------------------------------------------------
# Uncertainty-Based Adaptive Weighting
# ---------------------------------------------------------

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        # Initialize log variance for each task to 0. 
        # We use log variance for numerical stability to avoid dividing by zero.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, loss_v, loss_a):
        # Formula: L_total = exp(-log_var) * L + log_var
        # Task 0: Variant, Task 1: Airline
        precision_v = torch.exp(-self.log_vars[0])
        loss_variant_weighted = precision_v * loss_v + self.log_vars[0]
        
        precision_a = torch.exp(-self.log_vars[1])
        loss_airline_weighted = precision_a * loss_a + self.log_vars[1]
        
        return loss_variant_weighted + loss_airline_weighted