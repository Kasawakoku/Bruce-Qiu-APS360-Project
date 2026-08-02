import os
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)

# Print to confirm it is going to the right place
print(f"Saving models to: {os.environ.get('TORCH_HOME', 'Default (~/.cache/torch)')}")

print("Downloading EfficientNet-B3...")
_ = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

print("Downloading ResNet-50...")
_ = resnet50(weights=ResNet50_Weights.DEFAULT)

print("Downloading ConvNeXt-Tiny...")
_ = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

print("Downloading ViT-B-16...")
_ = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

print("Downloading MobileNet-V3...")
_ = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

print("All models cached successfully!")