import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def get_model():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    
    # Replace classifier head
    model.classifier[4] = nn.Sequential(
        nn.Conv2d(256, 1, kernel_size=1),
        nn.Sigmoid()
    )
    return model
