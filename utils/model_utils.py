import torch
from torch import nn
from typing import Tuple
import torchvision.models as models
import warnings
from utils import vgg
from transformers import ViTForImageClassification, ViTConfig 

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
device


def format_title(model_name: str) -> str:
    name = model_name.lower()

    special = {
        "resnet": "ResNet",
        "vgg": "VGG",
        "vit": "ViT",
        "inception v1": "Inception V1",
        "googlenet": "GoogLeNet"
    }

    return special.get(name, model_name.capitalize())


def _create_custom_head(in_features: int, num_classes: int) -> nn.Sequential:

    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

def get_model(num_classes: int,
              model_name: str,
              device: torch.device = 'cpu'
             ) -> Tuple[nn.Module, str]:
    """
    Loads a pre-trained model, freezes parameters, and attaches a custom classifier.

    Supported models: 'ResNet', 'InceptionV1' (GoogLeNet), 'ViT', 'VGG'.

    Args:
        num_classes (int): The number of output classes for the new classifier.
        model_name (str): Name of the model architecture to load. 
        device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.

    Returns:
        Tuple[nn.Module, str]: A tuple containing:
            - The modified PyTorch model.
            - The formatted model name string.

    Raises:
        ValueError: If `model_name` is not one of the supported architectures.
    """
    
    architecture_name = format_title(model_name)

    # --- RESNET ---
    if architecture_name == 'ResNet':
        print("Loading ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = _create_custom_head(model.fc.in_features, num_classes)

    # --- INCEPTION V1 (GoogLeNet) ---
    elif architecture_name == 'Inception V1' or model_name == 'GoogLeNet':
        print("Loading InceptionV1 (GoogLeNet)...")
        
        weights = models.GoogLeNet_Weights.DEFAULT
        model = models.googlenet(weights=weights) 
        model.aux_logits = False 
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = _create_custom_head(model.fc.in_features, num_classes)

    # --- VISION TRANSFORMER (ViT) ---
    elif architecture_name == 'ViT':
        print("Loading ViT-B/16 from Hugging Face...")
        
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        for param in model.vit.parameters():
            param.requires_grad = False
            
        in_features = model.config.hidden_size
        model.classifier = _create_custom_head(in_features, num_classes)

    # --- VGG ---
    elif architecture_name == 'VGG':
        print("Loading VGG19...")
        model = vgg.VGG19(num_classes=num_classes)
        vgg_feature_output = 512 * 7 * 7
        model.classifier = _create_custom_head(in_features=vgg_feature_output, num_classes=num_classes)

    else:
        raise ValueError(f"Model '{architecture_name}' is not supported. Choose: ResNet, Inception V1, ViT, VGG.")
        
    return model.to(device), architecture_name
