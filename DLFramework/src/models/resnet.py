"""
ResNet model architecture and utilities for image classification.

This module provides a clean interface for ResNet-50 models with
layer freezing/unfreezing functionality and model creation utilities.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Optional, Tuple

try:
    from ..utils.helpers import count_parameters
except ImportError:
    from utils.helpers import count_parameters


class ResNetClassifier(nn.Module):
    """
    ResNet-50 classifier wrapper with clean interface and layer management.
    
    This class wraps the torchvision ResNet-50 model and provides additional
    functionality for layer freezing/unfreezing and parameter management.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Load ResNet-50 model
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Store original fc layer for reference
        self.fc = self.backbone.fc
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the backbone layers.
        
        Args:
            freeze: If True, freeze backbone layers (only fc trainable).
                   If False, unfreeze all layers.
        """
        for name, param in self.backbone.named_parameters():
            if freeze:
                # Stage 1: Only the final classification layer (fc) is trainable
                if 'fc' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                # Stage 2: All layers are trainable
                param.requires_grad = True
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """
        Get the number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params
    
    def print_param_info(self) -> None:
        """Print information about trainable parameters."""
        trainable_params, total_params = self.get_trainable_params()
        print(f"Trainable parameters: {trainable_params:,} / "
              f"Total parameters: {total_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")


def create_model(num_classes: int, pretrained: bool = True, device: Optional[torch.device] = None) -> ResNetClassifier:
    """
    Factory function for creating ResNet classifier models.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to move the model to (if None, uses CPU)
        
    Returns:
        ResNetClassifier instance
    """
    model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
    
    if device is not None:
        model = model.to(device)
    
    return model


def freeze_resnet_layers(model: ResNetClassifier, freeze_backbone: bool = True) -> None:
    """
    Utility function to freeze or unfreeze ResNetClassifier layers.
    
    This function is specific to ResNetClassifier and uses the model's built-in
    freeze_backbone method along with parameter info printing.
    
    Args:
        model: ResNetClassifier instance
        freeze_backbone: True to freeze backbone (feature extraction stage),
                        False to unfreeze all layers (fine-tuning stage)
    """
    model.freeze_backbone(freeze_backbone)
    model.print_param_info()


def get_model_summary(model: ResNetClassifier) -> Dict[str, Any]:
    """
    Get a summary of the model architecture and parameters.
    
    Args:
        model: ResNetClassifier instance
        
    Returns:
        Dictionary containing model summary information
    """
    trainable_params, total_params = model.get_trainable_params()
    
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_counts:
            layer_counts[module_type] = 0
        layer_counts[module_type] += 1
    
    return {
        'model_type': 'ResNet-50',
        'num_classes': model.num_classes,
        'pretrained': model.pretrained,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_ratio': trainable_params / total_params,
        'layer_counts': layer_counts
    }





def print_model_summary(model: ResNetClassifier, input_size: Tuple[int, int, int] = (3, 224, 224)) -> None:
    """
    Print a detailed summary of the model architecture.
    
    Args:
        model: ResNetClassifier instance
        input_size: Input tensor size (channels, height, width)
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    summary = get_model_summary(model)
    
    print(f"Model Type: {summary['model_type']}")
    print(f"Number of Classes: {summary['num_classes']}")
    print(f"Pretrained: {summary['pretrained']}")
    print(f"Input Size: {input_size}")
    print()
    
    print("Parameter Information:")
    print(f"  Total Parameters: {summary['total_parameters']:,}")
    print(f"  Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"  Trainable Ratio: {summary['trainable_ratio']:.2%}")
    print()
    
    print("Layer Counts:")
    for layer_type, count in sorted(summary['layer_counts'].items()):
        if count > 1:  # Only show layers that appear multiple times
            print(f"  {layer_type}: {count}")
    
    print("=" * 80)


def save_model_state(model: ResNetClassifier, filepath: str, 
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model state with additional metadata.
    
    Args:
        model: ResNetClassifier instance
        filepath: Path to save the model
        additional_info: Additional information to save with the model
    """
    state = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': model.num_classes,
            'pretrained': model.pretrained
        },
        'parameter_counts': count_parameters(model)
    }
    
    if additional_info:
        state.update(additional_info)
    
    torch.save(state, filepath)
    print(f"Model saved to {filepath}")


def load_model_state(filepath: str, num_classes: int, 
                    device: Optional[torch.device] = None) -> Tuple[ResNetClassifier, Dict[str, Any]]:
    """
    Load model state from file.
    
    Args:
        filepath: Path to the saved model
        num_classes: Number of classes (must match saved model)
        device: Device to load the model to
        
    Returns:
        Tuple of (loaded_model, metadata)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Validate model configuration
    saved_config = checkpoint.get('model_config', {})
    if saved_config.get('num_classes') != num_classes:
        raise ValueError(f"Model was trained for {saved_config.get('num_classes')} classes, "
                        f"but {num_classes} classes were requested")
    
    # Create and load model
    model = ResNetClassifier(
        num_classes=num_classes,
        pretrained=saved_config.get('pretrained', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    # Return model and metadata
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    print(f"Model loaded from {filepath}")
    return model, metadata


def validate_model_architecture(model: ResNetClassifier, expected_classes: int) -> bool:
    """
    Validate that the model architecture matches expectations.
    
    Args:
        model: ResNetClassifier instance
        expected_classes: Expected number of output classes
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check number of classes
        if model.num_classes != expected_classes:
            print(f"ERROR: Model has {model.num_classes} classes, expected {expected_classes}")
            return False
        
        # Check that the model can process a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        if output.shape != (1, expected_classes):
            print(f"ERROR: Model output shape is {output.shape}, expected (1, {expected_classes})")
            return False
        
        print("✅ Model architecture validation passed")
        return True
        
    except Exception as e:
        print(f"ERROR: Model architecture validation failed: {str(e)}")
        return False


def compare_models(model1: ResNetClassifier, model2: ResNetClassifier) -> Dict[str, Any]:
    """
    Compare two ResNet models and return differences.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
        
    Returns:
        Dictionary containing comparison results
    """
    summary1 = get_model_summary(model1)
    summary2 = get_model_summary(model2)
    
    comparison = {
        'same_architecture': (
            summary1['num_classes'] == summary2['num_classes'] and
            summary1['total_parameters'] == summary2['total_parameters']
        ),
        'parameter_diff': summary2['total_parameters'] - summary1['total_parameters'],
        'trainable_diff': summary2['trainable_parameters'] - summary1['trainable_parameters'],
        'class_diff': summary2['num_classes'] - summary1['num_classes']
    }
    
    return comparison