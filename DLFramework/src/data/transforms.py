"""
Data transformation utilities for image preprocessing.

This module provides a centralized way to manage different transformation pipelines
for training and validation data, with configuration-driven transform selection.
"""

from typing import Optional, Dict, Any, List
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class DataTransforms:
    """
    Centralized class for managing different transformation pipelines.
    
    This class provides pre-configured transformation pipelines for different
    use cases (training, validation, testing) and supports configuration-driven
    transform selection.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for transforms
        train_transform (transforms.Compose): Training transformation pipeline
        val_transform (transforms.Compose): Validation transformation pipeline
        test_transform (transforms.Compose): Test transformation pipeline
    """
    
    # Default ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataTransforms with configuration.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary containing
                transform parameters. If None, uses default values.
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Create transformation pipelines
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        self.test_transform = self._create_test_transform()
        
        logger.info("Initialized DataTransforms with configuration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for transforms.
        
        Returns:
            Dict[str, Any]: Default configuration parameters
        """
        return {
            # Image size parameters
            'image_size': 224,
            'crop_size': 224,
            
            # Training augmentation parameters
            'random_resized_crop': True,
            'random_horizontal_flip': True,
            'random_vertical_flip': True,
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            
            # Color augmentation parameters
            'color_jitter': False,
            'color_jitter_brightness': 0.2,
            'color_jitter_contrast': 0.2,
            'color_jitter_saturation': 0.2,
            'color_jitter_hue': 0.1,
            
            # Rotation parameters
            'random_rotation': False,
            'rotation_degrees': 10,
            
            # Normalization parameters
            'normalize': True,
            'mean': self.IMAGENET_MEAN,
            'std': self.IMAGENET_STD,
            
            # Validation/test parameters
            'center_crop': True,
            'resize_size': 256,
        }
    
    def _create_train_transform(self) -> transforms.Compose:
        """
        Create training transformation pipeline with data augmentation.
        
        Returns:
            transforms.Compose: Training transformation pipeline
        """
        transform_list = []
        
        # Resize and crop
        if self.config['random_resized_crop']:
            transform_list.append(
                transforms.RandomResizedCrop(
                    self.config['crop_size'],
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                )
            )
        else:
            transform_list.extend([
                transforms.Resize(self.config['resize_size']),
                transforms.RandomCrop(self.config['crop_size'])
            ])
        
        # Flip augmentations
        if self.config['random_horizontal_flip']:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=self.config['horizontal_flip_prob'])
            )
        
        if self.config['random_vertical_flip']:
            transform_list.append(
                transforms.RandomVerticalFlip(p=self.config['vertical_flip_prob'])
            )
        
        # Rotation augmentation
        if self.config['random_rotation']:
            transform_list.append(
                transforms.RandomRotation(degrees=self.config['rotation_degrees'])
            )
        
        # Color augmentation
        if self.config['color_jitter']:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=self.config['color_jitter_brightness'],
                    contrast=self.config['color_jitter_contrast'],
                    saturation=self.config['color_jitter_saturation'],
                    hue=self.config['color_jitter_hue']
                )
            )
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.config['normalize']:
            transform_list.append(
                transforms.Normalize(
                    mean=self.config['mean'],
                    std=self.config['std']
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _create_val_transform(self) -> transforms.Compose:
        """
        Create validation transformation pipeline without augmentation.
        
        Returns:
            transforms.Compose: Validation transformation pipeline
        """
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize(self.config['resize_size']))
        
        # Center crop
        if self.config['center_crop']:
            transform_list.append(transforms.CenterCrop(self.config['crop_size']))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.config['normalize']:
            transform_list.append(
                transforms.Normalize(
                    mean=self.config['mean'],
                    std=self.config['std']
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _create_test_transform(self) -> transforms.Compose:
        """
        Create test transformation pipeline (same as validation).
        
        Returns:
            transforms.Compose: Test transformation pipeline
        """
        # Test transforms are the same as validation transforms
        return self._create_val_transform()
    
    def get_train_transform(self) -> transforms.Compose:
        """
        Get the training transformation pipeline.
        
        Returns:
            transforms.Compose: Training transformation pipeline
        """
        return self.train_transform
    
    def get_val_transform(self) -> transforms.Compose:
        """
        Get the validation transformation pipeline.
        
        Returns:
            transforms.Compose: Validation transformation pipeline
        """
        return self.val_transform
    
    def get_test_transform(self) -> transforms.Compose:
        """
        Get the test transformation pipeline.
        
        Returns:
            transforms.Compose: Test transformation pipeline
        """
        return self.test_transform
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration and recreate transforms.
        
        Args:
            new_config (Dict[str, Any]): New configuration parameters
        """
        self.config.update(new_config)
        
        # Recreate transforms with new configuration
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        self.test_transform = self._create_test_transform()
        
        logger.info("Updated transform configuration and recreated pipelines")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        return self.config.copy()
    
    def print_transforms(self) -> None:
        """Print information about current transforms."""
        print("Training Transform:")
        print(self.train_transform)
        print("\nValidation Transform:")
        print(self.val_transform)
        print("\nTest Transform:")
        print(self.test_transform)


def create_transforms_from_config(config: Dict[str, Any]) -> DataTransforms:
    """
    Factory function to create DataTransforms from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        DataTransforms: Configured DataTransforms instance
    """
    return DataTransforms(config)


def get_default_transforms() -> DataTransforms:
    """
    Get DataTransforms with default configuration.
    
    Returns:
        DataTransforms: DataTransforms instance with default settings
    """
    return DataTransforms()


def get_minimal_transforms(image_size: int = 224) -> DataTransforms:
    """
    Get minimal transforms for quick testing.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        DataTransforms: DataTransforms with minimal augmentation
    """
    config = {
        'image_size': image_size,
        'crop_size': image_size,
        'random_resized_crop': False,
        'random_horizontal_flip': False,
        'random_vertical_flip': False,
        'color_jitter': False,
        'random_rotation': False,
        'resize_size': image_size,
    }
    return DataTransforms(config)


def get_heavy_augmentation_transforms(image_size: int = 224) -> DataTransforms:
    """
    Get transforms with heavy augmentation for robust training.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        DataTransforms: DataTransforms with heavy augmentation
    """
    config = {
        'image_size': image_size,
        'crop_size': image_size,
        'random_resized_crop': True,
        'random_horizontal_flip': True,
        'random_vertical_flip': True,
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.3,
        'color_jitter': True,
        'color_jitter_brightness': 0.3,
        'color_jitter_contrast': 0.3,
        'color_jitter_saturation': 0.3,
        'color_jitter_hue': 0.1,
        'random_rotation': True,
        'rotation_degrees': 15,
        'resize_size': int(image_size * 1.15),
    }
    return DataTransforms(config)