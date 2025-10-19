"""
Data handling module for image classification.

This module provides classes and utilities for loading, transforming, and managing
image datasets with text-based labels.
"""

from .dataset import ImageFolderWithTxt, TransformDataset
from .transforms import DataTransforms, create_transforms_from_config, get_default_transforms
from .manager import DataManager

__all__ = [
    'ImageFolderWithTxt',
    'TransformDataset', 
    'DataTransforms',
    'DataManager',
    'create_transforms_from_config',
    'get_default_transforms'
]