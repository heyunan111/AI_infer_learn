"""
High-level data management interface for orchestrating data loading operations.

This module provides the DataManager class that handles dataset creation,
splitting, DataLoader creation, and provides access to class information.
"""

import os
from typing import Dict, Any, Tuple, List, Optional, Union
import torch
from torch.utils.data import DataLoader, random_split
import logging

from .dataset import ImageFolderWithTxt, TransformDataset
from .transforms import DataTransforms

logger = logging.getLogger(__name__)


class DataManager:
    """
    High-level interface for managing data loading operations.
    
    This class orchestrates the creation of datasets, handles train/validation
    splitting, creates DataLoaders, and provides access to class information
    and dataset statistics.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing data parameters
        
    Attributes:
        config (Dict[str, Any]): Data configuration parameters
        transforms (DataTransforms): Transform manager instance
        full_dataset (ImageFolderWithTxt): Complete dataset before splitting
        train_dataset (TransformDataset): Training dataset with transforms
        val_dataset (TransformDataset): Validation dataset with transforms
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any]):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: Configuration dictionary or Config object
        """
        # Convert Config object to dictionary if needed
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        else:
            config_dict = config
            
        self.config = self._validate_config(config_dict)
        
        # Initialize transforms
        transform_config = self.config.get('transforms', {})
        self.transforms = DataTransforms(transform_config)
        
        # Initialize datasets and loaders as None
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        logger.info("Initialized DataManager")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set default values for configuration.
        
        Args:
            config (Dict[str, Any]): Input configuration
            
        Returns:
            Dict[str, Any]: Validated configuration with defaults
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Required parameters
        required_params = ['data_path', 'train_csv_path']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Required parameter '{param}' missing from config")
        
        # Set defaults
        defaults = {
            'train_ratio': 0.8,
            'batch_size': 16,
            'num_workers': 0,
            'shuffle_train': True,
            'shuffle_val': False,
            'pin_memory': True,
            'drop_last_train': False,
            'drop_last_val': False,
        }
        
        validated_config = defaults.copy()
        validated_config.update(config)
        
        # Validate values
        if not (0.0 < validated_config['train_ratio'] < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")
        
        if validated_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        if validated_config['num_workers'] < 0:
            raise ValueError("num_workers must be non-negative")
        
        return validated_config
    
    def setup_datasets(self) -> None:
        """
        Set up datasets by loading data and creating train/validation splits.
        
        Raises:
            FileNotFoundError: If data paths don't exist
            ValueError: If dataset creation fails
        """
        try:
            # Create full dataset
            self.full_dataset = ImageFolderWithTxt(
                root_dir=self.config['data_path'],
                txt_path=self.config['train_csv_path'],
                transform=None  # Transforms will be applied by wrapper
            )
            
            # Calculate split sizes
            total_size = len(self.full_dataset)
            train_size = int(self.config['train_ratio'] * total_size)
            val_size = total_size - train_size
            
            logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")
            
            # Create train/validation splits
            train_subset, val_subset = random_split(
                self.full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Wrap subsets with transforms
            self.train_dataset = TransformDataset(
                train_subset,
                self.transforms.get_train_transform()
            )
            
            self.val_dataset = TransformDataset(
                val_subset,
                self.transforms.get_val_transform()
            )
            
            logger.info("Successfully set up datasets")
            
        except Exception as e:
            logger.error(f"Failed to setup datasets: {str(e)}")
            raise
    
    def setup_dataloaders(self) -> None:
        """
        Create DataLoaders for training and validation datasets.
        
        Raises:
            RuntimeError: If datasets haven't been set up first
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("Datasets must be set up before creating DataLoaders")
        
        try:
            # Create training DataLoader
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=self.config['shuffle_train'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                drop_last=self.config['drop_last_train']
            )
            
            # Create validation DataLoader
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=self.config['shuffle_val'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                drop_last=self.config['drop_last_val']
            )
            
            logger.info("Successfully created DataLoaders")
            
        except Exception as e:
            logger.error(f"Failed to create DataLoaders: {str(e)}")
            raise
    
    def setup(self) -> None:
        """
        Complete setup of datasets and DataLoaders.
        
        This is a convenience method that calls setup_datasets() and
        setup_dataloaders() in sequence.
        """
        self.setup_datasets()
        self.setup_dataloaders()
        logger.info("DataManager setup complete")
    
    def get_datasets(self) -> Tuple[TransformDataset, TransformDataset]:
        """
        Get training and validation datasets.
        
        Returns:
            Tuple[TransformDataset, TransformDataset]: (train_dataset, val_dataset)
            
        Raises:
            RuntimeError: If datasets haven't been set up
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("Datasets not set up. Call setup_datasets() first.")
        
        return self.train_dataset, self.val_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and validation DataLoaders.
        
        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
            
        Raises:
            RuntimeError: If DataLoaders haven't been created
        """
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("DataLoaders not created. Call setup_dataloaders() first.")
        
        return self.train_loader, self.val_loader
    
    def get_class_info(self) -> Dict[str, Any]:
        """
        Get information about classes in the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'num_classes': Number of classes
                - 'class_names': List of class names
                - 'class_to_id': Mapping from class names to IDs
                - 'id_to_class': Mapping from IDs to class names
                - 'class_counts': Count of samples per class
                
        Raises:
            RuntimeError: If dataset hasn't been set up
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not set up. Call setup_datasets() first.")
        
        return {
            'num_classes': len(self.full_dataset.label_to_id),
            'class_names': self.full_dataset.get_class_names(),
            'class_to_id': self.full_dataset.label_to_id.copy(),
            'id_to_class': self.full_dataset.id_to_label.copy(),
            'class_counts': self.full_dataset.get_class_counts()
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the datasets.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics
            
        Raises:
            RuntimeError: If datasets haven't been set up
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not set up. Call setup_datasets() first.")
        
        stats = {
            'total_samples': len(self.full_dataset),
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'train_ratio': self.config['train_ratio'],
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
        }
        
        # Add class information
        class_info = self.get_class_info()
        stats.update(class_info)
        
        return stats
    
    def print_dataset_info(self) -> None:
        """Print detailed information about the datasets."""
        try:
            stats = self.get_dataset_stats()
            
            print("=" * 50)
            print("DATASET INFORMATION")
            print("=" * 50)
            print(f"Total samples: {stats['total_samples']:,}")
            print(f"Training samples: {stats['train_samples']:,}")
            print(f"Validation samples: {stats['val_samples']:,}")
            print(f"Number of classes: {stats['num_classes']}")
            print(f"Train ratio: {stats['train_ratio']:.2f}")
            print(f"Batch size: {stats['batch_size']}")
            print(f"Number of workers: {stats['num_workers']}")
            
            print("\nClass distribution:")
            class_counts = stats['class_counts']
            for class_name, count in sorted(class_counts.items())[:10]:  # Show first 10
                print(f"  {class_name}: {count}")
            
            if len(class_counts) > 10:
                print(f"  ... and {len(class_counts) - 10} more classes")
            
            print("=" * 50)
            
        except RuntimeError as e:
            print(f"Cannot display dataset info: {e}")
    
    def update_transforms(self, transform_config: Dict[str, Any]) -> None:
        """
        Update transform configuration and recreate datasets.
        
        Args:
            transform_config (Dict[str, Any]): New transform configuration
        """
        self.transforms.update_config(transform_config)
        
        # Update transforms in existing datasets if they exist
        if self.train_dataset is not None:
            self.train_dataset.set_transform(self.transforms.get_train_transform())
        
        if self.val_dataset is not None:
            self.val_dataset.set_transform(self.transforms.get_val_transform())
        
        logger.info("Updated transforms in DataManager")
    
    def get_sample_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Get a sample batch from the specified split.
        
        Args:
            split (str): Which split to sample from ('train' or 'val')
            
        Returns:
            Dict[str, torch.Tensor]: Sample batch
            
        Raises:
            ValueError: If split is invalid
            RuntimeError: If DataLoaders haven't been created
        """
        if split == 'train':
            if self.train_loader is None:
                raise RuntimeError("Train loader not created")
            loader = self.train_loader
        elif split == 'val':
            if self.val_loader is None:
                raise RuntimeError("Validation loader not created")
            loader = self.val_loader
        else:
            raise ValueError("split must be 'train' or 'val'")
        
        # Get first batch
        for batch in loader:
            return batch
        
        raise RuntimeError(f"No data in {split} loader")