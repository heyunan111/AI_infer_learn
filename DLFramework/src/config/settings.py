"""
Configuration settings for ResNet-50 image classification system.

This module provides centralized configuration management for the ResNet-50 image
classification system. It includes the main Config dataclass with all training
parameters, validation functions, and utility functions for loading and creating
configurations.

Key Components:
    - Config: Main configuration dataclass with all system parameters
    - load_config: Function to load configuration from dictionary
    - validate_config: Function to validate configuration parameters
    - get_default_config: Function to get default configuration
    - create_config_from_original: Function to create config matching original script

Example:
    Basic usage:
        >>> config = Config()
        >>> validate_config(config)
        >>> print(f"Using device: {config.device}")
    
    Custom configuration:
        >>> custom_config = Config(
        ...     batch_size=32,
        ...     learning_rate=0.01,
        ...     epochs=100
        ... )
        >>> validate_config(custom_config)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import torch


@dataclass
class Config:
    """
    Configuration parameters for the ResNet-50 image classification system.
    
    This dataclass contains all configuration parameters needed for training,
    validation, and inference with the ResNet-50 model. It includes data paths,
    model settings, training hyperparameters, and output configurations.
    
    Attributes:
        data_path (str): Path to the root data directory containing images
        train_csv_path (str): Path to CSV file with training labels
        train_ratio (float): Ratio of data to use for training (0.0-1.0)
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of worker processes for data loading
        num_classes (int): Number of output classes for classification
        pretrained (bool): Whether to use pretrained ImageNet weights
        model_name (str): Name of the model architecture
        epochs (int): Maximum number of training epochs
        learning_rate (float): Initial learning rate for optimization
        patience (int): Early stopping patience (epochs without improvement)
        min_delta (float): Minimum change to qualify as improvement
        stage1_epochs (int): Number of epochs for stage 1 (feature extraction)
        stage1_patience (int): Early stopping patience for stage 1
        stage1_lr (float): Learning rate for stage 1
        stage1_step_size (int): Step size for stage 1 learning rate scheduler
        stage1_gamma (float): Gamma for stage 1 learning rate scheduler
        stage1_betas (tuple): Beta parameters for stage 1 Adam optimizer
        stage2_epochs (int): Number of epochs for stage 2 (fine-tuning)
        stage2_patience (int): Early stopping patience for stage 2
        stage2_lr (float): Learning rate for stage 2
        stage2_t0 (int): T_0 parameter for stage 2 cosine annealing scheduler
        stage2_t_mult (int): T_mult parameter for stage 2 cosine annealing scheduler
        stage2_betas (tuple): Beta parameters for stage 2 Adam optimizer
        image_size (int): Target size for input images (square)
        normalize_mean (tuple): Mean values for image normalization (RGB)
        normalize_std (tuple): Standard deviation values for image normalization (RGB)
        device (Optional[str]): Device to use ('cuda', 'cpu', or None for auto-detect)
        checkpoint_dir (str): Directory to save model checkpoints
        best_model_path (str): Path to save the best model during training
        final_model_path (str): Path to save the final best model
        training_history_plot (str): Path to save training history plot
        log_interval (int): Interval for logging training progress (batches)
        save_plots (bool): Whether to save training plots
    
    Example:
        >>> config = Config(
        ...     batch_size=32,
        ...     learning_rate=0.01,
        ...     epochs=50
        ... )
        >>> print(f"Training for {config.epochs} epochs with batch size {config.batch_size}")
    """
    
    # Data configuration
    data_path: str = "classify-leaves"
    train_csv_path: str = "classify-leaves/train.csv"
    train_ratio: float = 0.8
    batch_size: int = 16
    num_workers: int = 0
    
    # Model configuration
    num_classes: int = 176
    pretrained: bool = True
    model_name: str = "resnet50"
    
    # Training configuration - Basic
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    min_delta: float = 0.001
    
    # Training configuration - Two-stage
    stage1_epochs: int = 15
    stage1_patience: int = 5
    stage1_lr: float = 0.001
    stage1_step_size: int = 5
    stage1_gamma: float = 0.5
    stage1_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    
    stage2_epochs: int = 30
    stage2_patience: int = 8
    stage2_lr: float = 0.0001
    stage2_t0: int = 10
    stage2_t_mult: int = 2
    stage2_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    
    # Image preprocessing configuration
    image_size: int = 224
    normalize_mean: tuple = field(default_factory=lambda: (0.485, 0.456, 0.406))
    normalize_std: tuple = field(default_factory=lambda: (0.229, 0.224, 0.225))
    
    # Device configuration
    device: Optional[str] = None
    
    # Output configuration
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "best_model.pth"
    final_model_path: str = "final_best_model.pth"
    training_history_plot: str = "training_history.png"
    
    # Logging configuration
    log_interval: int = 100
    save_plots: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-detect device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Validate paths
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        if not os.path.exists(self.train_csv_path):
            raise ValueError(f"Train CSV path does not exist: {self.train_csv_path}")


def load_config(config_dict: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from dictionary or use defaults.
    
    Args:
        config_dict: Optional dictionary of configuration parameters
        
    Returns:
        Config: Configuration object
    """
    if config_dict is None:
        return Config()
    
    return Config(**config_dict)


def validate_config(config: Config) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Validate data parameters
    if config.train_ratio <= 0 or config.train_ratio >= 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {config.train_ratio}")
    
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {config.num_workers}")
    
    # Validate model parameters
    if config.num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {config.num_classes}")
    
    # Validate training parameters
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")
    
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    
    if config.patience <= 0:
        raise ValueError(f"patience must be positive, got {config.patience}")
    
    if config.min_delta < 0:
        raise ValueError(f"min_delta must be non-negative, got {config.min_delta}")
    
    # Validate two-stage training parameters
    if config.stage1_epochs <= 0:
        raise ValueError(f"stage1_epochs must be positive, got {config.stage1_epochs}")
    
    if config.stage1_patience <= 0:
        raise ValueError(f"stage1_patience must be positive, got {config.stage1_patience}")
    
    if config.stage1_lr <= 0:
        raise ValueError(f"stage1_lr must be positive, got {config.stage1_lr}")
    
    if config.stage2_epochs <= 0:
        raise ValueError(f"stage2_epochs must be positive, got {config.stage2_epochs}")
    
    if config.stage2_patience <= 0:
        raise ValueError(f"stage2_patience must be positive, got {config.stage2_patience}")
    
    if config.stage2_lr <= 0:
        raise ValueError(f"stage2_lr must be positive, got {config.stage2_lr}")
    
    # Validate image preprocessing parameters
    if config.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {config.image_size}")
    
    if len(config.normalize_mean) != 3:
        raise ValueError(f"normalize_mean must have 3 values, got {len(config.normalize_mean)}")
    
    if len(config.normalize_std) != 3:
        raise ValueError(f"normalize_std must have 3 values, got {len(config.normalize_std)}")
    
    # Validate device
    if config.device not in ["cpu", "cuda"]:
        if not config.device.startswith("cuda:"):
            raise ValueError(f"Invalid device: {config.device}")
    
    return True


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Config: Default configuration object
    """
    return Config()


def create_config_from_original() -> Config:
    """
    Create configuration that matches the original script parameters.
    
    Returns:
        Config: Configuration matching original script
    """
    return Config(
        data_path="classify-leaves",
        train_csv_path="classify-leaves/train.csv",
        train_ratio=0.8,
        batch_size=16,
        num_workers=0,
        num_classes=176,
        pretrained=True,
        epochs=50,
        learning_rate=0.001,
        patience=10,
        min_delta=0.001,
        stage1_epochs=15,
        stage1_patience=5,
        stage1_lr=0.001,
        stage2_epochs=30,
        stage2_patience=8,
        stage2_lr=0.0001,
        image_size=224,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        log_interval=100,
        save_plots=True
    )