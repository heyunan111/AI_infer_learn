"""
General utility functions for the ResNet classification system.

This module provides helper functions for:
- Reproducibility (seed setting)
- Device detection and management
- Model checkpoint saving and loading
- General utility functions
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Dict, Any, Optional, Union
import logging


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Detect and return the best available device for computation.
    
    Returns:
        torch.device: The device to use (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() 
                           if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filepath: Path to save the checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load the model on
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"Accuracy: {checkpoint.get('accuracy', 'N/A')}")
    
    return checkpoint


def save_model_state(model: nn.Module, filepath: str) -> None:
    """
    Save only the model state dictionary (for inference).
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model state
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(model.state_dict(), filepath)
    print(f"Model state saved to {filepath}")


def load_model_state(model: nn.Module, filepath: str, 
                     device: Optional[torch.device] = None) -> None:
    """
    Load model state dictionary.
    
    Args:
        model: PyTorch model to load state into
        filepath: Path to the model state file
        device: Device to load the model on
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model state file not found: {filepath}")
    
    if device is None:
        device = get_device()
    
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model state loaded from {filepath}")


def get_model_summary(model: nn.Module, 
                      input_size: tuple = (3, 224, 224)) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        String containing model summary
    """
    param_info = count_parameters(model)
    
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Input size: {input_size}")
    summary.append(f"Total parameters: {param_info['total_parameters']:,}")
    summary.append(f"Trainable parameters: "
                   f"{param_info['trainable_parameters']:,}")
    non_trainable = param_info['non_trainable_parameters']
    summary.append(f"Non-trainable parameters: {non_trainable:,}")
    summary.append("=" * 60)
    
    return "\n".join(summary)


def freeze_layers(model: nn.Module, freeze_backbone: bool = True) -> None:
    """
    Freeze or unfreeze model layers for transfer learning.
    
    Args:
        model: PyTorch model
        freeze_backbone: If True, freeze backbone layers 
                        (keep only classifier trainable)
                        If False, unfreeze all layers
    """
    for name, param in model.named_parameters():
        if freeze_backbone:
            # Stage 1: Only classifier layer (fc) is trainable
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # Stage 2: All layers are trainable
            param.requires_grad = True
    
    # Print parameter information
    param_info = count_parameters(model)
    stage = "Feature Extraction" if freeze_backbone else "Fine-tuning"
    print(f"{stage} mode:")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Total parameters: {param_info['total_parameters']:,}")


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage if CUDA is available.
    
    Returns:
        Dictionary containing memory usage information
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'max_allocated_gb': max_allocated
        }
    else:
        return {
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'max_allocated_gb': 0.0
        }


def clear_gpu_cache() -> None:
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    else:
        print("CUDA not available, no cache to clear")

# ============================================================================
# LOGGING AND MONITORING UTILITIES
# ============================================================================

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class TrainingLogger:
    """
    Structured logging for training progress and metrics.
    """
    
    def __init__(self, log_file: Optional[str] = None, 
                 console_output: bool = True):
        """
        Initialize the training logger.
        
        Args:
            log_file: Path to log file (optional)
            console_output: Whether to output to console
        """
        self.log_file = log_file
        self.console_output = console_output
        self.start_time = None
        self.epoch_start_time = None
        
        # Setup logging
        self.logger = logging.getLogger('ResNetTraining')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            ensure_dir(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_training_start(self, config: Dict[str, Any]) -> None:
        """Log training start with configuration."""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: "
                         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch end with metrics."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            metrics['epoch_time'] = epoch_time
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} completed - {metric_str}")
    
    def log_validation(self, metrics: Dict[str, float]) -> None:
        """Log validation metrics."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Validation - {metric_str}")
    
    def log_best_model(self, epoch: int, metric: str, value: float) -> None:
        """Log when a new best model is saved."""
        self.logger.info(f"💾 New best model saved at epoch {epoch}: "
                         f"{metric}={value:.4f}")
    
    def log_early_stopping(self, epoch: int, patience_counter: int, 
                           patience: int) -> None:
        """Log early stopping information."""
        self.logger.info(f"⏰ Early stopping counter: "
                         f"{patience_counter}/{patience} at epoch {epoch}")
    
    def log_early_stopping_triggered(self, epoch: int, 
                                     best_metric: float) -> None:
        """Log when early stopping is triggered."""
        self.logger.info(f"🛑 Early stopping triggered at epoch {epoch}! "
                         f"Best metric: {best_metric:.4f}")
    
    def log_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Log training completion."""
        if self.start_time:
            total_time = time.time() - self.start_time
            final_metrics['total_training_time'] = total_time
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info("Final Results:")
        for key, value in final_metrics.items():
            if 'time' in key:
                self.logger.info(f"  {key}: {format_time(value)}")
            else:
                self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("=" * 60)
    
    def log_stage_start(self, stage_name: str, 
                        stage_config: Dict[str, Any]) -> None:
        """Log training stage start (for multi-stage training)."""
        self.logger.info("-" * 40)
        self.logger.info(f"🔥 {stage_name.upper()} STARTED")
        self.logger.info("-" * 40)
        for key, value in stage_config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_stage_end(self, stage_name: str, 
                      stage_metrics: Dict[str, float]) -> None:
        """Log training stage completion."""
        self.logger.info(f"✅ {stage_name.upper()} COMPLETED")
        for key, value in stage_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("-" * 40)
    
    def log_error(self, error_msg: str, 
                  exception: Optional[Exception] = None) -> None:
        """Log error messages."""
        self.logger.error(f"❌ ERROR: {error_msg}")
        if exception:
            self.logger.error(f"Exception details: {str(exception)}")
    
    def log_warning(self, warning_msg: str) -> None:
        """Log warning messages."""
        self.logger.warning(f"⚠️  WARNING: {warning_msg}")
    
    def log_info(self, info_msg: str) -> None:
        """Log general information."""
        self.logger.info(info_msg)


class MetricsTracker:
    """
    Track and monitor training metrics over time.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = {}
        self.best_metrics = {}
        self.current_epoch = 0
    
    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Update metrics for the current epoch.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append({
                'epoch': epoch,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track best metrics (assuming higher is better for accuracy, 
            # lower for loss)
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = {
                    'value': value,
                    'epoch': epoch,
                    'is_best': True
                }
            else:
                is_better = False
                if 'loss' in metric_name.lower():
                    # For loss metrics, lower is better
                    is_better = value < self.best_metrics[metric_name]['value']
                else:
                    # For other metrics (accuracy, etc.), higher is better
                    is_better = value > self.best_metrics[metric_name]['value']
                
                if is_better:
                    self.best_metrics[metric_name] = {
                        'value': value,
                        'epoch': epoch,
                        'is_best': True
                    }
    
    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get the best value for a specific metric."""
        return self.best_metrics.get(metric_name)
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get the history for a specific metric."""
        return self.metrics_history.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest values for all metrics."""
        latest_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_metrics[metric_name] = history[-1]['value']
        return latest_metrics
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics history to JSON file."""
        ensure_dir(os.path.dirname(filepath))
        
        data = {
            'metrics_history': self.metrics_history,
            'best_metrics': self.best_metrics,
            'current_epoch': self.current_epoch,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics history from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = data.get('metrics_history', {})
        self.best_metrics = data.get('best_metrics', {})
        self.current_epoch = data.get('current_epoch', 0)
        
        print(f"Metrics loaded from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked metrics."""
        summary = {
            'total_epochs': self.current_epoch,
            'metrics_tracked': list(self.metrics_history.keys()),
            'best_metrics': {}
        }
        
        for metric_name, best_info in self.best_metrics.items():
            summary['best_metrics'][metric_name] = {
                'value': best_info['value'],
                'epoch': best_info['epoch']
            }
        
        return summary


class ProgressMonitor:
    """
    Monitor training progress and provide status updates.
    """
    
    def __init__(self, total_epochs: int, log_interval: int = 10):
        """
        Initialize progress monitor.
        
        Args:
            total_epochs: Total number of training epochs
            log_interval: Interval for logging progress updates
        """
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.start_time = None
        self.epoch_times = []
        self.current_epoch = 0
    
    def start_training(self) -> None:
        """Mark the start of training."""
        self.start_time = time.time()
        print(f"Training started for {self.total_epochs} epochs")
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Update progress for the current epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Current epoch metrics
        """
        self.current_epoch = epoch
        current_time = time.time()
        
        if self.start_time:
            elapsed_time = current_time - self.start_time
            self.epoch_times.append(elapsed_time)
            
            # Calculate ETA
            if len(self.epoch_times) > 1:
                avg_epoch_time = elapsed_time / epoch
                remaining_epochs = self.total_epochs - epoch
                eta = avg_epoch_time * remaining_epochs
                
                progress_pct = (epoch / self.total_epochs) * 100
                
                if (epoch % self.log_interval == 0 or 
                        epoch == self.total_epochs):
                    print(f"Progress: {progress_pct:.1f}% | "
                          f"Epoch {epoch}/{self.total_epochs} | "
                          f"Elapsed: {format_time(elapsed_time)} | "
                          f"ETA: {format_time(eta)}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        if not self.start_time:
            return {'status': 'not_started'}
        
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.current_epoch / self.total_epochs) * 100
        
        info = {
            'status': 'running',
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'progress_percentage': progress_pct,
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': format_time(elapsed_time)
        }
        
        if len(self.epoch_times) > 1:
            avg_epoch_time = elapsed_time / self.current_epoch
            remaining_epochs = self.total_epochs - self.current_epoch
            eta = avg_epoch_time * remaining_epochs
            
            info.update({
                'average_epoch_time': avg_epoch_time,
                'estimated_time_remaining': eta,
                'eta_formatted': format_time(eta)
            })
        
        return info


def setup_logging(log_dir: str = "logs", 
                  log_level: str = "INFO") -> TrainingLogger:
    """
    Setup logging for training with default configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured TrainingLogger instance
    """
    ensure_dir(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)
    
    logger = TrainingLogger(log_file=log_file, console_output=True)
    logger.log_info(f"Logging setup complete. Log file: {log_file}")
    
    return logger