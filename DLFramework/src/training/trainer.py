"""Training module for ResNet-50 image classification system."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging
from abc import ABC, abstractmethod

try:
    from ..config.settings import Config
    from ..utils.helpers import save_checkpoint, load_checkpoint
except ImportError:
    from config.settings import Config
    from utils.helpers import save_checkpoint, load_checkpoint


class BaseTrainer(ABC):
    """
    Base trainer class providing core training functionality.
    
    This class implements the fundamental training logic including single epoch training,
    progress tracking, and comprehensive error handling.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Config, 
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration object containing training parameters
            device: Device to run training on (CPU/GPU)
            logger: Optional logger for training progress
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or self._setup_logger()
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'learning_rates': []
        }
        
        # Validation
        self._validate_inputs()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger if none provided."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_inputs(self) -> None:
        """Validate trainer inputs and configuration."""
        if not isinstance(self.model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module")
        
        if not isinstance(self.config, Config):
            raise TypeError("Config must be a Config object")
        
        if not isinstance(self.device, torch.device):
            raise TypeError("Device must be a torch.device")
        
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for model parameters
            criterion: Loss function
            
        Returns:
            Dict containing training metrics for the epoch
            
        Raises:
            RuntimeError: If training fails
            ValueError: If inputs are invalid
        """
        try:
            self._validate_training_inputs(train_loader, optimizer, criterion)
            
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Progress bar for epoch
            pbar = tqdm(
                train_loader, 
                desc=f'Epoch {self.current_epoch + 1}',
                leave=False
            )
            
            for batch_idx, data in enumerate(pbar):
                try:
                    # Extract data
                    inputs = data["image"].to(self.device)
                    labels = data["label"].to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    current_acc = 100.0 * correct / total
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })
                    
                    # Log progress at intervals
                    if batch_idx % self.config.log_interval == 0 and batch_idx > 0:
                        avg_loss = running_loss / (batch_idx + 1)
                        self.logger.info(
                            f'Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}: '
                            f'Loss={avg_loss:.4f}, Acc={current_acc:.2f}%'
                        )
                
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    raise RuntimeError(f"Training failed at batch {batch_idx}") from e
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            
            # Log epoch results
            self.logger.info(
                f'Epoch {self.current_epoch + 1} completed: '
                f'Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%'
            )
            
            return {
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            self.logger.error(f"Training epoch failed: {str(e)}")
            raise
    
    def _validate_training_inputs(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> None:
        """Validate training inputs."""
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a DataLoader")
        
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("optimizer must be a PyTorch optimizer")
        
        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion must be a PyTorch loss function")
        
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty")
    
    @abstractmethod
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: Optional[optim.lr_scheduler._LRScheduler], 
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """
        Abstract method for full training process.
        
        Must be implemented by subclasses to define specific training strategies.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for model parameters
            scheduler: Optional learning rate scheduler
            criterion: Loss function
            
        Returns:
            Dict containing training results and history
        """
        pass
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the complete training history.
        
        Returns:
            Dict containing training metrics history
        """
        return self.training_history.copy()
    
    def reset_training_history(self) -> None:
        """Reset training history for new training session."""
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'learning_rates': []
        }
        self.current_epoch = 0
    



class Trainer(BaseTrainer):
    """
    Standard trainer implementation for basic training without early stopping.
    
    This trainer implements a straightforward training loop with validation
    after each epoch.
    """
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: Optional[optim.lr_scheduler._LRScheduler], 
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for model parameters
            scheduler: Optional learning rate scheduler
            criterion: Loss function
            
        Returns:
            Dict containing training results and history
        """
        try:
            self.logger.info(f"Starting training for {self.config.epochs} epochs")
            
            # Import validator here to avoid circular imports
            from .validator import Validator
            validator = Validator(self.model, self.device, self.logger)
            
            best_val_acc = 0.0
            
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_metrics = validator.validate(val_loader, criterion)
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                # Record metrics
                self.training_history['train_losses'].append(train_metrics['loss'])
                self.training_history['val_losses'].append(val_metrics['loss'])
                self.training_history['train_accs'].append(train_metrics['accuracy'])
                self.training_history['val_accs'].append(val_metrics['accuracy'])
                self.training_history['learning_rates'].append(current_lr)
                
                # Track best validation accuracy
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=val_metrics['loss'],
                        accuracy=val_metrics['accuracy'],
                        filepath=self.config.best_model_path,
                        additional_info={'best_val_acc': best_val_acc}
                    )
                
                # Log epoch summary
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.config.epochs}: '
                    f'LR={current_lr:.6f}, '
                    f'Train Loss={train_metrics["loss"]:.4f}, '
                    f'Train Acc={train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss={val_metrics["loss"]:.4f}, '
                    f'Val Acc={val_metrics["accuracy"]:.2f}%'
                )
            
            self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
            
            return {
                'best_val_acc': best_val_acc,
                'final_train_acc': train_metrics['accuracy'],
                'final_val_acc': val_metrics['accuracy'],
                'epochs_completed': self.config.epochs,
                'training_history': self.training_history
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


class EarlyStoppingTrainer(BaseTrainer):
    """
    Trainer with early stopping capability.
    
    This trainer monitors validation accuracy and stops training early if no
    improvement is observed for a specified number of epochs (patience).
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Config, 
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the early stopping trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration object containing training parameters
            device: Device to run training on (CPU/GPU)
            logger: Optional logger for training progress
        """
        super().__init__(model, config, device, logger)
        
        # Early stopping state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: Optional[optim.lr_scheduler._LRScheduler], 
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping based on validation accuracy.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for model parameters
            scheduler: Optional learning rate scheduler
            criterion: Loss function
            
        Returns:
            Dict containing training results and history
        """
        try:
            self.logger.info(
                f"Starting training with early stopping for max {self.config.epochs} epochs"
            )
            self.logger.info(
                f"Early stopping patience: {self.config.patience}, "
                f"min_delta: {self.config.min_delta}"
            )
            
            # Import validator here to avoid circular imports
            from .validator import Validator
            validator = Validator(self.model, self.device, self.logger)
            
            # Reset early stopping state
            self.best_val_acc = 0.0
            self.patience_counter = 0
            self.best_model_state = None
            
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_metrics = validator.validate(val_loader, criterion)
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                # Record metrics
                self.training_history['train_losses'].append(train_metrics['loss'])
                self.training_history['val_losses'].append(val_metrics['loss'])
                self.training_history['train_accs'].append(train_metrics['accuracy'])
                self.training_history['val_accs'].append(val_metrics['accuracy'])
                self.training_history['learning_rates'].append(current_lr)
                
                # Early stopping logic
                current_val_acc = val_metrics['accuracy']
                improvement = current_val_acc - self.best_val_acc
                
                if improvement > self.config.min_delta:
                    # Improvement found
                    self.best_val_acc = current_val_acc
                    self.patience_counter = 0
                    
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                    save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=val_metrics['loss'],
                        accuracy=self.best_val_acc,
                        filepath=self.config.best_model_path,
                        additional_info={
                            'best_val_acc': self.best_val_acc,
                            'patience_counter': self.patience_counter
                        }
                    )
                    
                    self.logger.info(
                        f"💾 New best model saved! Validation accuracy: {self.best_val_acc:.2f}% "
                        f"(improvement: +{improvement:.3f}%)"
                    )
                else:
                    # No improvement
                    self.patience_counter += 1
                    self.logger.info(
                        f"⏰ No improvement. Patience counter: {self.patience_counter}/{self.config.patience}"
                    )
                
                # Log epoch summary
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.config.epochs}: '
                    f'LR={current_lr:.6f}, '
                    f'Train Loss={train_metrics["loss"]:.4f}, '
                    f'Train Acc={train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss={val_metrics["loss"]:.4f}, '
                    f'Val Acc={val_metrics["accuracy"]:.2f}%'
                )
                
                # Check early stopping condition
                if self.patience_counter >= self.config.patience:
                    self.logger.info(
                        f"🛑 Early stopping triggered! Best validation accuracy: {self.best_val_acc:.2f}%"
                    )
                    break
            
            # Load best model if early stopping occurred
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                self.logger.info("Restored best model weights")
            
            epochs_completed = epoch + 1
            self.logger.info(
                f"Training completed after {epochs_completed} epochs. "
                f"Best validation accuracy: {self.best_val_acc:.2f}%"
            )
            
            return {
                'best_val_acc': self.best_val_acc,
                'final_train_acc': train_metrics['accuracy'],
                'final_val_acc': val_metrics['accuracy'],
                'epochs_completed': epochs_completed,
                'early_stopped': self.patience_counter >= self.config.patience,
                'patience_counter': self.patience_counter,
                'training_history': self.training_history
            }
            
        except Exception as e:
            self.logger.error(f"Early stopping training failed: {str(e)}")
            raise
    
    def get_best_model_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the state dict of the best model found during training.
        
        Returns:
            Best model state dict or None if no best model saved
        """
        return self.best_model_state
    
    def reset_early_stopping_state(self) -> None:
        """Reset early stopping state for new training session."""
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        self.reset_training_history()


class TwoStageTrainer(BaseTrainer):
    """
    Two-stage trainer for feature extraction and fine-tuning phases.
    
    Stage 1: Feature extraction - freeze backbone, train only classifier
    Stage 2: Fine-tuning - unfreeze all layers, train with lower learning rate
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Config, 
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the two-stage trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration object containing training parameters
            device: Device to run training on (CPU/GPU)
            logger: Optional logger for training progress
        """
        super().__init__(model, config, device, logger)
        
        # Two-stage training state
        self.stage1_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'learning_rates': []
        }
        self.stage2_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'learning_rates': []
        }
        self.stage1_best_acc = 0.0
        self.stage2_best_acc = 0.0
        
    def freeze_model_layers(self, freeze_backbone: bool = True) -> None:
        """
        Freeze or unfreeze model layers for different training stages.
        
        Args:
            freeze_backbone: If True, freeze backbone (stage 1), 
                           if False, unfreeze all layers (stage 2)
        """
        trainable_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            if freeze_backbone:
                # Stage 1: Only classifier (fc) layer is trainable
                if 'fc' in name or 'classifier' in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            else:
                # Stage 2: All layers are trainable
                param.requires_grad = True
                trainable_params += param.numel()
        
        stage = "Stage 1 (Feature Extraction)" if freeze_backbone else "Stage 2 (Fine-tuning)"
        self.logger.info(
            f"{stage}: Trainable parameters: {trainable_params:,} / "
            f"Total parameters: {total_params:,} "
            f"({100.0 * trainable_params / total_params:.1f}%)"
        )
    
    def _create_stage_optimizer(self, stage: int) -> optim.Optimizer:
        """
        Create optimizer for specific training stage.
        
        Args:
            stage: Training stage (1 or 2)
            
        Returns:
            Configured optimizer for the stage
        """
        if stage == 1:
            lr = self.config.stage1_lr
            betas = self.config.stage1_betas
        elif stage == 2:
            lr = self.config.stage2_lr
            betas = self.config.stage2_betas
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
        
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer = optim.Adam(trainable_params, lr=lr, betas=betas)
        
        self.logger.info(f"Stage {stage} optimizer created with lr={lr}, betas={betas}")
        return optimizer
    
    def _create_stage_scheduler(
        self, 
        optimizer: optim.Optimizer, 
        stage: int
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler for specific training stage.
        
        Args:
            optimizer: Optimizer to schedule
            stage: Training stage (1 or 2)
            
        Returns:
            Configured scheduler for the stage
        """
        if stage == 1:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config.stage1_step_size, 
                gamma=self.config.stage1_gamma
            )
            self.logger.info(
                f"Stage 1 scheduler: StepLR(step_size={self.config.stage1_step_size}, "
                f"gamma={self.config.stage1_gamma})"
            )
        elif stage == 2:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.stage2_t0, 
                T_mult=self.config.stage2_t_mult
            )
            self.logger.info(
                f"Stage 2 scheduler: CosineAnnealingWarmRestarts(T_0={self.config.stage2_t0}, "
                f"T_mult={self.config.stage2_t_mult})"
            )
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
        
        return scheduler
    
    def _train_stage(
        self,
        stage: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """
        Train a specific stage with early stopping.
        
        Args:
            stage: Training stage (1 or 2)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Dict containing stage training results
        """
        # Configure model for stage
        if stage == 1:
            self.freeze_model_layers(freeze_backbone=True)
            epochs = self.config.stage1_epochs
            patience = self.config.stage1_patience
            stage_history = self.stage1_history
        else:
            self.freeze_model_layers(freeze_backbone=False)
            epochs = self.config.stage2_epochs
            patience = self.config.stage2_patience
            stage_history = self.stage2_history
        
        # Create stage-specific optimizer and scheduler
        optimizer = self._create_stage_optimizer(stage)
        scheduler = self._create_stage_scheduler(optimizer, stage)
        
        # Import validator here to avoid circular imports
        from .validator import Validator
        validator = Validator(self.model, self.device, self.logger)
        
        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        self.logger.info(f"🔥 Starting Stage {stage} training")
        self.logger.info(f"Epochs: {epochs}, Patience: {patience}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics = validator.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Record metrics
            stage_history['train_losses'].append(train_metrics['loss'])
            stage_history['val_losses'].append(val_metrics['loss'])
            stage_history['train_accs'].append(train_metrics['accuracy'])
            stage_history['val_accs'].append(val_metrics['accuracy'])
            stage_history['learning_rates'].append(current_lr)
            
            # Early stopping logic
            current_val_acc = val_metrics['accuracy']
            improvement = current_val_acc - best_val_acc
            
            if improvement > self.config.min_delta:
                # Improvement found
                best_val_acc = current_val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                # Save stage-specific checkpoint
                checkpoint_path = f"stage{stage}_best_model.pth"
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    accuracy=best_val_acc,
                    filepath=checkpoint_path,
                    additional_info={
                        'stage': stage,
                        'best_val_acc': best_val_acc
                    }
                )
                
                self.logger.info(
                    f"💾 Stage {stage} - New best model saved! "
                    f"Validation accuracy: {best_val_acc:.2f}% "
                    f"(improvement: +{improvement:.3f}%)"
                )
            else:
                # No improvement
                patience_counter += 1
                self.logger.info(
                    f"⏰ Stage {stage} - No improvement. "
                    f"Patience counter: {patience_counter}/{patience}"
                )
            
            # Log epoch summary
            self.logger.info(
                f'Stage {stage} - Epoch {epoch + 1}/{epochs}: '
                f'LR={current_lr:.6f}, '
                f'Train Loss={train_metrics["loss"]:.4f}, '
                f'Train Acc={train_metrics["accuracy"]:.2f}%, '
                f'Val Loss={val_metrics["loss"]:.4f}, '
                f'Val Acc={val_metrics["accuracy"]:.2f}%'
            )
            
            # Check early stopping condition
            if patience_counter >= patience:
                self.logger.info(
                    f"🛑 Stage {stage} early stopping triggered! "
                    f"Best validation accuracy: {best_val_acc:.2f}%"
                )
                break
        
        # Load best model for this stage
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored Stage {stage} best model weights")
        
        epochs_completed = epoch + 1
        self.logger.info(
            f"✅ Stage {stage} completed after {epochs_completed} epochs. "
            f"Best validation accuracy: {best_val_acc:.2f}%"
        )
        
        # Store stage best accuracy
        if stage == 1:
            self.stage1_best_acc = best_val_acc
        else:
            self.stage2_best_acc = best_val_acc
        
        return {
            'best_val_acc': best_val_acc,
            'epochs_completed': epochs_completed,
            'early_stopped': patience_counter >= patience,
            'final_train_acc': train_metrics['accuracy'],
            'final_val_acc': val_metrics['accuracy']
        }
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: Optional[optim.lr_scheduler._LRScheduler], 
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """
        Execute two-stage training process.
        
        Note: The optimizer and scheduler parameters are ignored as this trainer
        creates stage-specific optimizers and schedulers.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Ignored - stage-specific optimizers are created
            scheduler: Ignored - stage-specific schedulers are created
            criterion: Loss function
            
        Returns:
            Dict containing complete two-stage training results
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("🚀 Starting Two-Stage Training")
            self.logger.info("=" * 60)
            
            # Reset training histories
            self.stage1_history = {
                'train_losses': [], 'val_losses': [], 'train_accs': [], 
                'val_accs': [], 'learning_rates': []
            }
            self.stage2_history = {
                'train_losses': [], 'val_losses': [], 'train_accs': [], 
                'val_accs': [], 'learning_rates': []
            }
            
            # Stage 1: Feature extraction
            stage1_results = self._train_stage(1, train_loader, val_loader, criterion)
            
            # Stage 2: Fine-tuning
            stage2_results = self._train_stage(2, train_loader, val_loader, criterion)
            
            # Save final best model
            final_checkpoint_path = self.config.final_model_path
            save_checkpoint(
                model=self.model,
                optimizer=None,  # No optimizer needed for final save
                scheduler=None,  # No scheduler needed for final save
                epoch=0,  # Combined epochs from both stages
                loss=0.0,  # Not applicable for final save
                accuracy=self.stage2_best_acc,
                filepath=final_checkpoint_path,
                additional_info={
                    'stage1_best_acc': self.stage1_best_acc,
                    'stage2_best_acc': self.stage2_best_acc,
                    'stage1_history': self.stage1_history,
                    'stage2_history': self.stage2_history
                }
            )
            
            # Calculate improvement
            improvement = self.stage2_best_acc - self.stage1_best_acc
            
            # Combine training histories
            combined_history = {
                'train_losses': self.stage1_history['train_losses'] + self.stage2_history['train_losses'],
                'val_losses': self.stage1_history['val_losses'] + self.stage2_history['val_losses'],
                'train_accs': self.stage1_history['train_accs'] + self.stage2_history['train_accs'],
                'val_accs': self.stage1_history['val_accs'] + self.stage2_history['val_accs'],
                'learning_rates': self.stage1_history['learning_rates'] + self.stage2_history['learning_rates']
            }
            
            self.training_history = combined_history
            
            # Final summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎉 Two-Stage Training Complete!")
            self.logger.info("=" * 60)
            self.logger.info(f"Stage 1 (Feature Extraction) Best Accuracy: {self.stage1_best_acc:.2f}%")
            self.logger.info(f"Stage 2 (Fine-tuning) Best Accuracy: {self.stage2_best_acc:.2f}%")
            self.logger.info(f"Total Improvement: {improvement:+.2f}%")
            self.logger.info(f"Final model saved to: {final_checkpoint_path}")
            
            return {
                'stage1_best_acc': self.stage1_best_acc,
                'stage2_best_acc': self.stage2_best_acc,
                'total_improvement': improvement,
                'stage1_results': stage1_results,
                'stage2_results': stage2_results,
                'stage1_history': self.stage1_history,
                'stage2_history': self.stage2_history,
                'combined_history': combined_history,
                'training_history': combined_history,
                'best_val_acc': self.stage2_best_acc,
                'epochs_completed': stage1_results['epochs_completed'] + stage2_results['epochs_completed']
            }
            
        except Exception as e:
            self.logger.error(f"Two-stage training failed: {str(e)}")
            raise
    
    def get_stage_history(self, stage: int) -> Dict[str, Any]:
        """
        Get training history for a specific stage.
        
        Args:
            stage: Stage number (1 or 2)
            
        Returns:
            Training history for the specified stage
        """
        if stage == 1:
            return self.stage1_history.copy()
        elif stage == 2:
            return self.stage2_history.copy()
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    def reset_two_stage_state(self) -> None:
        """Reset two-stage training state for new training session."""
        self.stage1_history = {
            'train_losses': [], 'val_losses': [], 'train_accs': [], 
            'val_accs': [], 'learning_rates': []
        }
        self.stage2_history = {
            'train_losses': [], 'val_losses': [], 'train_accs': [], 
            'val_accs': [], 'learning_rates': []
        }
        self.stage1_best_acc = 0.0
        self.stage2_best_acc = 0.0
        self.reset_training_history()