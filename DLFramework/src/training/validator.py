"""Validation module for ResNet-50 image classification system."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_fscore_support)


class Validator:
    """
    Comprehensive validator class for model evaluation.
    
    Provides basic validation functionality with loss and accuracy calculation,
    batch processing, progress tracking, and comprehensive error handling.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the validator.
        
        Args:
            model: PyTorch model to validate
            device: Device to run validation on (CPU/GPU)
            logger: Optional logger for validation progress
        """
        self.model = model
        self.device = device
        self.logger = logger or self._setup_logger()
    
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
    
    def validate(self, val_loader: DataLoader, 
                 criterion: nn.Module) -> Dict[str, float]:
        """
        Validate the model on validation data with basic metrics.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Dict containing validation metrics (loss, accuracy, correct, total)
            
        Raises:
            RuntimeError: If validation fails due to model or data issues
            ValueError: If inputs are invalid
        """
        if not val_loader:
            raise ValueError("Validation DataLoader cannot be None or empty")
        
        if criterion is None:
            raise ValueError("Criterion cannot be None")
        
        try:
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            self.logger.info("Starting validation...")
            
            with torch.no_grad():
                # Use tqdm for progress tracking
                pbar = tqdm(val_loader, desc='Validating', leave=False)
                
                for batch_idx, data in enumerate(pbar):
                    try:
                        # Handle different data formats
                        if isinstance(data, dict):
                            inputs = data["image"].to(self.device)
                            labels = data["label"].to(self.device)
                        else:
                            inputs, labels = data
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Accumulate metrics
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # Update progress bar
                        current_acc = (100.0 * correct / total 
                                      if total > 0 else 0.0)
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Acc': f'{current_acc:.2f}%'
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_idx}: "
                                          f"{str(e)}")
                        raise RuntimeError(f"Validation failed at batch "
                                          f"{batch_idx}: {str(e)}")
            
            # Calculate final metrics
            if total == 0:
                raise RuntimeError("No valid samples processed "
                                  "during validation")
            
            accuracy = 100.0 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            
            self.logger.info(f"Validation complete - Loss: {avg_val_loss:.4f}, "
                             f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
            
            return {
                'loss': avg_val_loss,
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
    
    def evaluate_detailed(
        self, 
        test_loader: DataLoader, 
        criterion: nn.Module,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with detailed 
        classification metrics.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function
            class_names: Optional list of class names for reporting
            
        Returns:
            Dict containing comprehensive evaluation metrics
            
        Raises:
            RuntimeError: If evaluation fails
            ValueError: If inputs are invalid
        """
        if not test_loader:
            raise ValueError("Test DataLoader cannot be None or empty")
        
        if criterion is None:
            raise ValueError("Criterion cannot be None")
        
        try:
            self.model.eval()
            all_predictions = []
            all_labels = []
            all_probabilities = []
            test_loss = 0.0
            
            self.logger.info("Starting detailed evaluation...")
            
            with torch.no_grad():
                pbar = tqdm(test_loader, desc='Evaluating', leave=False)
                
                for batch_idx, data in enumerate(pbar):
                    try:
                        # Handle different data formats
                        if isinstance(data, dict):
                            inputs = data["image"].to(self.device)
                            labels = data["label"].to(self.device)
                        else:
                            inputs, labels = data
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        
                        # Get predictions and probabilities
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        
                        # Store results
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())
                        
                        # Update progress
                        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                        
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_idx}: "
                                          f"{str(e)}")
                        raise RuntimeError(f"Evaluation failed at batch "
                                          f"{batch_idx}: {str(e)}")
            
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_probabilities = np.array(all_probabilities)
            
            # Calculate basic metrics
            accuracy = (100.0 * np.sum(all_predictions == all_labels) / 
                       len(all_labels))
            avg_test_loss = test_loss / len(test_loader)
            
            # Calculate detailed metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            
            # Generate classification report
            if class_names is not None:
                # Ensure we only use class names for classes that appear in the data
                unique_labels = sorted(set(all_labels))
                used_class_names = [
                    class_names[i] if i < len(class_names) else f"Class_{i}" 
                    for i in unique_labels
                ]
                class_report = classification_report(
                    all_labels, all_predictions, 
                    target_names=used_class_names,
                    output_dict=True,
                    zero_division=0
                )
            else:
                class_report = classification_report(
                    all_labels, all_predictions, 
                    output_dict=True,
                    zero_division=0
                )
            
            # Calculate per-class accuracy
            per_class_accuracy = {}
            unique_labels = sorted(set(all_labels))
            for label in unique_labels:
                mask = all_labels == label
                if np.sum(mask) > 0:
                    class_acc = (np.sum(all_predictions[mask] == label) / 
                                 np.sum(mask))
                    class_name = (class_names[label] 
                                 if class_names and label < len(class_names) 
                                 else f"Class_{label}")
                    per_class_accuracy[class_name] = class_acc * 100.0
            
            # Log results
            self.logger.info(f"Detailed evaluation complete:")
            self.logger.info(f"  Accuracy: {accuracy:.2f}%")
            self.logger.info(f"  Average Loss: {avg_test_loss:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1-Score: {f1:.4f}")
            self.logger.info(f"  Total Samples: {len(all_labels)}")
            
            return {
                'accuracy': accuracy,
                'test_loss': avg_test_loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': all_predictions.tolist(),
                'true_labels': all_labels.tolist(),
                'probabilities': all_probabilities.tolist(),
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'per_class_accuracy': per_class_accuracy,
                'total_samples': len(all_labels)
            }
            
        except Exception as e:
            self.logger.error(f"Detailed evaluation failed: {str(e)}")
            raise
    
    def get_predictions(
        self, 
        data_loader: DataLoader,
        return_probabilities: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get model predictions for a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
            probabilities is None if return_probabilities is False
            
        Raises:
            RuntimeError: If prediction fails
            ValueError: If inputs are invalid
        """
        if not data_loader:
            raise ValueError("DataLoader cannot be None or empty")
        
        try:
            self.model.eval()
            all_predictions = []
            all_labels = []
            all_probabilities = [] if return_probabilities else None
            
            with torch.no_grad():
                for data in tqdm(data_loader, desc='Getting predictions', leave=False):
                    # Handle different data formats
                    if isinstance(data, dict):
                        inputs = data["image"].to(self.device)
                        labels = data["label"].to(self.device)
                    else:
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    if return_probabilities:
                        probabilities = torch.softmax(outputs, dim=1)
                        all_probabilities.extend(probabilities.cpu().numpy())
                    
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            predictions = np.array(all_predictions)
            true_labels = np.array(all_labels)
            probabilities = np.array(all_probabilities) if return_probabilities else None
            
            return predictions, true_labels, probabilities
            
        except Exception as e:
            self.logger.error(f"Getting predictions failed: {str(e)}")
            raise RuntimeError(f"Failed to get predictions: {str(e)}")
    
    def calculate_class_metrics(
        self, 
        true_labels: np.ndarray, 
        predictions: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            true_labels: Array of true labels
            predictions: Array of predicted labels
            class_names: Optional list of class names
            
        Returns:
            Dict mapping class names to their metrics
        """
        try:
            unique_labels = sorted(set(true_labels))
            class_metrics = {}
            
            for label in unique_labels:
                # Get class name
                class_name = class_names[label] if class_names and label < len(class_names) else f"Class_{label}"
                
                # Calculate metrics for this class
                mask = true_labels == label
                if np.sum(mask) > 0:
                    # True positives, false positives, false negatives
                    tp = np.sum((predictions == label) & (true_labels == label))
                    fp = np.sum((predictions == label) & (true_labels != label))
                    fn = np.sum((predictions != label) & (true_labels == label))
                    
                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    accuracy = tp / np.sum(mask)
                    
                    class_metrics[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': accuracy * 100.0,
                        'support': int(np.sum(mask)),
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'false_negatives': int(fn)
                    }
            
            return class_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate class metrics: {str(e)}")
            raise