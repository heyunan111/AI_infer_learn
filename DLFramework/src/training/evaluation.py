"""Comprehensive evaluation utilities for ResNet-50 image classification system."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score,
    top_k_accuracy_score
)
from .validator import Validator


class ModelEvaluator:
    """
    High-level model evaluation class providing comprehensive evaluation capabilities.
    
    This class extends the basic Validator functionality with additional evaluation
    methods, reporting capabilities, and analysis tools.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on (CPU/GPU)
            logger: Optional logger for evaluation progress
        """
        self.model = model
        self.device = device
        self.logger = logger or self._setup_logger()
        self.validator = Validator(model, device, logger)
    
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
    
    def evaluate_model(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        class_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation matching the original script's evaluate_model function.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function
            class_names: Optional list of class names for reporting
            verbose: Whether to print detailed results
            
        Returns:
            Dict containing comprehensive evaluation results
        """
        try:
            self.logger.info("Starting comprehensive model evaluation...")
            
            # Get detailed evaluation results
            results = self.validator.evaluate_detailed(test_loader, criterion, class_names)
            
            # Extract key metrics
            accuracy = results['accuracy']
            test_loss = results['test_loss']
            predictions = np.array(results['predictions'])
            true_labels = np.array(results['true_labels'])
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"MODEL EVALUATION RESULTS")
                print(f"{'='*60}")
                print(f"Test Accuracy: {accuracy:.2f}%")
                print(f"Average Test Loss: {test_loss:.4f}")
                print(f"Total Samples: {results['total_samples']}")
                print(f"Precision (weighted): {results['precision']:.4f}")
                print(f"Recall (weighted): {results['recall']:.4f}")
                print(f"F1-Score (weighted): {results['f1_score']:.4f}")
                
                # Print per-class accuracy
                if results['per_class_accuracy']:
                    print(f"\nPer-Class Accuracy:")
                    print(f"{'-'*40}")
                    for class_name, acc in results['per_class_accuracy'].items():
                        print(f"{class_name}: {acc:.2f}%")
                
                # Print classification report
                print(f"\nDetailed Classification Report:")
                print(f"{'-'*60}")
                if class_names:
                    unique_labels = sorted(set(true_labels))
                    used_class_names = [class_names[i] if i < len(class_names) else f"Class_{i}" 
                                      for i in unique_labels]
                    print(classification_report(
                        true_labels, predictions, 
                        target_names=used_class_names,
                        zero_division=0
                    ))
                else:
                    print(classification_report(true_labels, predictions, zero_division=0))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def evaluate_with_top_k(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        k_values: List[int] = [1, 3, 5],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model with top-k accuracy metrics.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function
            k_values: List of k values for top-k accuracy
            class_names: Optional list of class names
            
        Returns:
            Dict containing evaluation results with top-k accuracies
        """
        try:
            self.logger.info(f"Evaluating with top-k accuracy for k={k_values}")
            
            # Get predictions and probabilities
            predictions, true_labels, probabilities = self.validator.get_predictions(
                test_loader, return_probabilities=True
            )
            
            # Calculate basic metrics
            test_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for data in test_loader:
                    if isinstance(data, dict):
                        inputs = data["image"].to(self.device)
                        labels = data["label"].to(self.device)
                    else:
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            
            # Calculate top-k accuracies
            top_k_accuracies = {}
            for k in k_values:
                if k <= probabilities.shape[1]:  # Ensure k doesn't exceed number of classes
                    top_k_acc = top_k_accuracy_score(true_labels, probabilities, k=k)
                    top_k_accuracies[f'top_{k}_accuracy'] = top_k_acc * 100.0
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(true_labels, predictions)
            
            results = {
                'test_loss': avg_test_loss,
                'predictions': predictions.tolist(),
                'true_labels': true_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'confusion_matrix': conf_matrix.tolist(),
                'total_samples': len(true_labels),
                **top_k_accuracies
            }
            
            # Log results
            self.logger.info(f"Top-k evaluation complete:")
            self.logger.info(f"  Average Loss: {avg_test_loss:.4f}")
            for k, acc in top_k_accuracies.items():
                self.logger.info(f"  {k.replace('_', '-').title()}: {acc:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Top-k evaluation failed: {str(e)}")
            raise
    
    def analyze_misclassifications(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze model misclassifications to identify common error patterns.
        
        Args:
            test_loader: DataLoader for test data
            class_names: Optional list of class names
            top_n: Number of top misclassification pairs to return
            
        Returns:
            Dict containing misclassification analysis
        """
        try:
            self.logger.info("Analyzing misclassifications...")
            
            # Get predictions
            predictions, true_labels, probabilities = self.validator.get_predictions(
                test_loader, return_probabilities=True
            )
            
            # Find misclassified samples
            misclassified_mask = predictions != true_labels
            misclassified_indices = np.where(misclassified_mask)[0]
            
            if len(misclassified_indices) == 0:
                return {
                    'total_misclassified': 0,
                    'misclassification_rate': 0.0,
                    'top_confusion_pairs': [],
                    'class_error_rates': {}
                }
            
            # Calculate misclassification rate
            misclassification_rate = len(misclassified_indices) / len(true_labels) * 100.0
            
            # Analyze confusion pairs
            confusion_pairs = {}
            for idx in misclassified_indices:
                true_label = true_labels[idx]
                pred_label = predictions[idx]
                
                true_name = class_names[true_label] if class_names and true_label < len(class_names) else f"Class_{true_label}"
                pred_name = class_names[pred_label] if class_names and pred_label < len(class_names) else f"Class_{pred_label}"
                
                pair = (true_name, pred_name)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            # Get top confusion pairs
            top_confusion_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Calculate per-class error rates
            class_error_rates = {}
            unique_labels = sorted(set(true_labels))
            for label in unique_labels:
                class_mask = true_labels == label
                class_misclassified = np.sum(misclassified_mask & class_mask)
                class_total = np.sum(class_mask)
                error_rate = class_misclassified / class_total * 100.0 if class_total > 0 else 0.0
                
                class_name = class_names[label] if class_names and label < len(class_names) else f"Class_{label}"
                class_error_rates[class_name] = {
                    'error_rate': error_rate,
                    'misclassified': int(class_misclassified),
                    'total': int(class_total)
                }
            
            results = {
                'total_misclassified': len(misclassified_indices),
                'misclassification_rate': misclassification_rate,
                'top_confusion_pairs': [
                    {
                        'true_class': pair[0],
                        'predicted_class': pair[1],
                        'count': count,
                        'percentage': count / len(misclassified_indices) * 100.0
                    }
                    for (pair, count) in top_confusion_pairs
                ],
                'class_error_rates': class_error_rates,
                'misclassified_indices': misclassified_indices.tolist()
            }
            
            self.logger.info(f"Misclassification analysis complete:")
            self.logger.info(f"  Total misclassified: {len(misclassified_indices)}")
            self.logger.info(f"  Misclassification rate: {misclassification_rate:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Misclassification analysis failed: {str(e)}")
            raise
    
    def generate_evaluation_report(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        class_names: Optional[List[str]] = None,
        include_misclassification_analysis: bool = True,
        include_top_k: bool = True,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function
            class_names: Optional list of class names
            include_misclassification_analysis: Whether to include misclassification analysis
            include_top_k: Whether to include top-k accuracy
            k_values: List of k values for top-k accuracy
            
        Returns:
            Dict containing comprehensive evaluation report
        """
        try:
            self.logger.info("Generating comprehensive evaluation report...")
            
            report = {}
            
            # Basic evaluation
            basic_results = self.evaluate_model(test_loader, criterion, class_names, verbose=False)
            report['basic_metrics'] = basic_results
            
            # Top-k evaluation
            if include_top_k:
                top_k_results = self.evaluate_with_top_k(test_loader, criterion, k_values, class_names)
                report['top_k_metrics'] = top_k_results
            
            # Misclassification analysis
            if include_misclassification_analysis:
                misclass_results = self.analyze_misclassifications(test_loader, class_names)
                report['misclassification_analysis'] = misclass_results
            
            # Summary statistics
            report['summary'] = {
                'total_samples': basic_results['total_samples'],
                'accuracy': basic_results['accuracy'],
                'test_loss': basic_results['test_loss'],
                'precision': basic_results['precision'],
                'recall': basic_results['recall'],
                'f1_score': basic_results['f1_score'],
                'num_classes': len(set(basic_results['true_labels']))
            }
            
            self.logger.info("Comprehensive evaluation report generated successfully")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation report: {str(e)}")
            raise


def evaluate_model_simple(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Simple function to evaluate a model (matches original script interface).
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        class_names: Optional list of class names
        
    Returns:
        Dict containing evaluation results
    """
    evaluator = ModelEvaluator(model, device)
    return evaluator.evaluate_model(test_loader, criterion, class_names)


def get_model_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    return_probabilities: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Get model predictions for a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run on
        return_probabilities: Whether to return prediction probabilities
        
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    validator = Validator(model, device)
    return validator.get_predictions(data_loader, return_probabilities)