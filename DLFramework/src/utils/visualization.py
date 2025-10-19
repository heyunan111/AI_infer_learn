"""
Visualization utilities for training history and model evaluation.

This module provides functions for plotting training metrics, confusion matrices,
and other visualizations to help analyze model performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any
import os
from sklearn.metrics import confusion_matrix


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot training history with improved layout and comprehensive metrics.
    
    Args:
        history: Dictionary containing training metrics with keys:
                - 'train_losses': List of training losses per epoch
                - 'val_losses': List of validation losses per epoch  
                - 'train_accs': List of training accuracies per epoch
                - 'val_accs': List of validation accuracies per epoch
        save_path: Optional path to save the plot. If None, saves as 
                  'training_history.png'
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
    """
    if save_path is None:
        save_path = 'training_history.png'
    
    # Create subplots with improved layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_losses'], label='Training Loss', 
             color='#2E86AB', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, history['val_losses'], label='Validation Loss', 
             color='#A23B72', linewidth=2, marker='s', markersize=3)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, history['train_accs'], label='Training Accuracy', 
             color='#2E86AB', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, history['val_accs'], label='Validation Accuracy', 
             color='#A23B72', linewidth=2, marker='s', markersize=3)
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    
    # Plot 3: Loss difference (overfitting indicator)
    loss_diff = [abs(t - v) for t, v in zip(history['train_losses'], 
                                           history['val_losses'])]
    ax3.plot(epochs, loss_diff, color='#F18F01', linewidth=2, 
             marker='^', markersize=3)
    ax3.set_title('Train-Validation Loss Difference', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(epochs))
    
    # Plot 4: Accuracy difference (generalization gap)
    acc_diff = [abs(t - v) for t, v in zip(history['train_accs'], 
                                          history['val_accs'])]
    ax4.plot(epochs, acc_diff, color='#C73E1D', linewidth=2, 
             marker='d', markersize=3)
    ax4.set_title('Train-Validation Accuracy Difference', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, len(epochs))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Training history plot saved to: {save_path}")


def plot_loss_and_accuracy(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot training and validation loss and accuracy in a simplified 
    2-panel layout.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
    """
    if save_path is None:
        save_path = 'loss_accuracy_curves.png'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], label='Training', 
             color='#2E86AB', linewidth=2)
    ax1.plot(epochs, history['val_losses'], label='Validation', 
             color='#A23B72', linewidth=2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accs'], label='Training', 
             color='#2E86AB', linewidth=2)
    ax2.plot(epochs, history['val_accs'], label='Validation', 
             color='#A23B72', linewidth=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Loss and accuracy curves saved to: {save_path}")


def plot_training_comparison(
    histories: Dict[str, Dict[str, List[float]]], 
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (12, 8)
) -> None:
    """
    Compare multiple training runs or stages.
    
    Args:
        histories: Dictionary of training histories, e.g., 
                  {'Stage 1': history1, 'Stage 2': history2}
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
    """
    if save_path is None:
        save_path = 'training_comparison.png'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3D5A80']
    
    for i, (name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Training loss
        ax1.plot(epochs, history['train_losses'], label=f'{name}', 
                color=color, linewidth=2)
        
        # Validation loss
        ax2.plot(epochs, history['val_losses'], label=f'{name}', 
                color=color, linewidth=2)
        
        # Training accuracy
        ax3.plot(epochs, history['train_accs'], label=f'{name}', 
                color=color, linewidth=2)
        
        # Validation accuracy
        ax4.plot(epochs, history['val_accs'], label=f'{name}', 
                color=color, linewidth=2)
    
    # Configure subplots
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Training comparison plot saved to: {save_path}")

def plot_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (10, 8),
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix with improved visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for labeling
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
        normalize: Whether to normalize the confusion matrix
    """
    if save_path is None:
        save_path = 'confusion_matrix.png'
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Use seaborn for better visualization
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_class_performance(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (12, 8),
    top_n: int = 20
) -> None:
    """
    Plot per-class performance metrics.
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
        top_n: Number of top/bottom classes to show
    """
    if save_path is None:
        save_path = 'class_performance.png'
    
    # Extract per-class metrics (assuming sklearn classification_report format)
    if 'classification_report' not in evaluation_results:
        print("Warning: No classification report found in evaluation results")
        return
    
    report = evaluation_results['classification_report']
    
    # Extract class names and metrics
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            classes.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
    
    if not classes:
        print("Warning: No class-level metrics found")
        return
    
    # Sort by F1-score and take top/bottom classes
    sorted_indices = np.argsort(f1_scores)
    
    # Take bottom and top classes
    n_classes = len(classes)
    if n_classes > top_n:
        bottom_indices = sorted_indices[:top_n//2]
        top_indices = sorted_indices[-top_n//2:]
        selected_indices = np.concatenate([bottom_indices, top_indices])
    else:
        selected_indices = sorted_indices
    
    selected_classes = [classes[i] for i in selected_indices]
    selected_precisions = [precisions[i] for i in selected_indices]
    selected_recalls = [recalls[i] for i in selected_indices]
    selected_f1s = [f1_scores[i] for i in selected_indices]
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, 
                 fontweight='bold')
    
    x_pos = np.arange(len(selected_classes))
    
    # Precision
    bars1 = ax1.bar(x_pos, selected_precisions, color='#2E86AB', alpha=0.7)
    ax1.set_title('Precision by Class')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Recall
    bars2 = ax2.bar(x_pos, selected_recalls, color='#A23B72', alpha=0.7)
    ax2.set_title('Recall by Class')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # F1-Score
    bars3 = ax3.bar(x_pos, selected_f1s, color='#F18F01', alpha=0.7)
    ax3.set_title('F1-Score by Class')
    ax3.set_ylabel('F1-Score')
    ax3.set_xlabel('Class')
    ax3.set_ylim(0, 1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(selected_classes, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Class performance plot saved to: {save_path}")


def plot_model_summary(
    evaluation_results: Dict[str, Any],
    training_history: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (15, 10)
) -> None:
    """
    Create a comprehensive model performance summary visualization.
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        training_history: Optional training history for additional context
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height)
    """
    if save_path is None:
        save_path = 'model_summary.png'
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Model Performance Summary', fontsize=18, fontweight='bold')
    
    # 1. Overall metrics (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        evaluation_results.get('accuracy', 0),
        evaluation_results.get('macro_precision', 0),
        evaluation_results.get('macro_recall', 0),
        evaluation_results.get('macro_f1', 0)
    ]
    
    bars = ax1.bar(metrics, values, 
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_title('Overall Metrics', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training history (if available) - top-middle and top-right
    if training_history:
        ax2 = fig.add_subplot(gs[0, 1])
        epochs = range(1, len(training_history['train_losses']) + 1)
        ax2.plot(epochs, training_history['train_losses'], 
                 label='Train', color='#2E86AB')
        ax2.plot(epochs, training_history['val_losses'], 
                 label='Val', color='#A23B72')
        ax2.set_title('Loss Curves', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, training_history['train_accs'], 
                 label='Train', color='#2E86AB')
        ax3.plot(epochs, training_history['val_accs'], 
                 label='Val', color='#A23B72')
        ax3.set_title('Accuracy Curves', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 3. Confusion matrix (bottom-left, spans 2 columns)
    if 'predictions' in evaluation_results and 'true_labels' in evaluation_results:
        ax4 = fig.add_subplot(gs[1:, :2])
        
        y_true = evaluation_results['true_labels']
        y_pred = evaluation_results['predictions']
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax4.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax4.set_title('Confusion Matrix (Normalized)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Proportion', rotation=270, labelpad=15)
        
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
    
    # 4. Key statistics (bottom-right)
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.axis('off')
    
    # Create text summary
    stats_text = f"""
    Model Performance Statistics
    
    Total Samples: {evaluation_results.get('total_samples', 'N/A')}
    Correct Predictions: {evaluation_results.get('correct_predictions', 'N/A')}
    
    Best Validation Accuracy: {evaluation_results.get('best_val_acc', 
                                                        'N/A'):.3f}
    Test Loss: {evaluation_results.get('test_loss', 'N/A'):.4f}
    
    Number of Classes: {evaluation_results.get('num_classes', 'N/A')}
    
    Training Time: {evaluation_results.get('training_time', 'N/A')}
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Model summary plot saved to: {save_path}")


def save_all_plots(
    training_history: Dict[str, List[float]],
    evaluation_results: Dict[str, Any],
    output_dir: str = "plots",
    show_plots: bool = False
) -> None:
    """
    Save all visualization plots to a specified directory.
    
    Args:
        training_history: Dictionary containing training metrics
        evaluation_results: Dictionary containing evaluation results
        output_dir: Directory to save all plots
        show_plots: Whether to display plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving all plots to directory: {output_dir}")
    
    # Save training history plots
    plot_training_history(
        training_history, 
        save_path=os.path.join(output_dir, 'training_history.png'),
        show_plot=show_plots
    )
    
    plot_loss_and_accuracy(
        training_history,
        save_path=os.path.join(output_dir, 'loss_accuracy_curves.png'),
        show_plot=show_plots
    )
    
    # Save evaluation plots if data is available
    if ('predictions' in evaluation_results and 
            'true_labels' in evaluation_results):
        plot_confusion_matrix(
            evaluation_results['true_labels'],
            evaluation_results['predictions'],
            class_names=evaluation_results.get('class_names'),
            save_path=os.path.join(output_dir, 'confusion_matrix.png'),
            show_plot=show_plots
        )
        
        plot_confusion_matrix(
            evaluation_results['true_labels'],
            evaluation_results['predictions'],
            class_names=evaluation_results.get('class_names'),
            save_path=os.path.join(output_dir, 
                                   'confusion_matrix_normalized.png'),
            show_plot=show_plots,
            normalize=True
        )
    
    # Save class performance plot
    if 'classification_report' in evaluation_results:
        plot_class_performance(
            evaluation_results,
            save_path=os.path.join(output_dir, 'class_performance.png'),
            show_plot=show_plots
        )
    
    # Save comprehensive summary
    plot_model_summary(
        evaluation_results,
        training_history,
        save_path=os.path.join(output_dir, 'model_summary.png'),
        show_plot=show_plots
    )
    
    print(f"All plots saved successfully to: {output_dir}")