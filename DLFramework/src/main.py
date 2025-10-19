"""
Main entry point for ResNet-50 image classification system.

This module provides the command-line interface and orchestrates the complete
training pipeline for the ResNet-50 image classification system. It handles
configuration loading, data setup, model creation, training execution, and
results visualization.

Key Features:
    - Command-line argument parsing with comprehensive options
    - Multiple training strategies (single, early stopping, two-stage)
    - Automatic environment validation and dependency checking
    - Comprehensive error handling and logging
    - Results visualization and plot generation
    - Configuration validation and override capabilities

Usage:
    Basic training with default settings:
        $ python src/main.py
    
    Two-stage training with custom parameters:
        $ python src/main.py --training-strategy two_stage --epochs 100 --batch-size 32
    
    Training with early stopping and verbose logging:
        $ python src/main.py --training-strategy early_stopping --patience 15 --verbose
    
    Custom data path and configuration:
        $ python src/main.py --data-path /path/to/data --config config.json

Training Strategies:
    - single: Basic training without early stopping
    - early_stopping: Training with early stopping based on validation accuracy
    - two_stage: Two-stage transfer learning (feature extraction + fine-tuning)

Exit Codes:
    0: Success
    1: Error or user interruption
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

try:
    from config import Config, load_config, validate_config, create_config_from_original
    from data import DataManager
    from models import create_model, ResNetClassifier
    from training import TwoStageTrainer, EarlyStoppingTrainer, Trainer
    
    # Import utils with optional matplotlib dependency
    try:
        from utils import set_seed, get_device, setup_logging, TrainingLogger, save_all_plots
    except ImportError as utils_error:
        if "matplotlib" in str(utils_error):
            # Import only non-visualization utilities
            from utils.helpers import set_seed, get_device, setup_logging, TrainingLogger
            print("⚠️  Visualization features disabled (matplotlib not available)")
            save_all_plots = None
        else:
            raise utils_error
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the ResNet-50 training system.
    
    This function sets up the argument parser with all available command-line
    options for configuring the training process, including data paths, training
    parameters, model options, and utility settings.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Path to configuration file
            - data_path: Path to data directory
            - batch_size: Batch size for training
            - num_workers: Number of data loading workers
            - epochs: Number of training epochs
            - lr: Learning rate
            - patience: Early stopping patience
            - training_strategy: Training strategy to use
            - pretrained: Whether to use pretrained model
            - seed: Random seed for reproducibility
            - verbose: Enable verbose logging
            - save_plots: Save training plots
    
    Example:
        >>> args = parse_arguments()
        >>> print(f"Training strategy: {args.training_strategy}")
        >>> print(f"Batch size: {args.batch_size}")
    """
    parser = argparse.ArgumentParser(
        description="ResNet-50 Image Classification System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Data options
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, help="Number of data loading workers")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    
    # Training strategy options
    parser.add_argument("--training-strategy", type=str, 
                       choices=["single", "early_stopping", "two_stage"],
                       default="two_stage",
                       help="Training strategy to use")
    
    # Model options
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained ResNet model")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                       help="Don't use pretrained ResNet model")
    
    # Utility options
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--save-plots", action="store_true", default=True,
                       help="Save training plots")
    
    return parser.parse_args()


def load_and_validate_config(args: argparse.Namespace) -> Config:
    """
    Load and validate configuration from file and command line arguments.
    
    This function loads the base configuration from a file (if specified) or
    uses the default configuration, then overrides parameters with command-line
    arguments. Finally, it validates the resulting configuration.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing
            configuration overrides and the path to a config file.
    
    Returns:
        Config: Validated configuration object with all parameters set.
    
    Raises:
        ValueError: If configuration validation fails due to invalid parameters.
        FileNotFoundError: If specified config file doesn't exist.
    
    Example:
        >>> args = parse_arguments()
        >>> config = load_and_validate_config(args)
        >>> print(f"Using device: {config.device}")
    """
    # Load base configuration
    if args.config:
        try:
            config = load_config(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")
            print("Using default configuration...")
            config = create_config_from_original()
    else:
        config = create_config_from_original()
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.patience:
        config.patience = args.patience
    if args.seed:
        config.seed = args.seed
    elif not hasattr(config, 'seed'):
        config.seed = 42  # Default seed
    if hasattr(args, 'pretrained'):
        config.pretrained = args.pretrained
    
    # Validate configuration
    try:
        validate_config(config)
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise
    
    return config


def print_config_summary(config: Config) -> None:
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("RESNET-50 CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Data Path: {config.data_path}")
    print(f"Train CSV: {config.train_csv_path}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Number of Workers: {config.num_workers}")
    print(f"Number of Classes: {config.num_classes}")
    print(f"Device: {config.device}")
    print(f"Random Seed: {getattr(config, 'seed', 42)}")
    print(f"Pretrained Model: {config.pretrained}")
    print(f"Training Epochs: {config.epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Early Stopping Patience: {config.patience}")
    print(f"Two-Stage Training:")
    print(f"  Stage 1 - Epochs: {config.stage1_epochs}, LR: {config.stage1_lr}")
    print(f"  Stage 2 - Epochs: {config.stage2_epochs}, LR: {config.stage2_lr}")
    print("="*60)


def setup_data_and_model(config: Config) -> tuple:
    """Set up data loaders and model."""
    print("\n🔄 Setting up data and model...")
    
    # Initialize data manager
    data_manager = DataManager(config)
    data_manager.setup_datasets()
    data_manager.setup_dataloaders()
    train_loader, val_loader = data_manager.get_dataloaders()
    class_info = data_manager.get_class_info()
    
    print(f"✅ Data loaded successfully:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Number of classes: {class_info['num_classes']}")
    
    # Create model
    model = create_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        device=config.device
    )
    
    print(f"✅ Model created successfully:")
    print(f"  Architecture: ResNet-50")
    print(f"  Pretrained: {config.pretrained}")
    print(f"  Device: {config.device}")
    
    return train_loader, val_loader, model, class_info


def run_training(config: Config, train_loader, val_loader, model, 
                training_strategy: str) -> Dict[str, Any]:
    """Run training with the specified strategy."""
    print(f"\n🚀 Starting training with strategy: {training_strategy}")
    print("-" * 50)
    
    # Set up criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Convert device string to torch.device
    device = torch.device(config.device)
    
    try:
        if training_strategy == "single":
            trainer = Trainer(model, config, device)
            
            # Create optimizer and scheduler for single training
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=10, 
                gamma=0.1
            )
            
            results = trainer.train(train_loader, val_loader, optimizer, scheduler, criterion)
            
        elif training_strategy == "early_stopping":
            trainer = EarlyStoppingTrainer(model, config, device)
            
            # Create optimizer and scheduler for early stopping training
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=10, 
                gamma=0.1
            )
            
            results = trainer.train(train_loader, val_loader, optimizer, scheduler, criterion)
            
        elif training_strategy == "two_stage":
            trainer = TwoStageTrainer(model, config, device)
            
            # TwoStageTrainer creates its own optimizers and schedulers
            # Pass None for optimizer and scheduler as they are ignored
            results = trainer.train(train_loader, val_loader, None, None, criterion)
            
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def save_results_and_plots(results: Dict[str, Any], config: Config, 
                          save_plots: bool = True) -> None:
    """Save training results and generate plots."""
    print("\n📊 Saving results and generating plots...")
    
    if save_plots and save_all_plots is not None:
        try:
            save_all_plots(results, save_dir="plots")
            print("✅ Training plots saved to 'plots' directory")
        except Exception as e:
            print(f"⚠️  Failed to save plots: {e}")
    elif save_plots and save_all_plots is None:
        print("⚠️  Plot saving disabled (visualization dependencies not available)")
    
    # Print final results summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    if 'best_val_acc' in results:
        print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    
    if 'stage1_acc' in results and 'stage2_acc' in results:
        print(f"Stage 1 Best Accuracy: {results['stage1_acc']:.2f}%")
        print(f"Stage 2 Best Accuracy: {results['stage2_acc']:.2f}%")
        print(f"Total Improvement: {results['stage2_acc'] - results['stage1_acc']:.2f}%")
    
    if 'epochs_completed' in results:
        print(f"Epochs Completed: {results['epochs_completed']}")
    
    print("="*60)


def run_complete_pipeline(config: Config, training_strategy: str, 
                         save_plots: bool = True) -> Dict[str, Any]:
    """
    Run the complete training pipeline with comprehensive error handling.
    
    Args:
        config: Configuration object
        training_strategy: Training strategy to use
        save_plots: Whether to save training plots
        
    Returns:
        Dict containing complete pipeline results
    """
    pipeline_results = {}
    
    try:
        # Stage 1: Data and Model Setup
        print("\n📊 Stage 1: Setting up data and model...")
        train_loader, val_loader, model, class_info = setup_data_and_model(config)
        pipeline_results['data_setup'] = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'num_classes': class_info['num_classes'],
            'class_names': class_info.get('class_names', [])
        }
        
        # Stage 2: Training
        print("\n🚀 Stage 2: Running training...")
        training_results = run_training(config, train_loader, val_loader, model, training_strategy)
        pipeline_results['training'] = training_results
        
        # Stage 3: Results and Visualization
        print("\n📈 Stage 3: Generating results and visualizations...")
        save_results_and_plots(training_results, config, save_plots)
        pipeline_results['visualization'] = {'plots_saved': save_plots}
        
        # Stage 4: Final Model Evaluation (if requested)
        if hasattr(config, 'run_final_evaluation') and config.run_final_evaluation:
            print("\n🔍 Stage 4: Running final model evaluation...")
            from training import ModelEvaluator
            evaluator = ModelEvaluator(model, config.device)
            eval_results = evaluator.evaluate_detailed(
                val_loader, 
                torch.nn.CrossEntropyLoss(),
                class_info.get('class_names', [])
            )
            pipeline_results['evaluation'] = eval_results
        
        pipeline_results['status'] = 'success'
        return pipeline_results
        
    except Exception as e:
        pipeline_results['status'] = 'failed'
        pipeline_results['error'] = str(e)
        raise


def validate_environment() -> None:
    """Validate the environment and dependencies."""
    try:
        import torch
        import torchvision
        import PIL
        import matplotlib
        
        print("✅ All required dependencies are available")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA is not available, using CPU")
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        raise


def main() -> int:
    """Main execution function with comprehensive error handling and logging."""
    start_time = None
    
    try:
        import time
        start_time = time.time()
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up logging
        log_level_str = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level=log_level_str)
        
        print("🎯 ResNet-50 Image Classification System")
        print("=" * 50)
        
        # Validate environment
        validate_environment()
        
        # Load and validate configuration
        config = load_and_validate_config(args)
        print_config_summary(config)
        
        # Set random seed for reproducibility
        set_seed(config.seed)
        print(f"✅ Random seed set to: {config.seed}")
        
        # Run complete pipeline
        pipeline_results = run_complete_pipeline(
            config, 
            args.training_strategy, 
            args.save_plots
        )
        
        # Calculate total execution time
        if start_time:
            total_time = time.time() - start_time
            print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
        
        # Print final summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if 'training' in pipeline_results:
            training_results = pipeline_results['training']
            if 'best_val_acc' in training_results:
                print(f"Best Validation Accuracy: {training_results['best_val_acc']:.2f}%")
            
            if 'stage2_best_acc' in training_results:
                print(f"Two-Stage Training Results:")
                print(f"  Stage 1: {training_results['stage1_best_acc']:.2f}%")
                print(f"  Stage 2: {training_results['stage2_best_acc']:.2f}%")
                print(f"  Improvement: {training_results['total_improvement']:+.2f}%")
        
        if 'data_setup' in pipeline_results:
            data_info = pipeline_results['data_setup']
            print(f"Dataset: {data_info['train_samples']} train, {data_info['val_samples']} val samples")
            print(f"Classes: {data_info['num_classes']}")
        
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if start_time:
            elapsed = time.time() - start_time
            print(f"⏱️  Elapsed time before interruption: {elapsed:.2f} seconds")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        if start_time:
            elapsed = time.time() - start_time
            print(f"⏱️  Elapsed time before failure: {elapsed:.2f} seconds")
            
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            print("\n📋 Full error traceback:")
            traceback.print_exc()
        else:
            print("💡 Use --verbose flag for detailed error information")
            
        return 1


if __name__ == "__main__":
    exit(main())