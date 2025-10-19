# Design Document

## Overview

The refactored ResNet-50 system will be organized into a modular architecture with clear separation of concerns. The design follows Python best practices and creates reusable components that can be easily maintained and extended.

## Architecture

The system will be structured as follows:

```
src/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Custom dataset classes
│   └── transforms.py        # Data transformation utilities
├── models/
│   ├── __init__.py
│   └── resnet.py           # Model architecture and utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training logic and strategies
│   └── validator.py        # Validation and evaluation
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Plotting and visualization
│   └── helpers.py          # General utility functions
└── main.py                 # Main entry point
```

## Components and Interfaces

### 1. Configuration Module (`config/settings.py`)

**Purpose**: Centralize all configuration parameters and settings.

**Key Classes/Functions**:
- `Config`: Dataclass containing all configuration parameters
- `load_config()`: Function to load and validate configuration

**Interface**:
```python
@dataclass
class Config:
    # Data settings
    data_path: str
    train_ratio: float
    batch_size: int
    num_workers: int
    
    # Model settings
    num_classes: int
    pretrained: bool
    
    # Training settings
    epochs: int
    learning_rate: float
    patience: int
    min_delta: float
```

### 2. Data Module (`data/`)

**Purpose**: Handle all data loading, preprocessing, and transformation logic.

**Key Classes**:
- `ImageFolderWithTxt`: Custom dataset class for loading images with text labels
- `DataTransforms`: Class managing different transformation pipelines
- `DataManager`: High-level interface for data operations

**Interface**:
```python
class DataManager:
    def __init__(self, config: Config)
    def get_datasets(self) -> Tuple[Dataset, Dataset]
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]
    def get_class_info(self) -> Dict[str, Any]
```

### 3. Model Module (`models/resnet.py`)

**Purpose**: Define model architecture and related utilities.

**Key Classes/Functions**:
- `ResNetClassifier`: Wrapper class for ResNet model
- `create_model()`: Factory function for model creation
- `freeze_layers()`: Utility for layer freezing/unfreezing

**Interface**:
```python
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def freeze_backbone(self, freeze: bool = True) -> None
```

### 4. Training Module (`training/`)

**Purpose**: Handle all training-related logic and strategies.

**Key Classes**:
- `Trainer`: Main training orchestrator
- `EarlyStoppingTrainer`: Trainer with early stopping capability
- `TwoStageTrainer`: Specialized trainer for two-stage training

**Interface**:
```python
class Trainer:
    def __init__(self, model, config: Config, device: torch.device)
    def train_epoch(self, train_loader, optimizer, criterion) -> Dict[str, float]
    def train(self, train_loader, val_loader, optimizer, scheduler, criterion) -> Dict[str, Any]
```

### 5. Validation Module (`training/validator.py`)

**Purpose**: Handle model evaluation and performance metrics.

**Key Classes/Functions**:
- `Validator`: Main validation class
- `evaluate_model()`: Comprehensive model evaluation
- `calculate_metrics()`: Metric calculation utilities

**Interface**:
```python
class Validator:
    def __init__(self, model, device: torch.device)
    def validate(self, val_loader, criterion) -> Dict[str, float]
    def evaluate_detailed(self, test_loader, criterion, class_names) -> Dict[str, Any]
```

### 6. Visualization Module (`utils/visualization.py`)

**Purpose**: Handle all plotting and visualization tasks.

**Key Functions**:
- `plot_training_history()`: Plot training metrics over time
- `plot_confusion_matrix()`: Generate confusion matrix visualization
- `save_training_plots()`: Save all training visualizations

### 7. Utilities Module (`utils/helpers.py`)

**Purpose**: General utility functions and helpers.

**Key Functions**:
- `set_seed()`: Set random seeds for reproducibility
- `save_checkpoint()`: Save model checkpoints
- `load_checkpoint()`: Load model checkpoints
- `get_device()`: Device detection utility

## Data Models

### Configuration Data Model
```python
@dataclass
class Config:
    # Data configuration
    data_path: str = "classify-leaves"
    train_ratio: float = 0.8
    batch_size: int = 16
    num_workers: int = 0
    
    # Model configuration
    num_classes: int = 176
    pretrained: bool = True
    
    # Training configuration
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    min_delta: float = 0.001
    
    # Two-stage training
    stage1_epochs: int = 15
    stage1_patience: int = 5
    stage1_lr: float = 0.001
    stage2_epochs: int = 30
    stage2_patience: int = 8
    stage2_lr: float = 0.0001
```

### Training Results Data Model
```python
@dataclass
class TrainingResults:
    best_val_acc: float
    train_losses: List[float]
    val_losses: List[float]
    train_accs: List[float]
    val_accs: List[float]
    epochs_completed: int
```

## Error Handling

### Data Loading Errors
- Handle missing image files gracefully
- Validate CSV format and content
- Provide clear error messages for data issues

### Training Errors
- Handle CUDA out of memory errors
- Validate model architecture compatibility
- Handle checkpoint loading failures

### Configuration Errors
- Validate all configuration parameters
- Provide default values for missing parameters
- Clear error messages for invalid configurations

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (file system, CUDA)
- Test edge cases and error conditions

### Integration Tests
- Test component interactions
- Test full training pipeline with small dataset
- Validate data flow between modules

### Performance Tests
- Memory usage validation
- Training speed benchmarks
- Model accuracy validation

## Implementation Notes

### Backward Compatibility
- Maintain same functionality as original script
- Preserve all training strategies and options
- Keep same model performance characteristics

### Extensibility
- Design interfaces to support new model architectures
- Allow easy addition of new training strategies
- Support different data formats and sources

### Performance Considerations
- Minimize overhead from modularization
- Maintain efficient data loading pipelines
- Preserve GPU utilization patterns