"""Utility functions module for ResNet-50 system."""

# Import core helpers (no optional dependencies)
from .helpers import (
    set_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    save_model_state,
    load_model_state,
    get_model_summary,
    freeze_layers,
    ensure_dir,
    format_time,
    get_memory_usage,
    clear_gpu_cache,
    TrainingLogger,
    MetricsTracker,
    ProgressMonitor,
    setup_logging
)

# Try to import visualization functions (optional dependencies)
try:
    from .visualization import (
        plot_training_history,
        plot_loss_and_accuracy,
        plot_training_comparison,
        plot_confusion_matrix,
        plot_class_performance,
        plot_model_summary,
        save_all_plots
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError as e:
    # Create dummy functions for missing visualization
    def _dummy_plot(*args, **kwargs):
        print("⚠️  Visualization function not available "
              "(missing dependencies)")
    
    plot_training_history = _dummy_plot
    plot_loss_and_accuracy = _dummy_plot
    plot_training_comparison = _dummy_plot
    plot_confusion_matrix = _dummy_plot
    plot_class_performance = _dummy_plot
    plot_model_summary = _dummy_plot
    save_all_plots = _dummy_plot
    _VISUALIZATION_AVAILABLE = False

__all__ = [
    # Visualization functions
    'plot_training_history',
    'plot_loss_and_accuracy', 
    'plot_training_comparison',
    'plot_confusion_matrix',
    'plot_class_performance',
    'plot_model_summary',
    'save_all_plots',
    
    # Helper functions
    'set_seed',
    'get_device',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    'save_model_state',
    'load_model_state',
    'get_model_summary',
    'freeze_layers',
    'ensure_dir',
    'format_time',
    'get_memory_usage',
    'clear_gpu_cache',
    
    # Logging and monitoring classes
    'TrainingLogger',
    'MetricsTracker',
    'ProgressMonitor',
    'setup_logging'
]