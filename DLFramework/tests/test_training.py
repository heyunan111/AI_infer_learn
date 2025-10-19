"""
Unit tests for training module.

Tests the trainer classes and training functionality.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import BaseTrainer
from src.training.validator import Validator
from src.config.settings import Config
from src.models.resnet import ResNetClassifier


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def train(self, train_loader, val_loader, optimizer, scheduler, criterion):
        """Simple train implementation for testing."""
        return {
            'best_val_acc': 0.5,
            'train_losses': [1.0, 0.8],
            'val_losses': [1.2, 0.9],
            'train_accs': [40.0, 60.0],
            'val_accs': [35.0, 55.0],
            'epochs_completed': 2
        }


class MockDataset:
    """Mock dataset that returns data in the expected format."""
    
    def __init__(self, num_samples, num_classes):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "label": torch.randint(0, self.num_classes, (1,)).item()
        }


def create_mock_dataloader(num_batches=3, batch_size=2, num_classes=5):
    """Create a mock DataLoader for testing."""
    total_samples = num_batches * batch_size
    dataset = MockDataset(total_samples, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestBaseTrainer(unittest.TestCase):
    """Test cases for BaseTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 5
        self.device = torch.device('cpu')
        
        # Create mock model
        self.model = ResNetClassifier(num_classes=self.num_classes, pretrained=False)
        
        # Create mock config
        with patch('os.path.exists', return_value=True):
            self.config = Config(
                data_path="test_data",
                train_csv_path="test.csv",
                batch_size=2,
                learning_rate=0.001,
                epochs=5
            )
        
        # Create trainer
        self.trainer = ConcreteTrainer(self.model, self.config, self.device)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.config, self.config)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.current_epoch, 0)
        self.assertIn('train_losses', self.trainer.training_history)
    
    def test_trainer_invalid_inputs(self):
        """Test trainer with invalid inputs."""
        with self.assertRaises(TypeError):
            ConcreteTrainer("not_a_model", self.config, self.device)
        
        with self.assertRaises(TypeError):
            ConcreteTrainer(self.model, "not_a_config", self.device)
        
        with self.assertRaises(TypeError):
            ConcreteTrainer(self.model, self.config, "not_a_device")
    
    def test_train_epoch(self):
        """Test single epoch training."""
        # Create mock data loader
        train_loader = create_mock_dataloader(num_batches=2, batch_size=2, num_classes=self.num_classes)
        
        # Create optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train one epoch
        metrics = self.trainer.train_epoch(train_loader, optimizer, criterion)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('correct', metrics)
        self.assertIn('total', metrics)
        
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 100.0)


class TestValidator(unittest.TestCase):
    """Test cases for Validator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 5
        self.device = torch.device('cpu')
        
        # Create mock model
        self.model = ResNetClassifier(num_classes=self.num_classes, pretrained=False)
        
        # Create validator
        self.validator = Validator(self.model, self.device)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.model, self.model)
        self.assertEqual(self.validator.device, self.device)
        self.assertIsNotNone(self.validator.logger)
    
    def test_validate_basic(self):
        """Test basic validation functionality."""
        # Create mock data loader
        val_loader = create_mock_dataloader(num_batches=2, batch_size=2, num_classes=self.num_classes)
        
        # Create criterion
        criterion = nn.CrossEntropyLoss()
        
        # Validate
        metrics = self.validator.validate(val_loader, criterion)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('correct', metrics)
        self.assertIn('total', metrics)
        
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 100.0)
    
    def test_validate_invalid_inputs(self):
        """Test validation with invalid inputs."""
        criterion = nn.CrossEntropyLoss()
        
        # Test with None data loader
        with self.assertRaises(ValueError):
            self.validator.validate(None, criterion)
        
        # Test with None criterion
        val_loader = create_mock_dataloader()
        with self.assertRaises(ValueError):
            self.validator.validate(val_loader, None)
    
    def test_validate_empty_loader(self):
        """Test validation with empty data loader."""
        # Create empty data loader
        empty_dataset = TensorDataset(torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        criterion = nn.CrossEntropyLoss()
        
        with self.assertRaises(ValueError):
            self.validator.validate(empty_loader, criterion)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 3
        self.device = torch.device('cpu')
        
        # Create model
        self.model = ResNetClassifier(num_classes=self.num_classes, pretrained=False)
        
        # Create config
        with patch('os.path.exists', return_value=True):
            self.config = Config(
                data_path="test_data",
                train_csv_path="test.csv",
                batch_size=4,
                learning_rate=0.01,
                epochs=2
            )
    
    def test_trainer_validator_integration(self):
        """Test integration between trainer and validator."""
        # Create trainer and validator
        trainer = ConcreteTrainer(self.model, self.config, self.device)
        validator = Validator(self.model, self.device)
        
        # Create mock data loaders
        train_loader = create_mock_dataloader(num_batches=2, batch_size=4, num_classes=self.num_classes)
        val_loader = create_mock_dataloader(num_batches=2, batch_size=4, num_classes=self.num_classes)
        
        # Create optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Train one epoch
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validate
        val_metrics = validator.validate(val_loader, criterion)
        
        # Check that both return valid metrics
        self.assertIn('loss', train_metrics)
        self.assertIn('accuracy', train_metrics)
        self.assertIn('loss', val_metrics)
        self.assertIn('accuracy', val_metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(train_metrics['loss'], 0)
        self.assertGreater(val_metrics['loss'], 0)
        self.assertGreaterEqual(train_metrics['accuracy'], 0)
        self.assertGreaterEqual(val_metrics['accuracy'], 0)


if __name__ == '__main__':
    unittest.main()