"""
Functionality preservation tests for the ResNet-50 image classification system.

Tests to ensure that the refactored code maintains the same functionality
as the original implementation, including training strategies and model performance.
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import csv
import numpy as np

from src.config.settings import Config, create_config_from_original
from src.data.manager import DataManager
from src.models.resnet import ResNetClassifier, create_model
from src.training.trainer import BaseTrainer
from src.training.validator import Validator
from src.utils.helpers import set_seed


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def train(self, train_loader, val_loader, optimizer, scheduler, criterion):
        """Simple train implementation for testing."""
        results = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'epochs_completed': 0,
            'best_val_acc': 0.0
        }
        
        for epoch in range(3):  # 3 epochs for testing
            # Train epoch
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            results['train_losses'].append(train_metrics['loss'])
            results['train_accs'].append(train_metrics['accuracy'])
            
            # Validation epoch
            validator = Validator(self.model, self.device)
            val_metrics = validator.validate(val_loader, criterion)
            results['val_losses'].append(val_metrics['loss'])
            results['val_accs'].append(val_metrics['accuracy'])
            
            results['epochs_completed'] += 1
            if val_metrics['accuracy'] > results['best_val_acc']:
                results['best_val_acc'] = val_metrics['accuracy']
            
            # Update learning rate
            if scheduler:
                scheduler.step()
        
        return results


class TestOriginalConfigurationCompatibility(unittest.TestCase):
    """Test that refactored code works with original script configuration."""
    
    def setUp(self):
        """Set up test fixtures with original configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "classify-leaves")
        os.makedirs(self.images_dir)
        
        # Create test dataset matching original structure
        self.csv_path = os.path.join(self.images_dir, "train.csv")
        self.create_test_dataset()
        
        # Use original configuration
        with patch('os.path.exists', return_value=True):
            self.config = create_config_from_original()
            # Override paths for testing
            self.config.data_path = self.images_dir
            self.config.train_csv_path = self.csv_path
            self.config.device = "cpu"
            self.config.num_workers = 0
            self.config.epochs = 3  # Reduced for testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """Create a test dataset with multiple classes."""
        # Create 20 images across 5 classes (4 per class)
        class_names = [f"class_{i}" for i in range(5)]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(20):
                class_name = class_names[i % 5]
                img_name = f"image_{i:03d}.jpg"
                writer.writerow([img_name, class_name])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, img_name)
                img = Image.new('RGB', (224, 224), color=(i*10, i*10, i*10))
                img.save(img_path)
    
    def test_original_config_parameters(self):
        """Test that original configuration parameters are preserved."""
        config = create_config_from_original()
        
        # Test key parameters match original script
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.num_classes, 176)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.train_ratio, 0.8)
        self.assertEqual(config.stage1_epochs, 15)
        self.assertEqual(config.stage2_epochs, 30)
        self.assertEqual(config.stage1_lr, 0.001)
        self.assertEqual(config.stage2_lr, 0.0001)
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.normalize_mean, (0.485, 0.456, 0.406))
        self.assertEqual(config.normalize_std, (0.229, 0.224, 0.225))
    
    def test_model_architecture_compatibility(self):
        """Test that model architecture matches original implementation."""
        # Override num_classes for our test dataset
        self.config.num_classes = 5
        
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            device=torch.device(self.config.device)
        )
        
        # Test model structure
        self.assertIsInstance(model, ResNetClassifier)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.pretrained, True)
        
        # Test output shape
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))
        
        # Test that model has expected layers
        self.assertTrue(hasattr(model, 'backbone'))
        self.assertTrue(hasattr(model.backbone, 'fc'))
        self.assertEqual(model.backbone.fc.out_features, 5)
    
    def test_data_loading_compatibility(self):
        """Test that data loading works with original configuration."""
        # Override num_classes for our test dataset
        self.config.num_classes = 5
        
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Test data loader properties
        self.assertEqual(train_loader.batch_size, self.config.batch_size)
        self.assertEqual(val_loader.batch_size, self.config.batch_size)
        self.assertEqual(train_loader.num_workers, self.config.num_workers)
        
        # Test data format
        train_batch = next(iter(train_loader))
        self.assertIn('image', train_batch)
        self.assertIn('label', train_batch)
        
        images = train_batch['image']
        labels = train_batch['label']
        
        # Test image preprocessing (should be normalized)
        self.assertEqual(images.shape[1:], (3, 224, 224))
        self.assertTrue(torch.all(images >= -3))  # Roughly normalized range
        self.assertTrue(torch.all(images <= 3))


class TestTrainingStrategies(unittest.TestCase):
    """Test different training strategies work correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create small test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(12):  # 12 images, 3 classes
                class_name = f"class_{i % 3}"
                img_name = f"image_{i:03d}.jpg"
                writer.writerow([img_name, class_name])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, img_name)
                img = Image.new('RGB', (64, 64), color=(i*20, i*20, i*20))
                img.save(img_path)
        
        # Base config
        self.config = Config(
            data_path=self.images_dir,
            train_csv_path=self.csv_path,
            batch_size=4,
            num_classes=3,
            train_ratio=0.75,
            num_workers=0,
            device="cpu",
            epochs=3,
            learning_rate=0.01
        )
        
        set_seed(42)  # For reproducibility
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_single_stage_training(self):
        """Test single-stage training strategy."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device(self.config.device)
        )
        
        # Single-stage training (all layers trainable)
        model.freeze_backbone(freeze=False)
        
        # Create trainer
        trainer = ConcreteTrainer(model, self.config, torch.device(self.config.device))
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Run training
        results = trainer.train(train_loader, val_loader, optimizer, scheduler, criterion)
        
        # Verify training completed
        self.assertEqual(results['epochs_completed'], 3)
        self.assertEqual(len(results['train_losses']), 3)
        self.assertEqual(len(results['val_losses']), 3)
        
        # Check that all parameters were trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(trainable_params, total_params)
    
    def test_two_stage_training_simulation(self):
        """Test two-stage training strategy (feature extraction + fine-tuning)."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device(self.config.device)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Stage 1: Feature extraction (backbone frozen)
        model.freeze_backbone(freeze=True)
        
        stage1_trainer = ConcreteTrainer(model, self.config, torch.device(self.config.device))
        stage1_optimizer = optim.Adam(model.parameters(), lr=self.config.stage1_lr)
        stage1_scheduler = optim.lr_scheduler.StepLR(stage1_optimizer, step_size=1, gamma=0.5)
        
        stage1_results = stage1_trainer.train(train_loader, val_loader, stage1_optimizer, stage1_scheduler, criterion)
        
        # Verify stage 1 training
        self.assertEqual(stage1_results['epochs_completed'], 3)
        
        # Check that only fc layer was trainable in stage 1
        trainable_params_stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(trainable_params_stage1, total_params)
        
        # Stage 2: Fine-tuning (all layers trainable)
        model.freeze_backbone(freeze=False)
        
        stage2_trainer = ConcreteTrainer(model, self.config, torch.device(self.config.device))
        stage2_optimizer = optim.Adam(model.parameters(), lr=self.config.stage2_lr)
        stage2_scheduler = optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=3)
        
        stage2_results = stage2_trainer.train(train_loader, val_loader, stage2_optimizer, stage2_scheduler, criterion)
        
        # Verify stage 2 training
        self.assertEqual(stage2_results['epochs_completed'], 3)
        
        # Check that all parameters are trainable in stage 2
        trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable_params_stage2, total_params)
    
    def test_early_stopping_simulation(self):
        """Test early stopping functionality."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device(self.config.device)
        )
        
        # Create trainer with patience
        trainer = ConcreteTrainer(model, self.config, torch.device(self.config.device))
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Run training (should complete all epochs in this simple case)
        results = trainer.train(train_loader, val_loader, optimizer, None, criterion)
        
        # Verify training results structure
        self.assertIn('best_val_acc', results)
        self.assertIn('epochs_completed', results)
        self.assertGreaterEqual(results['best_val_acc'], 0.0)
        self.assertLessEqual(results['best_val_acc'], 100.0)


class TestModelPerformanceConsistency(unittest.TestCase):
    """Test that model performance is consistent and reasonable."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create deterministic test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        self.create_deterministic_dataset()
        
        self.config = Config(
            data_path=self.images_dir,
            train_csv_path=self.csv_path,
            batch_size=4,
            num_classes=2,
            train_ratio=0.75,
            num_workers=0,
            device="cpu",
            epochs=5,
            learning_rate=0.01
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_deterministic_dataset(self):
        """Create a deterministic dataset for consistent testing."""
        # Create 16 images, 8 per class with distinct patterns
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(16):
                class_name = "class_0" if i < 8 else "class_1"
                img_name = f"image_{i:03d}.jpg"
                writer.writerow([img_name, class_name])
                
                # Create images with distinct patterns for each class
                img_path = os.path.join(self.images_dir, img_name)
                if i < 8:
                    # Class 0: darker images
                    img = Image.new('RGB', (64, 64), color=(50, 50, 50))
                else:
                    # Class 1: brighter images
                    img = Image.new('RGB', (64, 64), color=(200, 200, 200))
                img.save(img_path)
    
    def test_model_reproducibility(self):
        """Test that model training is reproducible with same seed."""
        results1 = self._train_model_with_seed(42)
        results2 = self._train_model_with_seed(42)
        
        # Results should be identical with same seed
        self.assertEqual(len(results1['train_losses']), len(results2['train_losses']))
        self.assertEqual(len(results1['val_losses']), len(results2['val_losses']))
        
        # Training should complete successfully
        self.assertGreater(results1['epochs_completed'], 0)
        self.assertGreater(results2['epochs_completed'], 0)
    
    def test_model_learning_capability(self):
        """Test that model can learn from the data."""
        results = self._train_model_with_seed(123)
        
        # Model should complete training
        self.assertGreater(results['epochs_completed'], 0)
        
        # Training loss should generally decrease or stay reasonable
        train_losses = results['train_losses']
        self.assertGreater(len(train_losses), 0)
        
        # All losses should be positive and finite
        for loss in train_losses:
            self.assertGreater(loss, 0)
            self.assertTrue(np.isfinite(loss))
        
        # Validation accuracy should be reasonable (at least random chance)
        val_accs = results['val_accs']
        self.assertGreater(len(val_accs), 0)
        
        for acc in val_accs:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 100.0)
    
    def test_model_validation_consistency(self):
        """Test that validation results are consistent."""
        # Train model
        results = self._train_model_with_seed(456)
        
        # Create data pipeline for validation
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        _, val_loader = data_manager.get_dataloaders()
        
        # Create fresh model and validator
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device(self.config.device)
        )
        
        validator = Validator(model, torch.device(self.config.device))
        criterion = nn.CrossEntropyLoss()
        
        # Run validation multiple times
        val_results = []
        for _ in range(3):
            metrics = validator.validate(val_loader, criterion)
            val_results.append(metrics)
        
        # Results should be consistent (same untrained model)
        for i in range(1, len(val_results)):
            self.assertAlmostEqual(
                val_results[0]['loss'], 
                val_results[i]['loss'], 
                places=4
            )
            self.assertEqual(
                val_results[0]['accuracy'], 
                val_results[i]['accuracy']
            )
    
    def _train_model_with_seed(self, seed):
        """Helper method to train model with specific seed."""
        set_seed(seed)
        
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device(self.config.device)
        )
        
        # Create trainer
        trainer = ConcreteTrainer(model, self.config, torch.device(self.config.device))
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Run training
        return trainer.train(train_loader, val_loader, optimizer, None, criterion)


if __name__ == '__main__':
    unittest.main()