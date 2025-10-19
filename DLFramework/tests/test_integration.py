"""
Integration tests for the ResNet-50 image classification system.

Tests complete training pipeline with small dataset, module interactions,
and different configuration scenarios.
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

from src.config.settings import Config, validate_config
from src.data.dataset import ImageFolderWithTxt
from src.data.manager import DataManager
from src.data.transforms import DataTransforms
from src.models.resnet import ResNetClassifier, create_model
from src.training.trainer import BaseTrainer
from src.training.validator import Validator
from src.utils.helpers import set_seed


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def train(self, train_loader, val_loader, optimizer, scheduler, criterion):
        """Simple train implementation for testing."""
        # Run a few epochs of training
        results = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'epochs_completed': 0,
            'best_val_acc': 0.0
        }
        
        for epoch in range(2):  # Just 2 epochs for testing
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
        
        return results


class TestDataPipeline(unittest.TestCase):
    """Test complete data loading and preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures with small dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create small test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        self.class_names = ["class_a", "class_b", "class_c"]
        self.image_files = []
        
        # Create CSV and images
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(12):  # 12 images total, 4 per class
                class_name = self.class_names[i % 3]
                img_name = f"image_{i:03d}.jpg"
                writer.writerow([img_name, class_name])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, img_name)
                img = Image.new('RGB', (64, 64), color=(i*20, i*20, i*20))
                img.save(img_path)
                self.image_files.append(img_path)
        
        # Create config
        self.config = Config(
            data_path=self.images_dir,
            train_csv_path=self.csv_path,
            batch_size=4,
            num_classes=3,
            train_ratio=0.75,
            num_workers=0,
            epochs=2,
            learning_rate=0.01
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_data_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Test data transforms
        transform_config = {
            'image_size': self.config.image_size,
            'normalize_mean': self.config.normalize_mean,
            'normalize_std': self.config.normalize_std
        }
        transforms = DataTransforms(transform_config)
        train_transform = transforms.get_train_transform()
        val_transform = transforms.get_val_transform()
        
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)
        
        # Test data manager
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_dataset, val_dataset = data_manager.get_datasets()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Verify datasets
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertEqual(len(train_dataset) + len(val_dataset), 12)
        
        # Verify data loaders
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Test data loading
        train_batch = next(iter(train_loader))
        self.assertIn('image', train_batch)
        self.assertIn('label', train_batch)
        
        # Verify batch shapes
        images = train_batch['image']
        labels = train_batch['label']
        self.assertEqual(len(images.shape), 4)  # (batch, channels, height, width)
        self.assertEqual(images.shape[1], 3)    # RGB channels
        self.assertEqual(len(labels.shape), 1)  # (batch,)
    
    def test_data_manager_class_info(self):
        """Test data manager class information functionality."""
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        class_info = data_manager.get_class_info()
        
        self.assertIn('num_classes', class_info)
        self.assertIn('class_names', class_info)
        self.assertIn('class_counts', class_info)
        
        self.assertEqual(class_info['num_classes'], 3)
        self.assertEqual(len(class_info['class_names']), 3)
        
        # Check that all classes are represented
        for class_name in self.class_names:
            self.assertIn(class_name, class_info['class_names'])


class TestModelTrainingPipeline(unittest.TestCase):
    """Test complete model training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create small test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(8):  # 8 images total
                class_name = f"class_{i % 2}"  # 2 classes
                img_name = f"image_{i:03d}.jpg"
                writer.writerow([img_name, class_name])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, img_name)
                img = Image.new('RGB', (64, 64), color=(i*30, i*30, i*30))
                img.save(img_path)
        
        # Create config
        self.config = Config(
            data_path=self.images_dir,
            train_csv_path=self.csv_path,
            batch_size=2,
            num_classes=2,
            train_ratio=0.75,
            num_workers=0,
            epochs=2,
            learning_rate=0.01,
            device="cpu"
        )
        
        self.device = torch.device('cpu')
        set_seed(42)  # For reproducibility
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline with small dataset."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=self.device
        )
        
        # Create trainer
        trainer = ConcreteTrainer(model, self.config, self.device)
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Run training
        results = trainer.train(train_loader, val_loader, optimizer, scheduler, criterion)
        
        # Verify results
        self.assertIn('train_losses', results)
        self.assertIn('val_losses', results)
        self.assertIn('train_accs', results)
        self.assertIn('val_accs', results)
        self.assertIn('epochs_completed', results)
        self.assertIn('best_val_acc', results)
        
        # Check that training ran
        self.assertEqual(results['epochs_completed'], 2)
        self.assertEqual(len(results['train_losses']), 2)
        self.assertEqual(len(results['val_losses']), 2)
        
        # Check that metrics are reasonable
        for loss in results['train_losses']:
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)
        
        for acc in results['train_accs']:
            self.assertIsInstance(acc, float)
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 100)
    
    def test_model_validation_pipeline(self):
        """Test model validation pipeline."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        _, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=self.device
        )
        
        # Create validator
        validator = Validator(model, self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Run validation
        metrics = validator.validate(val_loader, criterion)
        
        # Verify metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('correct', metrics)
        self.assertIn('total', metrics)
        
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreater(metrics['loss'], 0)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 100)


class TestConfigurationScenarios(unittest.TestCase):
    """Test different configuration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create minimal test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(4):
                writer.writerow([f"image_{i}.jpg", f"class_{i % 2}"])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, f"image_{i}.jpg")
                img = Image.new('RGB', (32, 32), color=(i*60, i*60, i*60))
                img.save(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_different_batch_sizes(self):
        """Test training with different batch sizes."""
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                config = Config(
                    data_path=self.images_dir,
                    train_csv_path=self.csv_path,
                    batch_size=batch_size,
                    num_classes=2,
                    train_ratio=0.5,
                    num_workers=0,
                    device="cpu"
                )
                
                # Validate config
                self.assertTrue(validate_config(config))
                
                # Test data loading
                data_manager = DataManager(config)
                data_manager.setup_datasets()
                data_manager.setup_dataloaders()
                train_loader, val_loader = data_manager.get_dataloaders()
                
                # Verify batch sizes
                train_batch = next(iter(train_loader))
                self.assertLessEqual(len(train_batch['image']), batch_size)
    
    def test_different_train_ratios(self):
        """Test training with different train/validation splits."""
        train_ratios = [0.25, 0.5, 0.75]
        
        for train_ratio in train_ratios:
            with self.subTest(train_ratio=train_ratio):
                config = Config(
                    data_path=self.images_dir,
                    train_csv_path=self.csv_path,
                    batch_size=2,
                    num_classes=2,
                    train_ratio=train_ratio,
                    num_workers=0,
                    device="cpu"
                )
                
                # Validate config
                self.assertTrue(validate_config(config))
                
                # Test data splitting
                data_manager = DataManager(config)
                data_manager.setup_datasets()
                train_dataset, val_dataset = data_manager.get_datasets()
                
                total_samples = len(train_dataset) + len(val_dataset)
                actual_train_ratio = len(train_dataset) / total_samples
                
                # Allow some tolerance due to rounding
                self.assertAlmostEqual(actual_train_ratio, train_ratio, delta=0.3)
    
    def test_model_with_different_classes(self):
        """Test model creation with different number of classes."""
        num_classes_list = [2, 5, 10]
        
        for num_classes in num_classes_list:
            with self.subTest(num_classes=num_classes):
                model = create_model(
                    num_classes=num_classes,
                    pretrained=False,
                    device=torch.device('cpu')
                )
                
                # Test model output shape
                dummy_input = torch.randn(1, 3, 224, 224)
                output = model(dummy_input)
                
                self.assertEqual(output.shape, (1, num_classes))


class TestModuleInteractions(unittest.TestCase):
    """Test interactions between different modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create test dataset
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(6):
                writer.writerow([f"image_{i}.jpg", f"class_{i % 3}"])
                
                # Create dummy image
                img_path = os.path.join(self.images_dir, f"image_{i}.jpg")
                img = Image.new('RGB', (48, 48), color=(i*40, i*40, i*40))
                img.save(img_path)
        
        self.config = Config(
            data_path=self.images_dir,
            train_csv_path=self.csv_path,
            batch_size=2,
            num_classes=3,
            train_ratio=0.67,
            num_workers=0,
            device="cpu"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_data_manager_interaction(self):
        """Test interaction between Config and DataManager."""
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        
        # Verify that DataManager uses config parameters
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Check batch size
        train_batch = next(iter(train_loader))
        self.assertLessEqual(len(train_batch['image']), self.config.batch_size)
        
        # Check number of workers (should be 0 for testing)
        self.assertEqual(train_loader.num_workers, self.config.num_workers)
        self.assertEqual(val_loader.num_workers, self.config.num_workers)
    
    def test_data_model_interaction(self):
        """Test interaction between data pipeline and model."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, _ = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device('cpu')
        )
        
        # Test that model can process data from pipeline
        batch = next(iter(train_loader))
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        
        # Verify output shape matches expected classes
        self.assertEqual(outputs.shape[1], self.config.num_classes)
        self.assertEqual(outputs.shape[0], len(labels))
    
    def test_trainer_validator_interaction(self):
        """Test interaction between trainer and validator."""
        # Create data pipeline
        data_manager = DataManager(self.config)
        data_manager.setup_datasets()
        data_manager.setup_dataloaders()
        train_loader, val_loader = data_manager.get_dataloaders()
        
        # Create model
        model = create_model(
            num_classes=self.config.num_classes,
            pretrained=False,
            device=torch.device('cpu')
        )
        
        # Create trainer and validator
        trainer = ConcreteTrainer(model, self.config, torch.device('cpu'))
        validator = Validator(model, torch.device('cpu'))
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train one epoch
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validate
        val_metrics = validator.validate(val_loader, criterion)
        
        # Both should return compatible metrics
        for key in ['loss', 'accuracy']:
            self.assertIn(key, train_metrics)
            self.assertIn(key, val_metrics)
            self.assertIsInstance(train_metrics[key], float)
            self.assertIsInstance(val_metrics[key], float)


if __name__ == '__main__':
    unittest.main()