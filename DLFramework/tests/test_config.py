"""
Unit tests for configuration module.

Tests the Config dataclass, validation functions, and configuration loading.
"""

import unittest
import tempfile
import os
from unittest.mock import patch
import torch

from src.config.settings import (
    Config, 
    load_config, 
    validate_config, 
    get_default_config,
    create_config_from_original
)


class TestConfig(unittest.TestCase):
    """Test cases for Config dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv = os.path.join(self.temp_dir, "train.csv")
        
        # Create a dummy CSV file
        with open(self.temp_csv, 'w') as f:
            f.write("image1.jpg,class1\n")
            f.write("image2.jpg,class2\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_config_default_initialization(self):
        """Test Config initialization with default values."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv)
            
            # Test default values
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.num_classes, 176)
            self.assertEqual(config.learning_rate, 0.001)
            self.assertTrue(config.pretrained)
            self.assertEqual(config.image_size, 224)
    
    def test_config_custom_initialization(self):
        """Test Config initialization with custom values."""
        with patch('os.path.exists', return_value=True):
            config = Config(
                data_path=self.temp_dir,
                train_csv_path=self.temp_csv,
                batch_size=32,
                num_classes=100,
                learning_rate=0.01
            )
            
            self.assertEqual(config.batch_size, 32)
            self.assertEqual(config.num_classes, 100)
            self.assertEqual(config.learning_rate, 0.01)
    
    def test_config_device_auto_detection(self):
        """Test automatic device detection."""
        with patch('os.path.exists', return_value=True):
            with patch('torch.cuda.is_available', return_value=True):
                config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv)
                self.assertEqual(config.device, "cuda")
            
            with patch('torch.cuda.is_available', return_value=False):
                config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv)
                self.assertEqual(config.device, "cpu")
    
    def test_config_invalid_paths(self):
        """Test Config with invalid paths."""
        with self.assertRaises(ValueError):
            Config(data_path="/nonexistent/path", train_csv_path=self.temp_csv)
        
        with self.assertRaises(ValueError):
            Config(data_path=self.temp_dir, train_csv_path="/nonexistent/file.csv")


class TestConfigValidation(unittest.TestCase):
    """Test cases for configuration validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv = os.path.join(self.temp_dir, "train.csv")
        
        with open(self.temp_csv, 'w') as f:
            f.write("image1.jpg,class1\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_validate_config_valid(self):
        """Test validation with valid configuration."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv)
            self.assertTrue(validate_config(config))
    
    def test_validate_config_invalid_train_ratio(self):
        """Test validation with invalid train ratio."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv, train_ratio=0.0)
            with self.assertRaises(ValueError):
                validate_config(config)
            
            config.train_ratio = 1.0
            with self.assertRaises(ValueError):
                validate_config(config)
    
    def test_validate_config_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv, batch_size=0)
            with self.assertRaises(ValueError):
                validate_config(config)
    
    def test_validate_config_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv, learning_rate=0.0)
            with self.assertRaises(ValueError):
                validate_config(config)
    
    def test_validate_config_invalid_device(self):
        """Test validation with invalid device."""
        with patch('os.path.exists', return_value=True):
            config = Config(data_path=self.temp_dir, train_csv_path=self.temp_csv, device="invalid")
            with self.assertRaises(ValueError):
                validate_config(config)


class TestConfigUtilities(unittest.TestCase):
    """Test cases for configuration utility functions."""
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        with patch('os.path.exists', return_value=True):
            config = load_config()
            self.assertIsInstance(config, Config)
            self.assertEqual(config.batch_size, 16)
    
    def test_load_config_custom(self):
        """Test loading configuration from dictionary."""
        with patch('os.path.exists', return_value=True):
            config_dict = {
                'data_path': 'test_path',
                'train_csv_path': 'test.csv',
                'batch_size': 32,
                'learning_rate': 0.01
            }
            config = load_config(config_dict)
            self.assertEqual(config.batch_size, 32)
            self.assertEqual(config.learning_rate, 0.01)
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        with patch('os.path.exists', return_value=True):
            config = get_default_config()
            self.assertIsInstance(config, Config)
            self.assertEqual(config.data_path, "classify-leaves")
    
    def test_create_config_from_original(self):
        """Test creating configuration matching original script."""
        with patch('os.path.exists', return_value=True):
            config = create_config_from_original()
            self.assertIsInstance(config, Config)
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.num_classes, 176)
            self.assertEqual(config.stage1_epochs, 15)
            self.assertEqual(config.stage2_epochs, 30)


if __name__ == '__main__':
    unittest.main()