"""
Unit tests for model module.

Tests the ResNetClassifier class and model utility functions.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from src.models.resnet import (
    ResNetClassifier,
    create_model,
    freeze_resnet_layers,
    get_model_summary,
    save_model_state,
    load_model_state,
    validate_model_architecture
)


class TestResNetClassifier(unittest.TestCase):
    """Test cases for ResNetClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 10
        self.model = ResNetClassifier(num_classes=self.num_classes, pretrained=False)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.num_classes, self.num_classes)
        self.assertFalse(self.model.pretrained)
        self.assertIsInstance(self.model.backbone.fc, nn.Linear)
        self.assertEqual(self.model.backbone.fc.out_features, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        # Initially all parameters should be trainable
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Freeze backbone
        self.model.freeze_backbone(freeze=True)
        
        # Only fc layer should be trainable
        trainable_after_freeze = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertLess(trainable_after_freeze, trainable_before)
        
        # Check that fc layer is still trainable
        self.assertTrue(self.model.fc.weight.requires_grad)
        self.assertTrue(self.model.fc.bias.requires_grad)
        
        # Unfreeze backbone
        self.model.freeze_backbone(freeze=False)
        
        # All parameters should be trainable again
        trainable_after_unfreeze = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_after_unfreeze, trainable_before)
    
    def test_get_trainable_params(self):
        """Test trainable parameter counting."""
        trainable_params, total_params = self.model.get_trainable_params()
        
        self.assertIsInstance(trainable_params, int)
        self.assertIsInstance(total_params, int)
        self.assertGreater(trainable_params, 0)
        self.assertGreater(total_params, 0)
        self.assertLessEqual(trainable_params, total_params)
        
        # After freezing backbone, trainable params should be less
        self.model.freeze_backbone(freeze=True)
        trainable_frozen, total_frozen = self.model.get_trainable_params()
        
        self.assertEqual(total_frozen, total_params)  # Total shouldn't change
        self.assertLess(trainable_frozen, trainable_params)


class TestModelUtilities(unittest.TestCase):
    """Test cases for model utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 5
        self.device = torch.device('cpu')
    
    def test_create_model(self):
        """Test model creation factory function."""
        model = create_model(self.num_classes, pretrained=False, device=self.device)
        
        self.assertIsInstance(model, ResNetClassifier)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertFalse(model.pretrained)
    
    def test_create_model_with_device(self):
        """Test model creation with device specification."""
        model = create_model(self.num_classes, pretrained=False, device=self.device)
        
        # Check that model parameters are on the correct device
        param_device = next(model.parameters()).device
        self.assertEqual(param_device, self.device)
    
    def test_freeze_resnet_layers(self):
        """Test layer freezing utility function."""
        model = create_model(self.num_classes, pretrained=False)
        
        # Test freezing
        with patch.object(model, 'freeze_backbone') as mock_freeze:
            with patch.object(model, 'print_param_info') as mock_print:
                freeze_resnet_layers(model, freeze_backbone=True)
                mock_freeze.assert_called_once_with(True)
                mock_print.assert_called_once()
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        model = create_model(self.num_classes, pretrained=False)
        summary = get_model_summary(model)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('model_type', summary)
        self.assertIn('num_classes', summary)
        self.assertIn('total_parameters', summary)
        self.assertIn('trainable_parameters', summary)
        self.assertIn('trainable_ratio', summary)
        self.assertIn('layer_counts', summary)
        
        self.assertEqual(summary['model_type'], 'ResNet-50')
        self.assertEqual(summary['num_classes'], self.num_classes)
        self.assertIsInstance(summary['total_parameters'], int)
        self.assertIsInstance(summary['trainable_parameters'], int)
    
    def test_validate_model_architecture(self):
        """Test model architecture validation."""
        model = create_model(self.num_classes, pretrained=False)
        
        # Valid case
        self.assertTrue(validate_model_architecture(model, self.num_classes))
        
        # Invalid case - wrong number of classes
        self.assertFalse(validate_model_architecture(model, self.num_classes + 1))


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for model save/load functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 3
        self.model = create_model(self.num_classes, pretrained=False)
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_save_model_state(self):
        """Test model state saving."""
        additional_info = {'epoch': 10, 'loss': 0.5}
        
        save_model_state(self.model, self.model_path, additional_info)
        
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load and check contents
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('model_config', checkpoint)
        self.assertIn('parameter_counts', checkpoint)
        self.assertIn('epoch', checkpoint)
        self.assertIn('loss', checkpoint)
    
    def test_load_model_state(self):
        """Test model state loading."""
        # Save model first
        save_model_state(self.model, self.model_path)
        
        # Load model
        loaded_model, metadata = load_model_state(
            self.model_path, 
            self.num_classes, 
            device=torch.device('cpu')
        )
        
        self.assertIsInstance(loaded_model, ResNetClassifier)
        self.assertEqual(loaded_model.num_classes, self.num_classes)
        self.assertIsInstance(metadata, dict)
        self.assertIn('model_config', metadata)
    
    def test_load_model_state_wrong_classes(self):
        """Test loading model with wrong number of classes."""
        # Save model with original number of classes
        save_model_state(self.model, self.model_path)
        
        # Try to load with different number of classes
        with self.assertRaises(ValueError):
            load_model_state(self.model_path, self.num_classes + 1)


if __name__ == '__main__':
    unittest.main()