"""
Unit tests for dataset module.

Tests the ImageFolderWithTxt and TransformDataset classes.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import Subset
from PIL import Image
import numpy as np

from src.data.dataset import ImageFolderWithTxt, TransformDataset


class TestImageFolderWithTxt(unittest.TestCase):
    """Test cases for ImageFolderWithTxt dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create dummy CSV file
        self.csv_path = os.path.join(self.temp_dir, "labels.csv")
        with open(self.csv_path, 'w') as f:
            f.write("image1.jpg,class_a\n")
            f.write("image2.jpg,class_b\n")
            f.write("image3.jpg,class_a\n")
        
        # Create dummy images
        self.image_paths = []
        for i in range(1, 4):
            img_path = os.path.join(self.images_dir, f"image{i}.jpg")
            # Create a simple RGB image
            img = Image.new('RGB', (64, 64), color=(i*50, i*50, i*50))
            img.save(img_path)
            self.image_paths.append(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up files
        for img_path in self.image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.images_dir):
            os.rmdir(self.images_dir)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = ImageFolderWithTxt(self.images_dir, self.csv_path)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.label_to_id), 2)  # class_a, class_b
        self.assertIn('class_a', dataset.label_to_id)
        self.assertIn('class_b', dataset.label_to_id)
    
    def test_dataset_invalid_paths(self):
        """Test dataset with invalid paths."""
        with self.assertRaises(FileNotFoundError):
            ImageFolderWithTxt("/nonexistent", self.csv_path)
        
        with self.assertRaises(FileNotFoundError):
            ImageFolderWithTxt(self.images_dir, "/nonexistent.csv")
    
    def test_dataset_invalid_csv_format(self):
        """Test dataset with invalid CSV format."""
        invalid_csv = os.path.join(self.temp_dir, "invalid.csv")
        with open(invalid_csv, 'w') as f:
            f.write("invalid_line_without_comma\n")
        
        with self.assertRaises(ValueError):
            ImageFolderWithTxt(self.images_dir, invalid_csv)
        
        os.remove(invalid_csv)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        dataset = ImageFolderWithTxt(self.images_dir, self.csv_path)
        
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertIn('filename', sample)
        
        self.assertIsInstance(sample['image'], Image.Image)
        self.assertIsInstance(sample['label'], torch.Tensor)
        self.assertIsInstance(sample['filename'], str)
    
    def test_dataset_getitem_with_transform(self):
        """Test dataset __getitem__ with transform."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        dataset = ImageFolderWithTxt(self.images_dir, self.csv_path, transform=transform)
        sample = dataset[0]
        
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape, (3, 32, 32))
    
    def test_dataset_label_mappings(self):
        """Test label mapping functionality."""
        dataset = ImageFolderWithTxt(self.images_dir, self.csv_path)
        
        class_names = dataset.get_class_names()
        self.assertEqual(len(class_names), 2)
        self.assertIn('class_a', class_names)
        self.assertIn('class_b', class_names)
        
        class_counts = dataset.get_class_counts()
        self.assertEqual(class_counts['class_a'], 2)
        self.assertEqual(class_counts['class_b'], 1)
    
    def test_dataset_set_transform(self):
        """Test dynamic transform setting."""
        from torchvision import transforms
        
        dataset = ImageFolderWithTxt(self.images_dir, self.csv_path)
        
        # Initially no transform
        sample = dataset[0]
        self.assertIsInstance(sample['image'], Image.Image)
        
        # Set transform
        transform = transforms.ToTensor()
        dataset.set_transform(transform)
        
        sample = dataset[0]
        self.assertIsInstance(sample['image'], torch.Tensor)


class TestTransformDataset(unittest.TestCase):
    """Test cases for TransformDataset wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create dummy CSV file
        self.csv_path = os.path.join(self.temp_dir, "labels.csv")
        with open(self.csv_path, 'w') as f:
            f.write("image1.jpg,class_a\n")
            f.write("image2.jpg,class_b\n")
            f.write("image3.jpg,class_a\n")
        
        # Create dummy images
        self.image_paths = []
        for i in range(1, 4):
            img_path = os.path.join(self.images_dir, f"image{i}.jpg")
            img = Image.new('RGB', (64, 64), color=(i*50, i*50, i*50))
            img.save(img_path)
            self.image_paths.append(img_path)
        
        # Create base dataset and subset
        self.base_dataset = ImageFolderWithTxt(self.images_dir, self.csv_path)
        self.subset = Subset(self.base_dataset, [0, 2])  # Only indices 0 and 2
    
    def tearDown(self):
        """Clean up test fixtures."""
        for img_path in self.image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.images_dir):
            os.rmdir(self.images_dir)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_transform_dataset_initialization(self):
        """Test TransformDataset initialization."""
        transform_dataset = TransformDataset(self.subset)
        
        self.assertEqual(len(transform_dataset), 2)
        self.assertEqual(transform_dataset.indices, [0, 2])
    
    def test_transform_dataset_invalid_subset(self):
        """Test TransformDataset with invalid subset."""
        with self.assertRaises(TypeError):
            TransformDataset("not_a_subset")
    
    def test_transform_dataset_getitem(self):
        """Test TransformDataset __getitem__ method."""
        transform_dataset = TransformDataset(self.subset)
        
        sample = transform_dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertIn('filename', sample)
        
        # Should correspond to original dataset index 0
        original_sample = self.base_dataset[0]
        self.assertEqual(sample['filename'], original_sample['filename'])
    
    def test_transform_dataset_with_transform(self):
        """Test TransformDataset with transform."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        transform_dataset = TransformDataset(self.subset, transform=transform)
        sample = transform_dataset[0]
        
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape, (3, 32, 32))
    
    def test_transform_dataset_set_transform(self):
        """Test dynamic transform setting."""
        from torchvision import transforms
        
        transform_dataset = TransformDataset(self.subset)
        
        # Initially no transform
        sample = transform_dataset[0]
        self.assertIsInstance(sample['image'], Image.Image)
        
        # Set transform
        transform = transforms.ToTensor()
        transform_dataset.set_transform(transform)
        
        sample = transform_dataset[0]
        self.assertIsInstance(sample['image'], torch.Tensor)
    
    def test_transform_dataset_properties(self):
        """Test TransformDataset properties."""
        transform_dataset = TransformDataset(self.subset)
        
        self.assertEqual(transform_dataset.dataset, self.base_dataset)
        self.assertEqual(transform_dataset.indices, [0, 2])


if __name__ == '__main__':
    unittest.main()