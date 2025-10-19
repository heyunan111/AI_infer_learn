"""
Custom dataset classes for image classification with text labels.

This module provides dataset classes for loading images with text-based labels,
including improved error handling and documentation.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageFolderWithTxt(Dataset):
    """
    Custom dataset class for loading images with text labels from a CSV file.
    
    This dataset reads image paths and labels from a text file and loads images
    from a specified directory. It provides improved error handling and validation
    compared to the original implementation.
    
    Args:
        root_dir (str): Path to the directory containing images
        txt_path (str): Path to the CSV file containing image names and labels
        transform (Optional[callable]): Optional transform to be applied to images
        
    Attributes:
        root_dir (str): Root directory for images
        transform (Optional[callable]): Image transformation pipeline
        samples (List[Tuple[str, str]]): List of (image_name, label) pairs
        label_to_id (Dict[str, int]): Mapping from label names to integer IDs
        id_to_label (Dict[int, str]): Mapping from integer IDs to label names
        
    Raises:
        FileNotFoundError: If the text file or root directory doesn't exist
        ValueError: If the text file format is invalid
    """
    
    def __init__(self, root_dir: str, txt_path: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Validate inputs
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Label file not found: {txt_path}")
        
        # Load samples from text file
        self.samples = self._load_samples(txt_path)
        
        # Create label mappings
        self._create_label_mappings()
        
        logger.info(f"Loaded dataset with {len(self.samples)} samples and {len(self.label_to_id)} classes")
    
    def _load_samples(self, txt_path: str) -> List[Tuple[str, str]]:
        """
        Load image-label pairs from text file.
        
        Args:
            txt_path (str): Path to the text file
            
        Returns:
            List[Tuple[str, str]]: List of (image_name, label) pairs
            
        Raises:
            ValueError: If file format is invalid
        """
        samples = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse line
                    parts = line.split(',')
                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid format at line {line_num}: expected 'image,label', got '{line}'"
                        )
                    
                    img_name, label = parts[0].strip(), parts[1].strip()
                    if not img_name or not label:
                        raise ValueError(f"Empty image name or label at line {line_num}")
                    
                    samples.append((img_name, label))
                    
        except Exception as e:
            raise ValueError(f"Error reading label file {txt_path}: {str(e)}")
        
        if not samples:
            raise ValueError(f"No valid samples found in {txt_path}")
        
        return samples
    
    def _create_label_mappings(self) -> None:
        """Create bidirectional mappings between labels and integer IDs."""
        all_labels = sorted(set(label for _, label in self.samples))
        self.label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'image': Transformed image tensor
                - 'label': Label as integer tensor
                - 'filename': Original filename
                
        Raises:
            IndexError: If index is out of range
            FileNotFoundError: If image file doesn't exist
            Exception: If image cannot be loaded
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Check if image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert("RGB")
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            # Get label ID
            label_id = self.label_to_id[label]
            
            return {
                "image": image,
                "label": torch.tensor(label_id, dtype=torch.long),
                "filename": img_name
            }
            
        except Exception as e:
            raise Exception(f"Error loading image {img_path}: {str(e)}")
    
    def set_transform(self, transform: Optional[callable]) -> None:
        """
        Dynamically set the transform for the dataset.
        
        Args:
            transform (Optional[callable]): New transform to apply to images
        """
        self.transform = transform
        logger.info(f"Updated transform for dataset")
    
    def get_class_names(self) -> List[str]:
        """
        Get list of class names in order of their IDs.
        
        Returns:
            List[str]: List of class names sorted by ID
        """
        return [self.id_to_label[i] for i in range(len(self.id_to_label))]
    
    def get_class_counts(self) -> Dict[str, int]:
        """
        Get count of samples for each class.
        
        Returns:
            Dict[str, int]: Dictionary mapping class names to sample counts
        """
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


class TransformDataset(Dataset):
    """
    Wrapper dataset class that applies transforms to a subset of data.
    
    This class provides a cleaner interface for applying different transforms
    to train/validation splits of the same underlying dataset.
    
    Args:
        subset (Subset): PyTorch Subset object containing indices
        transform (Optional[callable]): Transform to apply to images
        
    Attributes:
        subset (Subset): The underlying data subset
        transform (Optional[callable]): Image transformation pipeline
    """
    
    def __init__(self, subset: Subset, transform: Optional[callable] = None):
        if not isinstance(subset, Subset):
            raise TypeError("subset must be a torch.utils.data.Subset instance")
        
        self.subset = subset
        self.transform = transform
        
        logger.info(f"Created TransformDataset with {len(subset)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.subset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the subset with applied transforms.
        
        Args:
            idx (int): Index within the subset
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'image': Transformed image tensor
                - 'label': Label as integer tensor
                - 'filename': Original filename
                
        Raises:
            IndexError: If index is out of range
            Exception: If image cannot be loaded or transformed
        """
        if idx >= len(self.subset):
            raise IndexError(f"Index {idx} out of range for subset of size {len(self.subset)}")
        
        # Get the actual dataset index
        actual_idx = self.subset.indices[idx]
        
        # Get sample info from original dataset
        img_name, label = self.subset.dataset.samples[actual_idx]
        img_path = os.path.join(self.subset.dataset.root_dir, img_name)
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            
            # Get label ID
            label_id = self.subset.dataset.label_to_id[label]
            
            return {
                "image": image,
                "label": torch.tensor(label_id, dtype=torch.long),
                "filename": img_name
            }
            
        except Exception as e:
            raise Exception(f"Error processing sample {idx} (image: {img_path}): {str(e)}")
    
    def set_transform(self, transform: Optional[callable]) -> None:
        """
        Set the transform for this dataset wrapper.
        
        Args:
            transform (Optional[callable]): New transform to apply
        """
        self.transform = transform
        logger.info(f"Updated transform for TransformDataset")
    
    @property
    def dataset(self) -> Dataset:
        """Get the underlying dataset."""
        return self.subset.dataset
    
    @property
    def indices(self) -> List[int]:
        """Get the indices used by this subset."""
        return self.subset.indices