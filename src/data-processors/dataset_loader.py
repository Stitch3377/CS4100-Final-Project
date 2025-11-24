"""
PyTorch Dataset class for loading dual-CNN training data
Loads map tiles and street view images with labels
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class DualCNNDataset(Dataset):
    """
    PyTorch Dataset for dual-CNN.
    Loads (map_tile, street_view_image) pairs with binary labels.
    """
    def __init__(self, json_path, map_transform=None, street_view_transform=None):
        """
        Initialize dataset.
        
        Args:
            json_path: Path to train_pairs.json or test_pairs.json
            map_transform: torchvision transforms for map tiles
            street_view_transform: torchvision transforms for street view images
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Default transforms if none provided
        if map_transform is None:
            self.map_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.map_transform = map_transform

        if street_view_transform is None:
            self.street_view_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.street_view_transform = street_view_transform

    def __len__(self):
        """Return total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict: {
                'map_image': torch.Tensor (C, H, W),
                'street_view_image': torch.Tensor (C, H, W),
                'label': torch.Tensor (binary: 0 or 1),
                'state_id': int,
                'metadata': dict (optional additional info)
            }
        """
        sample = self.data[idx]

        # Load images
        map_image = Image.open(sample['map_image_path']).convert('RGB')
        street_view_image = Image.open(sample['street_view_path']).convert('RGB')

        # Apply transforms
        map_tensor = self.map_transform(map_image)
        street_view_tensor = self.street_view_transform(street_view_image)

        # Get lable
        label = torch.tensor(sample['label'], dtype=torch.float32)

        return {
            'map_image': map_tensor,
            'street_view_image': street_view_tensor,
            'label': label,
            'state_id': sample['state_id'],
            'metadata': {
                'map_center_lat': sample['map_center_lat'],
                'map_center_lon': sample['map_center_lon'],
                'observation_lat': sample['observation_lat'],
                'observation_lon': sample['observation_lon'],
                'pair_type': sample['pair_type']
            }
        }
    
def create_dataloaders(train_json='data/train_pairs.json',
                      test_json='data/test_pairs.json',
                      batch_size=32,
                      num_workers=4,
                      map_transform=None,
                      street_view_transform=None):
    """
    Create train and test DataLoaders
    
    Args:
        train_json: Path to training data JSON
        test_json: Path to test data JSON
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        map_transform: Custom transforms for map tiles
        street_view_transform: Custom transforms for street view images
        
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = DualCNNDataset(
        train_json,
        map_transform=map_transform,
        street_view_transform=street_view_transform
    )

    test_dataset = DualCNNDataset(
        test_json,
        map_transform=map_transform,
        street_view_transform=street_view_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
