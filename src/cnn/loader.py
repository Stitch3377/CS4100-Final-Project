import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

class CrossViewDataset(Dataset):
    def __init__(self, pairs_file, mode='train', transform=None):
        """
        Args:
            pairs_file (str): Path to the json file containing image pairs.
            mode (str): 'train' or 'val'. 
                        If 'train', applies ColorJitter and RandomErasing.
            transform (callable, optional): Custom transform to override defaults.
        """
        self.mode = mode
        
        # Load the JSON metadata
        with open(pairs_file, 'r') as f:
            self.pairs = json.load(f)
            
        # --- Transform Pipelines ---
        if transform:
            self.transform = transform
        else:
            # 1. Base Normalization (ImageNet stats)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            
            if self.mode == 'train':
                # --- TRAINING AUGMENTATION ---
                # Forces model to learn structure, not just colors/textures
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    # Photometric Distortions
                    transforms.ColorJitter(
                        brightness=0.25, 
                        contrast=0.25, 
                        saturation=0.25, 
                        hue=0.05
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    normalize,
                    # Occlusion (Simulates trees/obstacles blocking view)
                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
                ])
            else:
                # --- VALIDATION / TEST (Deterministic) ---
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    normalize
                ])

    def rotate_image(self, image, angle):
        """
        Rotates map image to align Map-Up with Camera-Forward.
        """
        if angle == 0: return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation Matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Border Reflect prevents black corners when rotating
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        # 1. Get Paths
        map_path = item['map_image_path']
        street_path = item['street_view_path']
        
        # 2. Load Images (OpenCV loads in BGR)
        map_img = cv2.imread(map_path)
        street_img = cv2.imread(street_path)
        
        # Safety Check: Return zeros if image load fails
        if map_img is None or street_img is None:
            # print(f"Warning: Could not load {map_path} or {street_path}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 224, 224), torch.tensor(0.0)

        # 3. GEOMETRIC ALIGNMENT
        # Rotate map so "Up" in map = "Forward" in street view
        heading = item.get('heading', 0.0)
        map_img = self.rotate_image(map_img, heading)

        # 4. Resize
        # Map: 256x256 (provides context)
        # Street: 224x224 (standard ResNet input)
        map_img = cv2.resize(map_img, (256, 256))
        street_img = cv2.resize(street_img, (224, 224))
        
        # 5. Color Space Conversion (BGR -> RGB)
        # Essential because PyTorch transforms (and PIL) expect RGB
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        street_img = cv2.cvtColor(street_img, cv2.COLOR_BGR2RGB)
        
        # 6. Apply Transforms (Augmentation or Normalization)
        map_tensor = self.transform(map_img)
        street_tensor = self.transform(street_img)
            
        # 7. Label
        label = torch.tensor(float(item['label']), dtype=torch.float32)
        
        return map_tensor, street_tensor, label