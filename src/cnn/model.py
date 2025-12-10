import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SAFA(nn.Module):
    """
    Spatial Awareness Feature Aggregation.
    """
    def __init__(self, in_channels):
        super(SAFA, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1) 
        )

    def forward(self, x):
        # x: (B, C, H, W)
        mask = self.attn(x)            
        mask = mask.view(mask.size(0), -1)  
        weights = F.softmax(mask, dim=1)    
        weights = weights.view(x.size(0), 1, x.size(2), x.size(3)) 
        
        weighted_features = x * weights 
        aggregated = torch.sum(weighted_features.view(x.size(0), x.size(1), -1), dim=2) 
        
        return aggregated

class CrossViewProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(CrossViewProbabilityNet, self).__init__()
        
        # --- UPGRADE: ResNeXt-50 (32x4d) ---
        # "32x4d" means 32 parallel groups. This architecture generalizes better
        # than standard ResNet50.
        print("   -> Loading ResNeXt-50 Backbone...")
        resnext_map = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        resnext_street = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        
        # Remove FC and AvgPool to keep spatial features (B, 2048, 8, 8)
        self.map_encoder = nn.Sequential(*list(resnext_map.children())[:-2])
        self.street_encoder = nn.Sequential(*list(resnext_street.children())[:-2])
        
        # ResNeXt-50 outputs 2048 channels
        self.map_safa = SAFA(in_channels=2048)
        self.street_safa = SAFA(in_channels=2048)
        
        # --- ADDED DROPOUT ---
        # Helps prevent the overfitting you saw (Train Loss 0.04)
        self.dropout = nn.Dropout(p=0.3)
        
        self.proj = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, map_img, street_img):
        # 1. Extract Features
        map_feat = self.map_encoder(map_img)
        street_feat = self.street_encoder(street_img)
        
        # 2. SAFA Aggregation
        map_vec = self.map_safa(map_feat)
        street_vec = self.street_safa(street_feat)
        
        # 3. Dropout (Regularization)
        map_vec = self.dropout(map_vec)
        street_vec = self.dropout(street_vec)
        d
        # 4. Project & Normalize
        map_emb = F.normalize(self.proj(map_vec), p=2, dim=1)
        street_emb = F.normalize(self.proj(street_vec), p=2, dim=1)
        
        return map_emb, street_emb