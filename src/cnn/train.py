import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# --- Path Setup ---
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.cnn.model import CrossViewProbabilityNet
    from src.data.loader import CrossViewDataset
except ImportError:
    from model import CrossViewProbabilityNet
    from loader import CrossViewDataset

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
BATCH_SIZE = 32          # Higher is better for Triplet Loss (more negatives to choose from)
LEARNING_RATE = 0.0001
EPOCHS = 20              # Triplet loss converges slower but better
MARGIN = 0.3             # Distance margin (Matches must be 0.3 closer than non-matches)
CHECKPOINT_DIR = Path("checkpoints")

JSON_OUT_DIR = os.getenv("METADATA_OUT", 'data/metadata_out')
TRAIN_JSON = os.path.join(JSON_OUT_DIR, 'train_pairs.json')
TEST_JSON = os.path.join(JSON_OUT_DIR, 'test_pairs.json')

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"✅ Using Device: {DEVICE}")

# --- Batch Hard Triplet Loss ---
def batch_hard_triplet_loss(map_emb, street_emb, margin=0.3):
    """
    Mining the hardest negatives within the batch.
    """
    # 1. Compute pairwise distance matrix
    # Shape: (B, B) where entry [i, j] is dist(map_i, street_j)
    dists = torch.cdist(map_emb, street_emb, p=2)
    
    # 2. Get Positives (Diagonal elements)
    # dist(map_i, street_i)
    pos_dists = torch.diag(dists)
    
    # 3. Get Hardest Negatives
    # For each map_i, find the street_j (where i != j) that is CLOSEST to map_i
    # We add a large value to the diagonal so we don't pick the positive as the negative
    batch_size = map_emb.size(0)
    eye = torch.eye(batch_size, device=dists.device)
    inf_diag = eye * 1e9
    
    # closest negative for each map
    neg_dists_map, _ = torch.min(dists + inf_diag, dim=1) 
    
    # closest negative for each street (symmetric mining)
    neg_dists_street, _ = torch.min(dists + inf_diag, dim=0)
    
    # 4. Compute Triplet Loss
    # L = max(0, Pos - Neg + Margin)
    loss_map = F.relu(pos_dists - neg_dists_map + margin)
    loss_street = F.relu(pos_dists - neg_dists_street + margin)
    
    return loss_map.mean() + loss_street.mean()

# --- Training Loop ---
def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {JSON_OUT_DIR}...")
    if not os.path.exists(TRAIN_JSON):
        print(f"❌ Error: {TRAIN_JSON} not found.")
        return

    # 1. Load Data (Positive Pairs Only)
    # We filter out the static negatives because we will generate harder ones in the batch
    full_train_dataset = CrossViewDataset(TRAIN_JSON, mode='train')
    
    # Filter: Keep only label == 1
    print("   -> Filtering dataset to keep only POSITIVE pairs for Triplet Training...")
    full_train_dataset.pairs = [p for p in full_train_dataset.pairs if p['label'] == 1]
    print(f"   -> Training Samples: {len(full_train_dataset)}")

    test_dataset = CrossViewDataset(TEST_JSON, mode='val')

    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Model
    print("Initializing SAFA Model...")
    model = CrossViewProbabilityNet(embedding_dim=512).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_f1 = 0.0

    # 3. Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for map_imgs, street_imgs, _ in pbar:
            # Note: We ignore labels here because we know everything in the loader is a Match (1)
            # The negatives are implicitly the other items in the batch.
            map_imgs, street_imgs = map_imgs.to(DEVICE), street_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            
            map_emb, street_emb = model(map_imgs, street_imgs)
            
            loss = batch_hard_triplet_loss(map_emb, street_emb, margin=MARGIN)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # 4. Validation
        print(f"Validating Epoch {epoch+1}...")
        val_metrics = validate(model, test_loader)
        
        # Determine validation loss proxy (using negative F1 or custom)
        # Here we just step scheduler based on F1 plateauing (inverted)
        scheduler.step(1.0 - val_metrics['f1'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"  Current LR: {current_lr:.6f}")
        print(f"  Best Thresh:{val_metrics['threshold']:.4f}")
        print(f"  Accuracy:   {val_metrics['acc']:.4f}")
        print(f"  F1 Score:   {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  --> ✅ New Best Model Saved!")

def validate(model, loader):
    model.eval()
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for map_imgs, street_imgs, labels in loader:
            map_imgs, street_imgs, labels = map_imgs.to(DEVICE), street_imgs.to(DEVICE), labels.to(DEVICE)
            
            map_emb, street_emb = model(map_imgs, street_imgs)
            dist = F.pairwise_distance(map_emb, street_emb)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    # Find Optimal Threshold
    thresholds = np.arange(0.0, 2.0, 0.05)
    best_f1 = 0
    best_acc = 0
    best_thresh = 0
    
    for t in thresholds:
        preds = (all_distances < t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        acc = accuracy_score(all_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_thresh = t

    return {
        'f1': best_f1,
        'acc': best_acc,
        'threshold': best_thresh
    }

if __name__ == "__main__":
    train()