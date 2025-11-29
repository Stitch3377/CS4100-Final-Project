import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

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
BATCH_SIZE = 32 # Can be higher for eval since no gradients
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
JSON_OUT_DIR = os.getenv("METADATA_OUT", 'data/metadata_out')
TEST_JSON = os.path.join(JSON_OUT_DIR, 'test_pairs.json')

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def evaluate_performance():
    print(f"📂 Loading Test Data from {TEST_JSON}...")
    if not os.path.exists(TEST_JSON):
        print(f"❌ Error: {TEST_JSON} not found.")
        return

    # Mode='val' ensures no augmentation (deterministic)
    test_dataset = CrossViewDataset(TEST_JSON, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"🧠 Loading ResNeXt-50 Model from {CHECKPOINT_PATH}...")
    model = CrossViewProbabilityNet(embedding_dim=512).to(DEVICE)
    
    if CHECKPOINT_PATH.exists():
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("❌ Error: Checkpoint not found.")
        return

    model.eval()

    all_distances = []
    all_labels = []

    print("🚀 Running Inference on Test Set...")
    with torch.no_grad():
        for map_imgs, street_imgs, labels in tqdm(test_loader, desc="Eval"):
            map_imgs = map_imgs.to(DEVICE)
            street_imgs = street_imgs.to(DEVICE)
            
            # Get embeddings
            map_emb, street_emb = model(map_imgs, street_imgs)
            
            # Calculate Euclidean Distance
            dist = F.pairwise_distance(map_emb, street_emb)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # --- THRESHOLD OPTIMIZATION ---
    print("\n📊 Scanning for Optimal Threshold...")
    
    # Scan distances from 0.0 (perfect match) to 2.0 (max dist)
    thresholds = np.arange(0.0, 2.0, 0.05)
    
    best_f1 = 0
    best_metrics = {}
    best_thresh_dist = 0.0

    print(f"{'Dist Thresh':<12} | {'Prob Thresh':<12} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 60)

    for dist_t in thresholds:
        # Distance Logic: Match if dist < threshold
        preds = (all_distances < dist_t).astype(int)
        
        # Calculate Probability Threshold equivalent: P = 1 / (1 + dist)
        prob_t = 1.0 / (1.0 + dist_t)
        
        f1 = f1_score(all_labels, preds, zero_division=0)
        acc = accuracy_score(all_labels, preds)
        
        # Print valid candidates
        if f1 > 0.1 and dist_t % 0.2 == 0:
             print(f"{dist_t:.2f}         | {prob_t:.2f}         | {acc*100:.1f}%      | {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'acc': acc,
                'prec': precision_score(all_labels, preds, zero_division=0),
                'rec': recall_score(all_labels, preds, zero_division=0)
            }
            best_thresh_dist = dist_t

    best_thresh_prob = 1.0 / (1.0 + best_thresh_dist)

    print("\n" + "="*50)
    print(f"🏆 BEST RESULTS")
    print("="*50)
    print(f"Optimal Distance Cutoff:    {best_thresh_dist:.2f} (Lower is better)")
    print(f"Optimal Probability Cutoff: {best_thresh_prob:.2f} (Higher is better)")
    print("-" * 50)
    print(f"F1 Score:    {best_f1:.4f}")
    print(f"Accuracy:    {best_metrics['acc']*100:.2f}%")
    print(f"Precision:   {best_metrics['prec']*100:.2f}%")
    print(f"Recall:      {best_metrics['rec']*100:.2f}%")
    print("="*50)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    # Add density=True to make height relative to percentage, not raw count
    plt.hist(all_distances[all_labels==1], bins=50, alpha=0.5, color='green', 
             label='Matches (Positives)', density=True)
    
    plt.hist(all_distances[all_labels==0], bins=50, alpha=0.5, color='red', 
             label='Non-Matches (Negatives)', density=True)
    
    plt.axvline(best_thresh_dist, color='black', linestyle='--', linewidth=2, label=f'Cutoff ({best_thresh_dist:.2f})')
    
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density (Frequency)')  # Changed label
    plt.title(f'Model Separation (F1: {best_f1:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = 'distance_separation.png'
    plt.savefig(out_file)
    print(f"📈 Saved separation plot to {out_file}")

if __name__ == "__main__":
    evaluate_performance()