import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import os
import torch.nn.functional as F
import sys

# --- Path Setup ---
# Ensure we can import the model from src/cnn/model.py
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try importing from src structure first
    from src.cnn.model import CrossViewProbabilityNet
except ImportError:
    # Fallback
    try:
        from model import CrossViewProbabilityNet
    except ImportError:
        print("❌ Error: Could not import CrossViewProbabilityNet.")
        sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"

class MatchPredictor:
    def __init__(self, model_path=MODEL_PATH, dist_threshold=0.8):
        """
        Args:
            model_path: Path to the trained .pth file.
            dist_threshold: The distance cutoff (from eval.py). 
                            Distances LOWER than this are matches.
        """
        print(f"Loading Inference Model on {DEVICE}...")
        
        # Initialize the ResNeXt-50 based model
        self.model = CrossViewProbabilityNet(embedding_dim=512).to(DEVICE)
        self.dist_threshold = dist_threshold
        
        # Load Weights
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            print("✅ Model weights loaded successfully.")
        else:
            print(f"⚠️ WARNING: Checkpoint {model_path} not found. Using random weights (Garbage output).")
        
        self.model.eval()
        
        # Standard normalization (No augmentation for inference)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, map_path, street_path, heading):
        # 1. Load Images (OpenCV loads BGR)
        map_img = cv2.imread(str(map_path))
        street_img = cv2.imread(str(street_path))

        if map_img is None or street_img is None:
            print(f"❌ Error loading: {map_path} or {street_path}")
            return None, None

        # 2. ROTATE MAP (Critical Alignment Step)
        if heading != 0:
            h, w = map_img.shape[:2]
            center = (w // 2, h // 2)
            # Rotate negative heading to align Map-North with Camera-Forward
            M = cv2.getRotationMatrix2D(center, -heading, 1.0) 
            map_img = cv2.warpAffine(map_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 3. Resize (Match Training Resolution)
        # Assuming you used 256x256 for map and 224x224 for street in loader.py
        map_img = cv2.resize(map_img, (256, 256))
        street_img = cv2.resize(street_img, (224, 224))

        # 4. Color Space (BGR -> RGB)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        street_img = cv2.cvtColor(street_img, cv2.COLOR_BGR2RGB)

        # 5. Transform & Batch
        map_tensor = self.transform(map_img).unsqueeze(0).to(DEVICE)
        street_tensor = self.transform(street_img).unsqueeze(0).to(DEVICE)

        return map_tensor, street_tensor

    def predict(self, map_path, street_path, heading=0.0):
        """
        Returns dictionary with match probability and decision.
        """
        with torch.no_grad():
            map_t, street_t = self.preprocess(map_path, street_path, heading)
            
            if map_t is None:
                return {"error": "Image load failed"}

            # Get embeddings
            map_emb, street_emb = self.model(map_t, street_t)
            
            # Calculate Euclidean Distance
            dist = F.pairwise_distance(map_emb, street_emb).item()
            
            # --- Score Calculation ---
            # Convert Distance (0 to ~2.0) to Probability (1.0 to 0.0)
            # Logic: If distance is 0, prob is 1. If distance is high, prob is low.
            probability = 1.0 / (1.0 + dist)
            
            # Decision
            is_match = dist < self.dist_threshold
            
            return {
                "is_match": is_match,
                "match_probability": round(probability, 4),
                "distance_score": round(dist, 4)
            }

if __name__ == "__main__":
    # Example Usage
    predictor = MatchPredictor(dist_threshold=0.9) # Adjust threshold based on eval.py results
    
    # Replace with real paths from your data folder
    res = predictor.predict(
        map_path="data/metadata_out/test_samples/sample_map.png", 
        street_path="data/metadata_out/test_samples/sample_street.jpg",
        heading=0.0
    )
    
    print(f"Prediction Result: {res}")