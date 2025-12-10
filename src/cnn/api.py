import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

from cnn.inference import MatchPredictor

load_dotenv()

CHECKPOINT_PATH = os.getenv("CNN_PTH_SRC")
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT")
# src/cnn/api.py

class StreetViewMatcher:
    def __init__(self, model_path=None, map_dir=None, device=None):
        """
        Initializes the model once to avoid reloading overhead.
        """
        self.map_dir = map_dir 
        
        # --- FIX 1: Initialize the Predictor ONCE here ---
        # We load the heavy model into memory one time when the app starts.
        print("Initializing StreetViewMatcher and loading model...")
        self.predictor = MatchPredictor(
            model_path=model_path, 
            dist_threshold=0.6
        )
    
    def get_match_probability(self, street_view_input, map_tile_index):
        """
        API Function to calculate match probability.
        """
        map_filename = f"state_{map_tile_index}.png"
        map_path = os.path.join(self.map_dir, map_filename)

        # --- FIX 2: Use the pre-loaded self.predictor ---
        # Do NOT do: predictor = MatchPredictor(...) here.
        
        res = self.predictor.predict(
            map_path=map_path, 
            street_path=street_view_input,
            heading=0.0
        )

        # Keep your previous fix for the float return type
        if isinstance(res, dict):
            return res['match_probability'] # Or 'score', check your keys

        return res

# --- Usage Example ---
if __name__ == "__main__":

    # CHECKPOINT_PATH = os.getenv("CNN_PTH_SRC")
    # MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT")

    # 1. Initialize the matcher (Do this once at app startup)
    matcher = StreetViewMatcher(
        model_path=CHECKPOINT_PATH,
        map_dir=MAP_CACHE_PATH
    )

    # 2. Call the API
    # Example: Check if a street view matches map tile #226
    street_view_1 = "data/Heading/p1801_42.33347129_-71.09045776_251.00172143759187/gsv_0.jpg" 
    map_idx_1 = 43
    map_idx_2 = 44
    map_idx_3 = 100
    
    # Note: If testing without real files, this will return 0.0 due to file not found catch blocks
    try:
        # You can pass a path...
        prob = matcher.get_match_probability(street_view_1, map_idx_1)
        print(f"Probability of match (Path Input): {prob:.4f}")

        prob = matcher.get_match_probability(street_view_1, map_idx_2)
        print(f"Probability of match (Path Input): {prob:.4f}")

        prob = matcher.get_match_probability(street_view_1, map_idx_3)
        print(f"Probability of match (Path Input): {prob:.4f}")
        
        # ... or a PIL image directly
        # img_obj = Image.open(street_view_path)
        # prob_obj = matcher.get_match_probability(img_obj, map_idx)
        # print(f"Probability of match (PIL Input): {prob_obj:.4f}")
        
    except Exception as e:
        print(f"Could not run test: {e}")