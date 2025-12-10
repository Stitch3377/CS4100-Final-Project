import torch
import sys
import os
from dotenv import load_dotenv

# Import your model class
# Adjust this import to match where your model.py actually is
try:
    from src.cnn.model import CrossViewProbabilityNet
except ImportError:
    # If import fails, we define a dummy version just to check expected keys
    print("Could not import model from src. Using expected structure check...")
    from torchvision import models
    import torch.nn as nn
    class CrossViewProbabilityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.map_encoder = nn.Sequential(nn.Conv2d(3,3,3)) # Dummy
            self.street_encoder = nn.Sequential(nn.Conv2d(3,3,3)) # Dummy
            self.map_safa = nn.Linear(1,1) # Dummy
            self.street_safa = nn.Linear(1,1) # Dummy
            self.proj = nn.Linear(1,1) # Dummy

load_dotenv()
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Path to your file
MODEL_PATH = os.getenv("CNN_PTH_SRC", "checkpoints/best_model.pth")

def check_keys():
    print(f"🔍 Inspecting file: {MODEL_PATH}\n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File not found at {MODEL_PATH}")
        return

    # 1. Load the File
    try:
        file_state = torch.load(MODEL_PATH, map_location='cpu')
        # If saved as full checkpoint dict, extract state_dict
        if 'state_dict' in file_state: 
            file_state = file_state['state_dict']
            print("   (Loaded from 'state_dict' key inside checkpoint)")
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # 2. Get Keys from File
    file_keys = list(file_state.keys())
    print(f"📂 Keys found in FILE (First 5):")
    for k in file_keys[:5]:
        print(f"   - {k}")

    # 3. Get Keys from Code
    model = CrossViewProbabilityNet()
    code_keys = list(model.state_dict().keys())
    print(f"\n💻 Keys expected by CODE (First 5):")
    for k in code_keys[:5]:
        print(f"   - {k}")

    # 4. Compare
    print("\n" + "="*40)
    print("RESULT:")
    if file_keys[0].split('.')[0] == code_keys[0].split('.')[0]:
        print("✅ MATCH! The file and code have the same structure.")
    else:
        print("❌ MISMATCH!")
        print(f"   The file starts with: '{file_keys[0].split('.')[0]}...' (Old Model?)")
        print(f"   The code expects:     '{code_keys[0].split('.')[0]}...' (New Model?)")
    print("="*40)

if __name__ == "__main__":
    check_keys()