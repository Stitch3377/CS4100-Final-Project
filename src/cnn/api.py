import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_PATH = os.getenv("CNN_PTH_SRC")
MAP_CACHE_PATH = os.getenv("MAP_TILE_IMG_OUT")

# --- 1. Model Architecture (Must match training exactly) ---
class ObservationProbabilityModel(nn.Module):
    def __init__(self):
        super(ObservationProbabilityModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            
            nn.Flatten()
        )
        
        self.flattened_size = 128 * 32 * 32
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, map_img, street_img):
        map_features = self.cnn(map_img)
        street_features = self.cnn(street_img)
        combined = torch.cat((map_features, street_features), dim=1)
        logits = self.classifier(combined)
        return logits.squeeze()

# --- 2. The API Class ---
class StreetViewMatcher:
    def __init__(self, model_path=None, map_dir=None, device=None):
        """
        Initializes the model once to avoid reloading overhead.
        
        Args:
            model_path (str): Path to the .pth checkpoint file.
            map_dir (str): Directory where map tiles are stored (e.g., 'data/maps').
            device (str): 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
    
        
        # specific defaults if arguments not provided
        self.model_path = model_path 
        self.map_dir = map_dir 
        
        # Device configuration
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() 
                else "cuda" if torch.cuda.is_available() 
                else "cpu"
            )
            
        print(f"Initializing Matcher on {self.device}...")
        
        # Load Model
        self.model = ObservationProbabilityModel().to(self.device)
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # CRITICAL: Set to eval mode for inference
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        # Define Transforms (Must match training)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_match_probability(self, street_view_input, map_tile_index):
        """
        API Function to calculate match probability.

        Args:
            street_view_input (str or PIL.Image): Path to street view image OR PIL Image object.
            map_tile_index (int or str): The index/ID of the map tile (e.g., 226 for 'state_226.png').

        Returns:
            float: Probability (0.0 to 1.0) that the street view belongs to the map tile.
        """
        
        # 1. Load Map Image
        # Assumes naming convention: state_{index}.png
        map_filename = f"state_{map_tile_index}.png"
        map_path = os.path.join(self.map_dir, map_filename)
        
        try:
            map_img = Image.open(map_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Map tile {map_path} not found.")
            return 0.0

        # 2. Load Street View Image
        if isinstance(street_view_input, str):
            try:
                street_img = Image.open(street_view_input).convert("RGB")
            except FileNotFoundError:
                print(f"Error: Street view file {street_view_input} not found.")
                return 0.0
        elif isinstance(street_view_input, Image.Image):
            street_img = street_view_input.convert("RGB")
        else:
            raise ValueError("street_view_input must be a file path string or PIL Image")

        # 3. Preprocess
        # Add batch dimension (unsqueeze) because model expects [B, C, H, W]
        map_tensor = self.transform(map_img).unsqueeze(0).to(self.device)
        street_tensor = self.transform(street_img).unsqueeze(0).to(self.device)

        # 4. Inference
        with torch.no_grad():
            logits = self.model(map_tensor, street_tensor)
            probability = torch.sigmoid(logits).item()

        return probability

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