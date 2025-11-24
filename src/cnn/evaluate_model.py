import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
import numpy as np

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Get paths (same as training)
JSON_OUT_DIR = os.getenv("JSON_OUT", "data")
TEST_JSON = os.path.join(JSON_OUT_DIR, "test_pairs.json")
CHECKPOINT_PATH = "observation_model.pth"


# --- 1. Dataset (same as training) ---
class MapStreetDataset(Dataset):
    def __init__(self, json_path, transform=None):
        try:
            with open(json_path, "r") as f:
                self.pairs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {json_path}")

        self.transform = transform
        # Keep labels as ints only if you ever need them, not strictly required here
        self.labels = [int(p["label"]) for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            map_img = Image.open(item["map_image_path"]).convert("RGB")
            street_img = Image.open(item["street_view_path"]).convert("RGB")
        except Exception:
            # Fallback blank images if something is missing/broken
            map_img = Image.new("RGB", IMAGE_SIZE)
            street_img = Image.new("RGB", IMAGE_SIZE)

        if self.transform:
            map_img = self.transform(map_img)
            street_img = self.transform(street_img)

        return map_img, street_img, torch.tensor(float(item["label"]), dtype=torch.float32)


# --- 2. Model (same as training) ---
class ObservationProbabilityModel(nn.Module):
    def __init__(self):
        super(ObservationProbabilityModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

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


# --- 3. Evaluation Logic ---
def evaluate_model(checkpoint_path=CHECKPOINT_PATH, threshold=0.5):
    print(f"Using device: {DEVICE}")
    print(f"Loading test dataset from: {TEST_JSON}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = MapStreetDataset(TEST_JSON, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = ObservationProbabilityModel().to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded weights from: {checkpoint_path}")

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for map_imgs, street_imgs, labels in test_loader:
            map_imgs = map_imgs.to(DEVICE)
            street_imgs = street_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(map_imgs, street_imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    # Metrics
    total = tp + tn + fp + fn
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("===== TEST EVALUATION =====")
    print(f"Loss      : {avg_loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    print("===========================")


if __name__ == "__main__":
    evaluate_model()
