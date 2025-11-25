import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
import numpy as np

# --- Load Environment Variables ---
load_dotenv() 

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # Kept low for stability
EPOCHS = 10
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Get paths
JSON_OUT_DIR = os.getenv("METADATA_OUT", 'data')
TRAIN_JSON = os.path.join(JSON_OUT_DIR, 'train_pairs.json')
TEST_JSON = os.path.join(JSON_OUT_DIR, 'test_pairs.json')

print(f"Using device: {DEVICE}")

# --- 1. The Dataset Class ---
class MapStreetDataset(Dataset):
    def __init__(self, json_path, transform=None):
        try:
            with open(json_path, 'r') as f:
                self.pairs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {json_path}")
            
        self.transform = transform
        # Ensure labels are stored as integers for the sampler
        self.labels = [int(p['label']) for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            map_img = Image.open(item['map_image_path']).convert('RGB')
            street_img = Image.open(item['street_view_path']).convert('RGB')
        except Exception:
            map_img = Image.new('RGB', IMAGE_SIZE)
            street_img = Image.new('RGB', IMAGE_SIZE)

        if self.transform:
            map_img = self.transform(map_img)
            street_img = self.transform(street_img)

        return map_img, street_img, torch.tensor(float(item['label']), dtype=torch.float32)

# --- 2. The Model (Concatenation Architecture) ---
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
        
        # Classifier Head
        # Input size is doubled because we concatenate map + street
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, map_img, street_img):
        map_features = self.cnn(map_img)
        street_features = self.cnn(street_img)
        
        # Concatenate features (Side-by-Side)
        # This lets the model decide how to compare them
        combined = torch.cat((map_features, street_features), dim=1)
        
        logits = self.classifier(combined)
        return logits.squeeze()

# --- 3. Training Loop ---
def train():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = MapStreetDataset(TRAIN_JSON, transform=transform)
    test_dataset = MapStreetDataset(TEST_JSON, transform=transform)

    # --- FIXED SAMPLER LOGIC ---
    print("Calculating weights for balanced sampling...")
    targets = train_dataset.labels # These are now ints
    class_count = np.bincount(targets)
    print(f"   Class counts: 0 (Neg)={class_count[0]}, 1 (Pos)={class_count[1]}")
    
    class_weights = 1. / class_count
    samples_weights = [class_weights[t] for t in targets]
    samples_weights = torch.tensor(samples_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ObservationProbabilityModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Track training accuracy to ensure it's actually learning
        train_correct = 0
        train_total = 0
        
        for map_imgs, street_imgs, labels in train_loader:
            map_imgs, street_imgs, labels = map_imgs.to(DEVICE), street_imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(map_imgs, street_imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calc training stats
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        evaluate(model, test_loader)

    torch.save(model.state_dict(), 'observation_model.pth')
    print("Model saved.")

def evaluate(model, loader):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for map_imgs, street_imgs, labels in loader:
            map_imgs, street_imgs, labels = map_imgs.to(DEVICE), street_imgs.to(DEVICE), labels.to(DEVICE)
            
            logits = model(map_imgs, street_imgs)
            probs = torch.sigmoid(logits)
            
            # Threshold
            predicted = (probs > 0.5).float()
            
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"   [TEST] Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
    print(f"   [TEST] TP={int(tp)} | TN={int(tn)} | FP={int(fp)} | FN={int(fn)}")
    print("-" * 50)

if __name__ == "__main__":
    train()