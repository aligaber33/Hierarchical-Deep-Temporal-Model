import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from Baseline_B1 import find_clip_label, get_train_test_split_indices

# =====================================================================
# CONFIGURATION
# =====================================================================
EPOCHS = 10
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

videos_root = Path(r"videos-splitted")
features_root = Path(r'features/image-level/resnet')

categories_dct = {
    'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}

# =====================================================================
# ROBUST TEMPORAL DATASET
# =====================================================================
class TemporalDataSet(Dataset):
    def __init__(self, target_video_names, root_dir, transform=None, target_frames=9):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_frames = target_frames
        self.samples = []
        self.labels = []
        
        # Build an explicit lookup of where each named video folder lives inside the groups
        video_folder_lookup = {}
        for group_dir in sorted(self.root_dir.iterdir()):
            if group_dir.is_dir() and not group_dir.name.startswith('.'):
                for video_dir in sorted(group_dir.iterdir()):
                    if video_dir.is_dir() and not video_dir.name.startswith('.'):
                        # Store by its folder name (e.g., '1', '37')
                        video_folder_lookup[video_dir.name] = video_dir

        # Match exactly against the split names assigned by your split function
        selected_vids = [video_folder_lookup[name] for name in target_video_names if name in video_folder_lookup]

        for vid_dir in selected_vids:
            for clip_dir in sorted(vid_dir.iterdir()):
                if not clip_dir.is_dir() or clip_dir.name.startswith('.'):
                    continue
                    
                label_str = find_clip_label(clip_dir)
                if not label_str:
                    continue
                    
                label_str = label_str.replace("-", "_")
                label = categories_dct.get(label_str, -1)
                
                if label == -1:
                    continue 
                        
                self.samples.append(clip_dir)
                self.labels.append(label)
                        
        print(f"Loaded {len(self.samples)} clips successfully || Loaded {len(self.labels)} labels successfully")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        clip_path = self.samples[idx]
        label = self.labels[idx]
        frame_files = sorted([f for f in clip_path.iterdir() if f.suffix.lower() == ".jpg"])
        frame_files = frame_files[:self.target_frames]

        clip_frames = []
        for img_path in frame_files:
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                clip_frames.append(img)
            except Exception as e:
                pass

        if clip_frames:
            pad_frame = torch.zeros_like(clip_frames[0])
        else:
            pad_frame = torch.zeros(3, 224, 224)
 
        while len(clip_frames) < self.target_frames:
            clip_frames.append(pad_frame)

        clip_tensor = torch.stack(clip_frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return clip_tensor, label

# =====================================================================
# END-TO-END MODEL PIPELINE
# =====================================================================
class TemporalClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes=8):
        super(TemporalClassifier, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        self.feat_dim = resnet.fc.in_features
        
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size * frames, channels, height, width)

        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, frames, self.feat_dim)

        lstm_out, (hn, cs) = self.lstm(features)
        return self.classifier(hn[-1])

# =====================================================================
# MAIN RUNNER
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Processing Device: {device}")

    # Check features path folder existence
    if not features_root.exists():
        print(f"📁 Creating default directory pattern layout for: {features_root}")
        features_root.mkdir(parents=True, exist_ok=True)

    all_vids_on_disk = sorted([d for d in os.listdir(str(features_root)) if (features_root / d).is_dir()])
    total_video_count = len(all_vids_on_disk)
    print(f"Total Video Directories Detected on Disk: {total_video_count}")

    # --- 🔍 EXPLORATORY STRUCTURAL CHECK ---
    if total_video_count == 0:
        print("\n⚠️ WARNING: Your features_root folder is empty! Trying to read raw video layout names directly...")
        # Fallback tracking loop layout to read straight from videos-splitted if features folder is missing items
        for g in sorted(videos_root.iterdir()):
            if g.is_dir() and not g.name.startswith('.'):
                all_vids_on_disk.extend([v.name for v in g.iterdir() if v.is_dir()])
        all_vids_on_disk = sorted(list(set(all_vids_on_disk)))
        total_video_count = len(all_vids_on_disk)
        print(f"Adjusted Video Names List Count from raw disk space: {total_video_count}")

    torch.manual_seed(42)
    
    train_vid_idxs, test_vid_idxs = get_train_test_split_indices(total_video_count)
    
    # Map the splitting index integers directly to the specific folder name string items
    train_vid_names = [all_vids_on_disk[i] for i in train_vid_idxs if i < len(all_vids_on_disk)]
    test_vid_names = [all_vids_on_disk[i] for i in test_vid_idxs if i < len(all_vids_on_disk)]

    print(f"Processing matches using explicit Video Folder Targets: {len(train_vid_names)} train items, {len(test_vid_names)} test items.")

    train_dataset = TemporalDataSet(train_vid_names, root_dir=videos_root, transform=train_transforms, target_frames=9)
    test_dataset = TemporalDataSet(test_vid_names, root_dir=videos_root, transform=val_transforms, target_frames=9)

    if len(train_dataset) == 0:
        print("❌ Pipeline failed. Verify your 'videos-splitted' path contains structural video files/folders.")
        exit()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    
    model = TemporalClassifier(hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    print("\n🏋️ Starting Baseline End-to-End Image Model Training...")
    for epoch in range(EPOCHS):
        running_epoch_loss = 0.0
        correct = 0
        train_total = 0

        model.train()
        for feat, label in train_loader:
            feat, label = feat.to(device)[:100], label.to(device)[:100]

            optimizer.zero_grad()
            output = model(feat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_epoch_loss += loss.item() * feat.size(0)

            _, predicted = output.max(1)
            train_total += label.size(0)
            correct += predicted.eq(label).sum().item()

        epoch_loss = running_epoch_loss / len(train_dataset)
        epoch_accuracy = (correct / train_total) * 100
        print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_accuracy:.2f}%")


    