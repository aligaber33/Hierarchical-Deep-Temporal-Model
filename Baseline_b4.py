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
        
        all_vids = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        selected_vids = [all_vids[i] for i in target_video_names if i < len(all_vids)]

        for vids_group in selected_vids:
            for vid_dir in sorted(vids_group.iterdir()):
                if not vid_dir.is_dir():
                    continue
                for clip_dir in sorted(vid_dir.iterdir()): 
                    label_str = find_clip_label(clip_dir)
                    if not label_str:
                        continue
                    
                    label_str = label_str.replace("-", "_")
                    label = categories_dct.get(label_str, -1)
                
                # FIXED: Added explicit continue block to completely drop -1 out of bounds items
                    if label == -1:
                        print(f"⚠️ Warning: Label '{label_str}' found in {clip_dir.name} is missing from categories_dct. Skipping.")
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
                print(f"Error loading image {img_path}: {e}")

        # FIXED: Initialized padding tensor safely outside the image loop
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
        # FIXED: Corrected structural assignment name identifiers to match [Channels, Height, Width]
        batch_size, frames, channels, height, width = x.shape

        x = x.view(batch_size * frames, channels, height, width)

        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, frames, self.feat_dim)

        lstm_out, (hn, cs) = self.lstm(features)

        final_hidden = hn[-1]
        return self.classifier(final_hidden)

# =====================================================================
# MAIN RUNNER
# =====================================================================
if __name__ == "__main__":
    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Processing Device: {device}")

    all_vids_on_disk = sorted([d for d in os.listdir(str(features_root)) if (features_root / d).is_dir()])
    total_video_count = len(all_vids_on_disk)
    print(f"Total Video Directories Detected on Disk: {total_video_count}")

    torch.manual_seed(42)
    
    train_vid_idxs, test_vid_idxs = get_train_test_split_indices(total_video_count)
    print("Training indices are [", " ".join(map(str, train_vid_idxs)), "]")
    print("Testing indices are [", " ".join(map(str, test_vid_idxs)), "]")
    
    train_dataset = TemporalDataSet(train_vid_idxs, root_dir=videos_root, transform=train_transforms, target_frames=40)
    test_dataset = TemporalDataSet(test_vid_idxs, root_dir=videos_root, transform=val_transforms, target_frames=40)

    if len(train_dataset) == 0:
        print("Set up of Datasets failed. Ensure sample allocations are valid.")
        exit()

    train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=4
    )    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=4
        )
    
    model = TemporalClassifier(hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3) # Low LR to prevent fine-tuning explosion

    print("\n🏋️ Starting Baseline End-to-End Image Model Training...")
    for epoch in range(EPOCHS):
        running_epoch_loss = 0.0
        correct = 0
        train_total = 0

        model.train()
        for feat, label in train_loader:
            feat, label = feat.to(device), label.to(device)
            print("calculating gradients")
            optimizer.zero_grad()
            print("calculating output")
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


    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\n🔬 Evaluating Model Performance Against Unseen Test Set...")
    with torch.no_grad():
        for feat, label in test_loader:
            feat, label = feat.to(device), label.to(device)
            
            output = model(feat)
            _, predicted = output.max(1)
            
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
            
    final_test_accuracy = (test_correct / test_total) * 100
    print(f"📊 Final Accuracy on the Test Set: {final_test_accuracy:.2f}%")

    torch.save(model.state_dict(), "Pytorch_baseline_b4.pth")
    print("\n🎉 B1 Scene Model saved successfully as 'Pytorch_baseline_b4.pth'!")