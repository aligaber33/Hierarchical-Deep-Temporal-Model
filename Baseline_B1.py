import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import models, transforms
from extract_feat import find_video_dir_for_clip
from PIL import Image

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
total_vids = 55
videos_root = Path(r"videos-splitted")
features_root = Path(r'features/image-level/resnet')

train_percent = 0.8
test_percent = 0.2

categories_dct = {
    'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}
def get_train_test_split_indices(total_videos, train_percent=train_percent):
    """
    Returns two disjoint 1D torch.long tensors containing random unique indices 
    for training and testing sets.
    """

    shuffled_indices = torch.randperm(total_videos)
    
    num_train = int(total_videos * train_percent)
    
    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:]
    
    return train_indices, test_indices



class Image_Classifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=8):
        super(Image_Classifier, self).__init__()
        self.backbone =  models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_filters = self.backbone.fc.in_features
        

        #self.backbone.fc = nn.Linear(num_filters, num_classes)
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_filters, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)



def find_clip_label(clip_dir_path):
    clip_dir_path = Path(clip_dir_path)
    ant_path = clip_dir_path.parent / "annotations.txt"

    if not ant_path.exists():
        ant_path = clip_dir_path / "annotations.txt"
        if not ant_path.exists():
            return None

    clip_name = clip_dir_path.name

    with open(ant_path, 'r') as f:
        for line in f:
            items = line.strip().split(" ")
            if len(items) >= 2:
                file_base_name = items[0].replace('.jpg', '')
                if file_base_name == clip_name:   
                    return items[1]
    return None


class VolleyBallDataSet(Dataset):
    def __init__(self, allowed_indices, transform = None):
        """
        Args:
            allowed_indices (Tensor): A torch.long tensor containing the specific 
                                      video folder index positions allowed for this split.
        """
        self.samples = []
        self.transform = transform

        if not features_root.exists():
            print(f"🚨 Error: Features directory {features_root} does not exist.")
            return

        all_video_dirs = sorted([
            d for d in os.listdir(str(features_root)) 
            if (features_root / d).is_dir()
        ])

    
        allowed_indices_list = allowed_indices.tolist()

        for idx, video_dir in enumerate(all_video_dirs):
            if idx not in allowed_indices_list:
                continue

            video_feat_path = features_root / video_dir
            npy_feat_files = sorted(video_feat_path.glob("*.npy"))
            
            for npy_file in npy_feat_files:
                clip_dir = npy_file.stem  

                clip_images_path = find_video_dir_for_clip(video_dir, clip_dir)
                if clip_images_path is None:
                    continue

                label = find_clip_label(clip_images_path)
                if label is None or label not in categories_dct:
                    continue

                raw_frames = sorted(list(Path(clip_images_path).glob("*.jpg")))
                if not raw_frames:
                    continue

                mid_frame = raw_frames[len(raw_frames) //2]

                self.samples.append({
                    "img_path": mid_frame,
                    "label": categories_dct[label]
                })

        if not self.samples:
            print("\n🚨 Dataset Loading Failed: 0 samples gathered.")
        else:
            print(f"\n🚀 Success! Found and loaded {len(self.samples)} valid clips into B1 Dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img,torch.tensor(sample["label"], dtype=torch.long)


if __name__ == "__main__":
    EPOCHS = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Processing Device: {device}")

    all_vids_on_disk = sorted([d for d in os.listdir(str(features_root)) if (features_root / d).is_dir()])
    total_video_count = len(all_vids_on_disk)
    print(f"Total Video Directories Detected on Disk: {total_video_count}")

    torch.manual_seed(42)
    
    train_vid_idxs, test_vid_idxs = get_train_test_split_indices(total_video_count)
    print("Training indecies are [", end = " ")
    for i in train_vid_idxs:
        print(i, end=" ")
    print("]")

    print("Testing indecies are [", end = " ")
    for i in test_vid_idxs:
        print(i, end=" ")
    print("]")
    train_dataset = VolleyBallDataSet(train_vid_idxs, train_transforms)
    test_dataset = VolleyBallDataSet(test_vid_idxs, val_transforms)


    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Set up of Datasets failed. Ensure sample allocations are valid.")
        exit()


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    test_loader = DataLoader(test_dataset, shuffle = False, batch_size= 32)
    sample_mat, _ = train_dataset[0]

    #feat_dim = sample_mat.shape[-1]
    #print(f"Features structural dimension is strictly: {feat_dim}")
    
    model = Image_Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("\n🏋️ Starting Baseline B1 Image Model Training...")
    for epoch in range(EPOCHS):
        running_epoch_loss = 0.0
        correct = 0
        train_total = 0

        model.train()
        for feat, label in train_loader:
            feat, label = feat.to(device), label.to(device)

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

    torch.save(model.state_dict(), "Pytorch_baseline_b1.pth")
    print("\n🎉 B1 Scene Model saved successfully as 'Pytorch_baseline_b1.pth'!")