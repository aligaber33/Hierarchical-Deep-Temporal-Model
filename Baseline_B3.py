import os
import glob
import collections
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, dataloader


person_feats_path = r"features/person-level/resnet"
annotation_root_dir = "videos-splitted"

train_percent = 0.8
test_percent = 0.2


categories_dct = {
    'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}


def get_all_labels(annot_path):
    labels = {}
    search_path = os.path.join(annot_path, "*", "*", "annotations.txt")
    annot_files = glob.glob(search_path)

    print(f"Total annotation files found: {len(annot_files)}\n")

    for path in annot_files:
        parts = path.split(os.sep)
        clip_id = parts[-2]
        raw_video_dir = parts[-3]
        
        video_id = raw_video_dir.replace("videos_g", "").replace("vidoes_g", "")
        
        
        
        with open(path, "r") as f:
            for line in f:
                line_parts = line.strip().split()
                key = line_parts[0].replace(".jpg", "")
                if len(line_parts) >= 2:
                    group_activity = line_parts[1]
                    label = categories_dct.get(group_activity, -1)
                    labels[key] = label
                    
    print(f"Number of verified labels processed: {len(labels)}")
    return labels


def get_all_features(feats_path):
    feats_vectors = {}

    search_path = os.path.join(feats_path, "*", "*.npy")
    feat_files = glob.glob(search_path)

    print(f"Total feature files are {len(feat_files)}")

    for path in feat_files:
        parts = path.split(os.sep)
        video_id = parts[-2]
        clip_id = Path(path).stem
        
        key = clip_id
        feats_vectors[key] = path
        
    return feats_vectors


def get_feat_label_pair(feat_dct, label_dct):
    dict3 = {}
    for key in feat_dct:
        if key in label_dct:
            dict3[key] = [feat_dct[key], label_dct[key]]
    return dict3


class PooledPeopleDataset(Dataset):
    def __init__(self, data_dict):
        self.samples = []
        self.labels = []

        for val in data_dict.values():
            feat_path = val[0]
            label = val[1]
            self.samples.append(feat_path)
            self.labels.append(label)
            
        print(f"Found {len(self.samples)} verified matching pair examples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        loaded_data = np.load(self.samples[idx], allow_pickle=True)
        
        if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == np.object_:
            try:
                data_item = loaded_data.item()
            except ValueError:
                data_item = loaded_data
        else:
            data_item = loaded_data

        if isinstance(data_item, dict):
            feature_list = list(data_item.values())
            numpy_array = np.array(feature_list, dtype=np.float32)
        else:
            if hasattr(data_item, 'dtype') and data_item.dtype == np.object_:
                try:
                    numpy_array = data_item.astype(np.float32)
                except ValueError:
                    numpy_array = np.array(list(data_item), dtype=np.float32)
            else:
                numpy_array = np.array(data_item, dtype=np.float32)
        
        x = torch.tensor(numpy_array, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        target_people = 12
        current_frames, current_people, feature_dim = x.shape
        
        if current_people < target_people:
            padding_size = target_people - current_people
            
            padding_tensor = torch.zeros((current_frames, padding_size, feature_dim), dtype=torch.float32)
            
            x = torch.cat([x, padding_tensor], dim=1)
            
        elif current_people > target_people:
            x = x[:, :target_people, :]

        return x, y


class GroupActivityRecognition(nn.Module):
    def __init__(self, num_classes=8, num_players=12, feature_dim=2048):
        super().__init__()
        
        # Calculate the massive new input size for the fully connected layer
        # 12 players * 2048 features = 24,576
        self.concat_input_size = num_players * feature_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.concat_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        
        # 1. Average pool across the 9 frames (time dimension)
        # Shape becomes: [Batch, 12, 2048]
        x = torch.mean(x, dim=1)
        
        # Shape becomes: [Batch, 24576]
        x = torch.flatten(x, start_dim=1)
        
        return self.fc(x)


if __name__ == "__main__":
    labels = get_all_labels(annotation_root_dir)
    feats = get_all_features(person_feats_path)
    pair = get_feat_label_pair(feats, labels)
    
    print("\n--- 🔍 KEY INSPECTION DIAGNOSTIC ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PooledPeopleDataset(pair)
    if len(dataset) == 0:
        print("\n❌ Setup of data failed. Zero records processed.")
        print("Possible causes:")
        print("1. Your feature folder structure does not contain '.npy' files.")
        print("2. The action names in the text files don't match your 'label_to_idx' keys.")
        exit()


    sample, label = dataset[0]
    print(f"shape of input is {sample.shape}")
    

    Epochs = 15
    model = GroupActivityRecognition().to(device)
    train_size = int(train_percent * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 64)
    test_loader = DataLoader(val_dataset, shuffle = False, batch_size = 64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    

    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            outputs = model(feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1:02d}/{Epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")



          
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

    #torch.save(model.state_dict(), "Pytorch_baseline_b3.pth")
    #print("\n🎉 B1 Scene Model saved successfully as 'Pytorch_baseline_b3.pth'!")
    