import os
import glob
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ==========================================
# 1. Configuration Paths & Parameters
# ==========================================
person_feats_path = r"features/person-level"
annotation_path = r"volleyball_tracking_annotation"

train_percent = 0.8
test_percent = 0.2

label_to_idx = {
    "waiting": 0, "setting": 1, "digging": 2, "falling": 3,
    "spiking": 4, "blocking": 5, "jumping": 6, "moving": 7,
    "standing": 8
}

# ==========================================
# 2. PyTorch Dataset Implementation
# ==========================================
class FC7PersonFeat(Dataset):
    def __init__(self, annot_path, base_feat_path, label_to_idx):
        self.all_feat = []
        self.all_labels = []
        
        # FIX 1: Search recursively (**) to find all text files in subfolders
        txt_files = glob.glob(os.path.join(annot_path, "**", "*.txt"), recursive=True)
        print(f"Found {len(txt_files)} total annotation text files.")
        
        print("Scanning features directory to map .npy files...")
        npy_files = glob.glob(os.path.join(base_feat_path, "**", "*.npy"), recursive=True)
        npy_lookup = {os.path.basename(f): f for f in npy_files}
        print(f"Found {len(npy_lookup)} unique feature (.npy) files.")

        match_count_global = 0
        
        for text_file in txt_files:
            base_name = os.path.basename(text_file).replace(".txt", ".npy")
            
            if base_name not in npy_lookup:
                continue 

            current_feat_path = npy_lookup[base_name]
            label_dct = collections.defaultdict(dict)

            # Parse text file line by line
            with open(text_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    
                    
                    parts = line.split()
                    if len(parts) < 10:
                        continue

                    person_id = int(parts[0])
                    frame_id = int(parts[5])
                    action_str = parts[9].lower().strip() # Force lowercase and strip hidden spaces

                    label_dct[frame_id][person_id] = action_str

            # Load the 0-D array dictionary payload
            try:
                feat_dct = np.load(current_feat_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading {current_feat_path}: {e}")
                continue

            # Align features with extracted annotations row-by-row
            for frame_key, feats_mat in feat_dct.items():
                frame_id_int = int(frame_key)
                num_person = feats_mat.shape[0]

                for person_idx in range(num_person):
                    label = label_dct.get(frame_id_int, {}).get(person_idx, None)

                    if label in label_to_idx:
                        class_id = label_to_idx[label]
                        self.all_feat.append(feats_mat[person_idx])
                        self.all_labels.append(class_id)
                        match_count_global += 1

        print(f"Successfully processed and matched {match_count_global} total person tensors.")

    def __len__(self):
        return len(self.all_feat)

    def __getitem__(self, idx):
        x = torch.tensor(self.all_feat[idx], dtype=torch.float32)
        y = torch.tensor(self.all_labels[idx], dtype=torch.long)
        return x, y

# ==========================================
# 3. Model Architecture (MLP Classifier Head)
# ==========================================
class PersonClassifier(nn.Module):
    def __init__(self, input_size, num_classes=9):
        super(PersonClassifier, self).__init__()
    
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. Main Execution Phase
# ==========================================
if __name__ == "__main__":
    Epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using processing device: {device}")

    torch.manual_seed(42)

    dataset = FC7PersonFeat(annotation_path, person_feats_path, label_to_idx)
    
    if len(dataset) == 0:
        print("\n❌ Setup of data failed. Zero records processed.")
        print("Possible causes:")
        print("1. Your feature folder structure does not contain '.npy' files.")
        print("2. The action names in the text files don't match your 'label_to_idx' keys.")
        exit()

    # Random split for Training and Validation subsets
    train_size = int(train_percent * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    sample_x, _ = dataset[0]
    input_dimension = sample_x.shape[0]
    print(f"Dimension of input features is {input_dimension}")

    model = PersonClassifier(input_size=input_dimension, num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("\nStarting Baseline B2 Model Fine-Tuning...")
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_labels).sum().item()
            total_train += batch_labels.size(0)
            
        epoch_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1:02d}/{Epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

        
    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\n🔬 Evaluating Model Performance Against Unseen Test Set...")
    with torch.no_grad():
        for feat, label in val_loader:
            feat, label = feat.to(device), label.to(device)
            
            output = model(feat)
            _, predicted = output.max(1)
            
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
            
    final_test_accuracy = (test_correct / test_total) * 100
    print(f"📊 Final Accuracy on the Test Set: {final_test_accuracy:.2f}%")

    torch.save(model.state_dict(), "Pytorch_baseline_b2.pth")
    print("\n🎉 B1 Scene Model saved successfully as 'Pytorch_baseline_b2.pth'!")
    