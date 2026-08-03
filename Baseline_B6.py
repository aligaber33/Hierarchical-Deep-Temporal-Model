import os
import glob
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
from pathlib import Path
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Baseline_B3 import get_all_labels

vids_root = Path(r"/teamspace/studios/this_studio/videos-splitted")
annot_root = Path(r"/teamspace/studios/this_studio/volleyball_tracking_annotation")

train_percent = 0.8
test_percent = 0.2

categories_dct = {
    'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}

def getpalyer_seq(image_directory, annotation_file_path, frame_sequence, max_persons=12, crop_size=(224, 224)):

    frame_annotations = {f_id: {} for f_id in frame_sequence} 
    
    with open(annotation_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue
                
            parts = line.split()
            if len(parts) < 6:
                continue
                
            try:
                slot_idx = int(parts[0]) 
                
                frame_id = int(parts[5]) 
                
                if frame_id in frame_annotations:
                    if 0 <= slot_idx < max_persons:
                        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        frame_annotations[frame_id][slot_idx] = (x1, y1, x2, y2)
            except ValueError:
                continue

    all_frames_crops = []
    all_frames_masks = []
    
    # 2. Extract crops keeping fixed slots aligned across time
    for frame_id in frame_sequence:
        image_path = os.path.join(image_directory, f"{frame_id}.jpg")
        
        # If image is missing, pad ALL slots for this time step
        if not os.path.exists(image_path):
            all_frames_crops.append(torch.zeros(max_persons, 3, crop_size[0], crop_size[1]))
            all_frames_masks.append(torch.zeros(max_persons))
            continue
            
        full_image = Image.open(image_path).convert('RGB')
        img_w, img_h = full_image.size
        
        frame_crops_list = []
        frame_mask = torch.zeros(max_persons, dtype=torch.float32)
        
        # Iterating strictly by SLOT ID (0 to 11) to guarantee temporal alignment
        for current_slot in range(max_persons):
            if current_slot in frame_annotations[frame_id]:
                # The player for this specific ID is present in this frame
                x1, y1, x2, y2 = frame_annotations[frame_id][current_slot]
                x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
                y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))
                
                if x2 > x1 and y2 > y1:
                    player_crop = full_image.crop((x1, y1, x2, y2))
                    crop_tensor = TF.to_tensor(player_crop)
                    crop_tensor = TF.resize(crop_tensor, crop_size, antialias=True)
                    crop_tensor = TF.normalize(crop_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    
                    frame_crops_list.append(crop_tensor)
                    frame_mask[current_slot] = 1.0
                    continue 
            
            empty_crop = torch.zeros(3, crop_size[0], crop_size[1])
            frame_crops_list.append(empty_crop)

        # Stack the 12 slots for this specific frame
        frame_crops_tensor = torch.stack(frame_crops_list, dim=0)
        
        all_frames_crops.append(frame_crops_tensor)
        all_frames_masks.append(frame_mask)
        
    sequence_tensor = torch.stack(all_frames_crops, dim=0)  # Shape: (T, N, C, H, W)
    sequence_mask = torch.stack(all_frames_masks, dim=0)    # Shape: (T, N)
    
    return sequence_tensor, sequence_mask


class TwoStageModelB6(nn.Module):
    def __init__(self, num_classes=8, projection_dim=512, hidden_dim=512):
        super(TwoStageModelB6, self).__init__()
        
        # Load backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features
        
        # Projections & Activations
        self.feat_proj = nn.Linear(self.cnn_out_dim, projection_dim)
        self.relu = nn.ReLU()  # Renamed from self.residual to accurately reflect behavior
        self.dropout = nn.Dropout(p=0.4)
        
        # Group Temporal Model (LSTM)
        self.player_lstm = nn.LSTM(projection_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Classifier Head
        self.group_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask =None):
        B, T, N, C, W, H = x.shape
        
        # 1. Flatten into (B*T*N, C, W, H) for the ResNet backbone
        x_reshaped = x.view(B * T * N, C, W, H)
        fc7_feat = self.backbone(x_reshaped)   
        fc7_feat = torch.flatten(fc7_feat, 1)  # Shape: (B*T*N, 2048)
        
        # 2. Project
        projection = self.feat_proj(fc7_feat)   # Shape: (B*T*N, projection_dim)
        projection = self.relu(projection)
        projection = self.dropout(projection)

        # 3. Reshape: (B, T, N, proj_dim) -> (B, N, T, proj_dim)
        # We need N to be the second dimension so we can group all T frames per player
        projection = projection.view(B, T, N, -1)
        projection = projection.permute(0, 2, 1, 3) 


        # 4. LSTM Input: (B*N, T, proj_dim)
        lstm_in = projection.reshape(B * N, T, -1) 
        lstm_out, _ = self.player_lstm(lstm_in) # Shape: (B*N, T, hidden_dim)

        # 5. Extract final time step: (B*N, hidden_dim)
        player_feats = lstm_out[:, -1, :]
        
        # 6. Reshape back to (B, N, hidden_dim) to pool over players
        player_feats = player_feats.view(B, N, -1) 
        

        if mask is not None:
            # mask is expected to have temporal/spatial info. 
            # If mask shape is (B, T, N), aggregate over time to see if player appeared at all:
            if mask.ndim == 3:  # (B, T, N)
                player_presence = mask.max(dim=1)[0]  # Shape: (B, N)
            else:
                player_presence = mask  # Shape: (B, N)
                
            player_feats = player_feats * player_presence.unsqueeze(-1)
        
        # 7. Max pool over players (N) -> (B, hidden_dim)
        group_repr, _ = torch.max(player_feats, dim=1) 

        return self.group_classifier(group_repr)

class B6Dataset(Dataset):
    def __init__(self, labels_dct, vids_root, annot_root, max_persons=12, crop_size=(224, 224)):
        """
        Only stores paths and lightweight metadata to minimize RAM usage.
        """
        self.samples_metadata = []
        self.labels = []
        self.labels_dct = labels_dct
        
        self.max_persons = max_persons
        self.crop_size = crop_size

        for video in vids_root.iterdir():
            if not video.is_dir():
                continue
            for clip in video.iterdir():
                if not clip.is_dir():
                    continue

                for frame_name in clip.iterdir():
                    if not frame_name.is_dir():
                        continue
                    
                    key = frame_name.stem
                    if key not in labels_dct:
                        continue 
                        
                    label = labels_dct[key]
                    self.labels.append(label)
                    
                    seq_range = list(range(int(key) - 10, int(key) + 10)) 
                    
                    image_dir_path = vids_root / video.name / clip.name / key
                    annotation_file_path = annot_root / clip.name / key / f"{key}.txt"
                    
                    self.samples_metadata.append({
                        'image_dir': str(image_dir_path),
                        'annot_file': str(annotation_file_path),
                        'seq_range': seq_range
                    })
                    
        print(f"Dataset initialized with {len(self.samples_metadata)} metadata slots.")

    def __len__(self):
        return len(self.samples_metadata)
    
    def __getitem__(self, index):
        meta = self.samples_metadata[index]
        label = self.labels[index]


        seq_tensor, mask_tensor = getpalyer_seq(
            image_directory=meta['image_dir'],
            annotation_file_path=meta['annot_file'],
            frame_sequence=meta['seq_range'],
            max_persons=self.max_persons,
            crop_size=self.crop_size
        )

        label_tensor = torch.tensor(label, dtype=torch.long)

        return seq_tensor, mask_tensor, label_tensor


if __name__ == "__main__":
    Epochs = 20
    batch_size = 4

    labels = get_all_labels(vids_root, categories_dct)

    print("\n---  KEY INSPECTION DIAGNOSTIC ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = B6Dataset(labels,vids_root, annot_root)
    if len(dataset) == 0:
        print("\n Setup of data failed. Zero records processed.")
        exit()
    
    sample, mask,label = dataset[0]
    print(f"shape of input is {sample.shape}")


    model = TwoStageModelB6().to(device)
    train_size = int(train_percent * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers=4)
    test_loader = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("Start Trainig end to end model")
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        checkpoint_path =r"B6_checkpoint.pth"

        for feats, masks, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            outputs = model(feats, masks)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        scheduler.step()
        epoch_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1:02d}/{Epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")


    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\n Evaluating Model Performance Against Unseen Test Set...")
    with torch.no_grad():
        for feat, masks,label in test_loader:
            feat, label = feat.to(device), label.to(device)
            
            output = model(feat, masks)
            _, predicted = output.max(1)
            
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
            
    final_test_accuracy = (test_correct / test_total) * 100
    print(f" Final Accuracy on the Test Set: {final_test_accuracy:.2f}%")
