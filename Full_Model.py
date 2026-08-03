
import torch
from pathlib import Path
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Baseline_B3 import get_all_labels
from Baseline_B6 import B6Dataset



vids_root = Path(r"/teamspace/studios/this_studio/videos-splitted")
annot_root = Path(r"/teamspace/studios/this_studio/volleyball_tracking_annotation")

train_percent = 0.8
test_percent = 0.2

categories_dct = {
    'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}

class TwoStageModelB7(nn.Module):
    def __init__(self, num_classes=8, projection_dim=512, hidden_dim=256):
        super(TwoStageModelB7, self).__init__()
        
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
        
        # Person Temporal Model (LSTM-1)
        self.player_lstm = nn.LSTM(projection_dim, hidden_dim, num_layers=1, batch_first=True)
        

        #Group Temporal Model(LSTM-2)
        self.group_lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)

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
        player_lstm_out, _ = self.player_lstm(lstm_in) # Shape: (B*N, T, hidden_dim)

        player_feats = player_lstm_out.view(B, N, T, -1) #Shape (B, N, T, hidden_dim)


        if mask is not None:
            mask = mask.permute(0, 2, 1)

            player_feats = player_feats * mask
        
        # 7. Max pool over players (N)
        group_repr, _ = torch.max(player_feats, dim=1) #Shape (B, T, hidden_dim)

        #Feed to Second LSTM
        group_lstm_out, _ = self.group_lstm(group_repr)

        #Final representation
        final_repr = group_lstm_out[:, -1, :] #Shape (B, hidden_dim)

        return self.group_classifier(final_repr)


if __name__ == "__main__":
    Epochs = 25
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


    model = TwoStageModelB7().to(device)
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
