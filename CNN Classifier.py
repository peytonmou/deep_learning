from load_oxford_flowers102 import load_oxford_flowers102
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class CNN(nn.Module):
    def __init__(self, in_channels=3, fine=True):
        super().__init__()
        n_classes = 102 if fine else 10
        
        # Feature extraction with depthwise separable convolutions
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),    # 3*96*96 -> 32*96*96
            nn.BatchNorm2d(32),                        
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),  # Depthwise conv layer
            nn.Conv2d(32, 64, 1),                        # Pointwise conv-layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 2x2 maxpooling: 64*96*96 -> 64*48*48
            nn.Dropout(0.2),                             # dropout=0.2 to lower overfitting degree
            
            nn.Conv2d(64, 64, 3, padding=1, groups=64),     
            nn.Conv2d(64, 128, 1),                          # 64*48*48 -> 128*48*48
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128),  
            nn.Conv2d(128, 256, 1),                         # 128*48*48 -> 256*48*48 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 2x2 maxpooling: 256*48*48 -> 256*24*24 
            nn.Dropout(0.3),                                # dropout=0.3 to further lower overfitting degree
            
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.Conv2d(256, 512, 1),                         # 256*24*24 -> 512*24*24
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))                        # global pooling: 512*24*24 -> 512*1*1
        
        # Fully connected classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),                            # full-connected layer
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)) 
        
        # Custom weight initialization for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)          # batchnorm weights set to 1
                nn.init.constant_(m.bias, 0)            # batchnorm weights set to 0 
    
    def forward(self, x):
        x = self.features(x)             # [16, 512, 1, 1]
        x = x.view(x.size(0), -1)        # [16, 512]
        x = self.classifier(x)           # [16, n_classes] 
        return x

def load_dataset(fine=True):
    if fine:
        training_set, validation_set, test_set, class_names = load_oxford_flowers102(imsize=96, fine=True) 
    else:
        training_set, validation_set, test_set, class_names = load_oxford_flowers102(imsize=96, fine=False)
    
    train_classes = [training_set[i][1] for i in range(len(training_set))]
    class_counts = Counter(train_classes)
    n_classes = len(class_counts)
    class_weights = torch.tensor([1.0 / class_counts[c] for c in sorted(class_counts)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * n_classes
    
    # training dataset is adjusted based on weights of each class
    sample_weights = torch.tensor([class_weights[y] for y in train_classes], dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)
    train_data = torch.utils.data.DataLoader(training_set, batch_size=16, sampler=sampler, num_workers=2, drop_last=True)
    
    # validation and test dataset remain the original version 
    val_data = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    return train_data, val_data, test_data, class_names 
    
def train_test(model, epochs, lr=0.001, load_from_file=True, fine=True):
    model = model.to(device)
    
    # load the corresponding dataset and pre-trained weights 
    train_loader, val_loader, test_loader, class_names = load_dataset(fine=fine) 
    pre_filename =  'fine_CNN_100.pth' if fine else 'coarse_CNN_100.pth'
    filename = 'new_fine_CNN.pth' if fine else 'new_coarse_CNN.pth'

    if load_from_file:
        model.load_state_dict(torch.load(pre_filename)) 
        print(f'Loaded weights from {pre_filename}') 
       
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # dynamic learning rate, after 3 epochs without improvement, learning rate reduce 50% 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    best_acc = 0
    best_epoch = 0 
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0 
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()                  # clean previous gradients
            
            with autocast('cuda'):                 # mixed precision, automatic FP16 conversion 
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()          # gradient scaling
            scaler.step(optimizer)                 # optimizer step
            scaler.update()                        # adjust scale factor
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation
        val_acc, val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        # save best model 
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), filename)  
            print(f"Saved best weights at {epoch+1}")
           
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}%')
        
    # After training load the best model weights from the checkpoint
    print(f'Loading weights from {filename} (epoch {best_epoch + 1})')
    model.load_state_dict(torch.load(filename, map_location=device)) 

    # Test: confusion matrix and classification performance report to be created 
    model.eval() 
    all_preds, all_true = [], [] 
    total, test_correct = 0, 0 
    
    with torch.no_grad():                              # Disable gradient computation for testing
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs) 
            _, preds = torch.max(outputs, dim=1)
            test_correct += (preds == targets).sum().item()
            total += targets.size(0) 
           
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(targets.cpu().numpy())

    test_acc = 100 * test_correct / total 
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%") 
    print(classification_report(all_true, all_preds, target_names=class_names, digits=4))

    # plot confusion matrix of test results
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (Test Accuracy: {test_acc:.2f}%)") 
    plt.show()

def validate(model, val_loader, criterion, device):
    model.eval()                                               # stop training, validation status
    val_loss, correct, total = 0, 0, 0 
    
    with torch.no_grad():                                      # disable gradient computation 
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)            
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
     
    val_acc = 100 * correct / total
    return val_acc, val_loss / len(val_loader) 

"""
Fine-grained (102 classes): fine=True
Coarse-grained (10 classes): fine=False
* Please ensure the 'fine' setting is consistent in both the model and train_test() call.

Train from scratch: load_from_file=False
Continue training from saved weights: load_from_file=True
* For stable performance, please train 10 epochs or more, as early training stages may show volatility.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(fine=False).to(device) 
train_test(model, epochs=10, lr=0.001, load_from_file=True, fine=False)   