"""
ESC-50 CNN Architecture and Training.
Features a Conv2D architecture as requested in Phase 11.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data.esc50_loader import ESC50Dataset, load_esc50_metadata, US8K_CLASSES
from src.utils import set_seed, setup_logging

class ESC50_CNN(nn.Module):
    """CNN Architecture as specified by user."""
    
    def __init__(self, num_classes=10):
        super(ESC50_CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

def train_cnn():
    set_seed(42)
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Load data
    train_df, test_df = load_esc50_metadata("data/ESC50/meta/esc50.csv")
    audio_dir = "data/ESC50/audio/"
    
    train_ds = ESC50Dataset(train_df, audio_dir)
    test_ds = ESC50Dataset(test_df, audio_dir)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 2. Setup model
    model = ESC50_CNN(num_classes=10).to(device)
    # Using weighted loss isn't strictly necessary as data is balanced for the 5 classes,
    # but US8K labels range 0-9 whereas we only have 5 labels. 
    # Loss for the 5 active classes.
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 50
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/cnn_esc50.pth")
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")

    print(f"Training complete. Best Accuracy: {best_acc:.4f}")
    
    # Final Eval
    model.load_state_dict(torch.load("models/cnn_esc50.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Final Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Final Macro F1: {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Only show active classes in CM
    # dogbark=3, children_playing=2, drilling=4, engine_idling=5, siren=8
    active_indices = sorted([0, 22, 41, 44, 42]) # Wait, labels are US8K indices
    # Maps categories to US8K_CLASSES
    # siren(8), engine(5), chainsaw(4), clapping(2), dog(3)
    active_indices = [3, 2, 4, 5, 8] # Indices for: dog_bark, children_playing, drilling, engine_idling, siren
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[US8K_CLASSES[i] for i in sorted(active_indices)],
                yticklabels=[US8K_CLASSES[i] for i in sorted(active_indices)])
    plt.title("Confusion Matrix - ESC-50 CNN")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("results/cm_esc50_cnn.png")
    plt.close()

if __name__ == "__main__":
    train_cnn()
