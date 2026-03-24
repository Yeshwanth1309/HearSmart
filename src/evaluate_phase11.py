"""
Comprehensive evaluation of Phase 11 ESC-50 models.
Compares ESC-50-trained CNN and YAMNet on the new test set.
"""

import os
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.esc50_loader import ESC50Dataset, load_esc50_metadata, US8K_CLASSES
from src.train_esc50_cnn import ESC50_CNN

def evaluate_new_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data
    train_df, test_df = load_esc50_metadata("data/ESC50/meta/esc50.csv")
    audio_dir = "data/ESC50/audio/"
    
    # ── 1. CNN EVAL ─────────────────────────────────────────────────────────
    cnn = ESC50_CNN(num_classes=10).to(device)
    cnn.load_state_dict(torch.load("models/cnn_esc50.pth", map_location=device))
    cnn.eval()
    
    test_ds = ESC50Dataset(test_df, audio_dir)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    cnn_preds, cnn_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.argmax(cnn(x), dim=1).item()
            cnn_preds.append(p)
            cnn_labels.append(y.item())
            
    # ── 2. YAMNet EVAL ───────────────────────────────────────────────────────
    y_head = tf.keras.models.load_model("models/yamnet_esc50.h5")
    y_base = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # Load US8K training stats for embedding normalization (if used during fit)
    try:
        E_train_us8k = np.load("features/yamnet_embeddings/train.npy")
        emb_mean = E_train_us8k.mean(axis=0)
        emb_std = E_train_us8k.std(axis=0) + 1e-8
    except:
        emb_mean, emb_std = 0, 1
        
    y_preds, y_labels = [], []
    import librosa
    for idx, row in test_df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        audio, _ = librosa.load(file_path, sr=16000, duration=3.0)
        if len(audio) < 48000:
            audio = np.pad(audio, (0, 48000 - len(audio)), mode='constant')
        else:
            audio = audio[:48000]
            
        _, emb, _ = y_base(audio)
        emb_pool = tf.reduce_mean(emb, axis=0).numpy()
        emb_norm = (emb_pool - emb_mean) / emb_std
        p = y_head.predict(emb_norm.reshape(1, -1), verbose=0).argmax()
        
        y_preds.append(p)
        from src.data.esc50_loader import ESC50_TO_US8K_MAP
        target_name = ESC50_TO_US8K_MAP[row['category']]
        y_labels.append(US8K_CLASSES.index(target_name))
        
    # ── 3. Combine & Compare ────────────────────────────────────────────────
    results = {
        "CNN (ESC50)": {
            "Accuracy": accuracy_score(cnn_labels, cnn_preds),
            "Macro F1": f1_score(cnn_labels, cnn_preds, average='macro')
        },
        "YAMNet (ESC50)": {
            "Accuracy": accuracy_score(y_labels, y_preds),
            "Macro F1": f1_score(y_labels, y_preds, average='macro')
        }
    }
    
    print("\nPhase 11: ESC-50 Evaluation Results")
    print("="*60)
    for model, metrics in results.items():
        print(f"{model:20s} | Acc: {metrics['Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
    
    # Export Detailed Report
    with open("results/phase11_report.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    
    # Individual reports (active categories only)
    active_indices = sorted(list(set(y_labels)))
    active_names = [US8K_CLASSES[i] for i in active_indices]
    
    print("\nCNN Report (Active Groups):")
    print(classification_report(cnn_labels, cnn_preds, labels=active_indices, target_names=active_names))
    
    print("\nYAMNet Report (Active Groups):")
    print(classification_report(y_labels, y_preds, labels=active_indices, target_names=active_names))

if __name__ == "__main__":
    evaluate_new_models()
