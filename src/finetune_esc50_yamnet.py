"""
YAMNet fine-tuning for ESC-50 subset.
Re-uses the YAMNet embedding pipeline and trains a new head or fine-tunes the current one.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, f1_score
from src.data.esc50_loader import load_esc50_metadata, ESC50_TO_US8K_MAP, US8K_CLASSES
import librosa

def extract_yamnet_embeddings(df, audio_dir, model_base, target_sr=16000):
    """YAMNet specifically expects 16kHz."""
    embeddings = []
    labels = []
    
    for idx, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        try:
            audio, _ = librosa.load(file_path, sr=target_sr, duration=5.0) # ESC-50 is 5s
            # Pad/Trim to 3.0s as per user's request for preprocessing consistency
            target_samples = int(target_sr * 3.0)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
            
            # Extract
            _, emb, _ = model_base(audio)
            # Pool embeddings over time
            emb_mean = tf.reduce_mean(emb, axis=0).numpy()
            embeddings.append(emb_mean)
            
            # Label
            us8k_class = ESC50_TO_US8K_MAP[row['category']]
            label = US8K_CLASSES.index(us8k_class)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return np.array(embeddings), np.array(labels)

def train_yamnet_esc50():
    # 1. Load base and existing head
    yamnet_base = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # Load ESC-50 metadata
    train_df, test_df = load_esc50_metadata("data/ESC50/meta/esc50.csv")
    audio_dir = "data/ESC50/audio/"
    
    # 2. Extract Embeddings (caching in RAM as subset is small)
    print("Extracting training embeddings...")
    X_train, y_train = extract_yamnet_embeddings(train_df, audio_dir, yamnet_base)
    print("Extracting testing embeddings...")
    X_test, y_test = extract_yamnet_embeddings(test_df, audio_dir, yamnet_base)
    
    # Normalize with UrbanSound training stats (for consistency)
    # If these don't exist, use ESC-50 stats
    try:
        E_train_us8k = np.load("features/yamnet_embeddings/train.npy")
        emb_mean = E_train_us8k.mean(axis=0)
        emb_std = E_train_us8k.std(axis=0) + 1e-8
    except:
        emb_mean = X_train.mean(axis=0)
        emb_std = X_train.std(axis=0) + 1e-8
        
    X_train_norm = (X_train - emb_mean) / emb_std
    X_test_norm = (X_test - emb_mean) / emb_std
    
    # 3. Create/Fine-tune head
    # We will build a new head for ESC-50 specific robustness, 
    # but starting from the Architecture used in US8K.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(US8K_CLASSES), activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # 4. Fit
    history = model.fit(X_train_norm, y_train, 
                        epochs=50, 
                        batch_size=8,
                        validation_data=(X_test_norm, y_test),
                        verbose=1)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_norm).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"YAMNet ESC-50 Test Accuracy: {acc:.4f}")
    print(f"YAMNet ESC-50 Macro F1: {f1:.4f}")
    
    # 6. Save
    model.save("models/yamnet_esc50.h5")
    print("Saved models/yamnet_esc50.h5")

if __name__ == "__main__":
    train_yamnet_esc50()
