import os
import tensorflow as tf
from preprocessing import load_data, preprocess_data
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def train():
    # Load and preprocess data
    raw_path = "data/real/Training"
    print("Loading data...")
    X, y = load_data(raw_path)
    
    if len(X) == 0:
        print("No data found. Please place images in data/raw/Tumor and data/raw/No Tumor")
        return

    print(f"Data loaded: {len(X)} samples")
    X, y = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model()
    
    # Callbacks
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "brain_tumor_model.keras")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10, # Start with smaller epochs for testing
        batch_size=8, # Small batch size for dummy data
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    print("Training complete.")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    model.save(os.path.join(checkpoint_dir, "final_model.h5"))

if __name__ == "__main__":
    train()
