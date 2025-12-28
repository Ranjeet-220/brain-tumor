import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(224, 224)):
    """
    Loads images from the data directory.
    Assumes structure:
    data_dir/
        Tumor/
        No Tumor/
    """
    categories = ['No Tumor', 'Tumor'] # 0: No Tumor, 1: Tumor
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist.")
            continue
            
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    continue
                new_array = cv2.resize(img_array, img_size)
                data.append(new_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                
    return np.array(data), np.array(labels)

def preprocess_data(data, labels):
    """
    Normalizes data and converts labels to categorical.
    """
    # Normalize pixel values to [0, 1]
    data = data / 255.0
    
    # Reshape if necessary (already handled by resize, but good to check)
    # Convert labels to numpy array if not already
    labels = np.array(labels)
    
    return data, labels

if __name__ == "__main__":
    # Test the loading
    raw_path = "data/raw"
    X, y = load_data(raw_path)
    print(f"Loaded {len(X)} images.")
    if len(X) > 0:
        X, y = preprocess_data(X, y)
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
