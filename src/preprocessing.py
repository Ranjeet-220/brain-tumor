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
        glioma/
        meningioma/
        notumor/
        pituitary/
    Maps to Binary: 0 (No Tumor), 1 (Tumor)
    """
    # Map directory names to binary labels
    # notumor -> 0
    # others -> 1
    categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
    
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_dir, category)
        
        # Determine label
        if category == 'notumor':
            binary_label = 0
        else:
            binary_label = 1
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist. Skipping...")
            continue
            
        print(f"Loading {category}...")
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    continue
                new_array = cv2.resize(img_array, img_size)
                data.append(new_array)
                labels.append(binary_label)
            except Exception as e:
                # print(f"Error loading {img_name}: {e}")
                pass
                
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
