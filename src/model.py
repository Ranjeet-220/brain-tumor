import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flattening
        Flatten(),
        
        # Fully Connected Layer
        Dense(128, activation='relu'),
        Dropout(0.5), # Add dropout to prevent overfitting
        
        # Output Layer (Binary Classification: Tumor vs No Tumor)
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
