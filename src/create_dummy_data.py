import os
import numpy as np
import cv2

def create_dummy_data(base_path, num_samples=20):
    categories = ['Tumor', 'No Tumor']
    
    for category in categories:
        path = os.path.join(base_path, category)
        os.makedirs(path, exist_ok=True)
        
        print(f"Generating {num_samples} samples for {category}...")
        for i in range(num_samples):
            # Create a black image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Add some noise/shapes to distinguish
            if category == 'Tumor':
                # Draw a white circle to simulate tumor
                center = (np.random.randint(50, 174), np.random.randint(50, 174))
                radius = np.random.randint(10, 40)
                cv2.circle(img, center, radius, (255, 255, 255), -1)
            
            # Add random noise
            noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            filename = os.path.join(path, f"sample_{i}.jpg")
            cv2.imwrite(filename, img)
            
    print("Dummy dataset generation complete.")

if __name__ == "__main__":
    create_dummy_data("data/raw")
