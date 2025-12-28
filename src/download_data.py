import kagglehub
import shutil
import os

def download_and_setup_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    
    # Define target path
    target_path = os.path.join(os.getcwd(), "data", "real")
    
    # Clean previous real data if exists
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    
    # Copy to project directory
    print(f"Copying to {target_path}...")
    shutil.copytree(path, target_path)
    print("Dataset setup complete.")

if __name__ == "__main__":
    download_and_setup_data()
