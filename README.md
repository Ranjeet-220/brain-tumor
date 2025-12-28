# CNN-Based Brain Tumor Detection

## Overview
This project aims to detect brain tumors from MRI images using Convolutional Neural Networks (CNN).

## Project Structure
- `data/`: Contains dataset (raw and processed).
- `src/`: Source code for preprocessing, modeling, training, and evaluation.
- `models/`: Saved model files.
- `app.py`: Streamlit web application.
- `notebooks/`: Jupyter notebooks for experiments.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place your dataset in `data/raw/`.
3. Run training: `python src/train.py`
4. Run web app: `streamlit run app.py`
