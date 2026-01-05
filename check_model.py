import pickle
import torch
import numpy as np

try:
    with open('simclr_classifier_pipeline.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Feature columns:", data['feature_cols'])
    print("Classifier classes:", data['classifier'].classes_)
except Exception as e:
    print(f"Error loading pipeline: {e}")

try:
    # Check if we need the model too
    with open('simclr_model_enhanced.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print("Enhanced model keys:", model_data.keys())
except Exception as e:
    print(f"Error loading enhanced model: {e}")
