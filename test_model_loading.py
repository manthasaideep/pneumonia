#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    # Check which model files exist
    model_files = []
    if os.path.exists('best_pneumonia_model.h5'):
        model_files.append('best_pneumonia_model.h5')
    if os.path.exists('pneumonia_model.h5'):
        model_files.append('pneumonia_model.h5')
    
    print(f"Found model files: {model_files}")
    
    if not model_files:
        print("ERROR: No model files found!")
        return False
    
    # Try to load the first available model
    model_path = model_files[0]
    print(f"Attempting to load: {model_path}")
    
    try:
        # Method 1: Standard loading
        print("Trying standard loading...")
        model = keras.models.load_model(model_path, compile=False)
        print("SUCCESS: Model loaded with standard method!")
        return True
    except Exception as e1:
        print(f"Standard loading failed: {e1}")
    
    try:
        # Method 2: With custom objects
        print("Trying custom objects loading...")
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Dense': tf.keras.layers.Dense,
            'Flatten': tf.keras.layers.Flatten,
            'Dropout': tf.keras.layers.Dropout,
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D
        }
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print("SUCCESS: Model loaded with custom objects!")
        return True
    except Exception as e2:
        print(f"Custom objects loading failed: {e2}")
    
    try:
        # Method 3: With safe_mode=False
        print("Trying safe_mode=False loading...")
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        print("SUCCESS: Model loaded with safe_mode=False!")
        return True
    except Exception as e3:
        print(f"Safe mode loading failed: {e3}")
    
    print("ERROR: All loading methods failed!")
    return False

def test_prediction():
    """Test if the model can make predictions"""
    print("\nTesting model prediction...")
    
    # Create a dummy image for testing
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Preprocess the image (same as in GUI)
    img = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    print(f"Preprocessed image shape: {img.shape}")
    
    # Try to load model and make prediction
    model_files = []
    if os.path.exists('best_pneumonia_model.h5'):
        model_files.append('best_pneumonia_model.h5')
    if os.path.exists('pneumonia_model.h5'):
        model_files.append('pneumonia_model.h5')
    
    if not model_files:
        print("ERROR: No model files found for prediction test!")
        return False
    
    model_path = model_files[0]
    
    try:
        # Try loading with custom objects
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Dense': tf.keras.layers.Dense,
            'Flatten': tf.keras.layers.Flatten,
            'Dropout': tf.keras.layers.Dropout,
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D
        }
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)
        print(f"SUCCESS: Model prediction shape: {prediction.shape}")
        print(f"Prediction value: {prediction[0][0] if len(prediction[0]) > 0 else prediction[0]}")
        return True
        
    except Exception as e:
        print(f"ERROR: Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Model Loading Test ===")
    success = test_model_loading()
    
    if success:
        print("\n=== Prediction Test ===")
        test_prediction()
    
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")
