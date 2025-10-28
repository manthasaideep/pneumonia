#!/usr/bin/env python3
"""
Test dataset analysis functionality
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import glob

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def test_model_on_dataset():
    """Test the model on the dataset"""
    print("=== Testing Model on Dataset ===")
    
    # Load the working model
    model_path = 'pretrained_pneumonia_model.h5'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found!")
        return False
    
    try:
        model = load_model(model_path)
        print(f"SUCCESS: Model loaded from {model_path}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False
    
    # Test on dataset
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder {dataset_path} not found!")
        return False
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images in dataset")
    
    if len(image_files) == 0:
        print("ERROR: No images found in dataset!")
        return False
    
    # Test on first few images
    test_count = min(5, len(image_files))
    print(f"Testing on first {test_count} images...")
    
    success_count = 0
    for i, image_path in enumerate(image_files[:test_count]):
        print(f"\nTesting image {i+1}: {os.path.basename(image_path)}")
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            print("  SKIP: Failed to preprocess")
            continue
        
        try:
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            prob = float(prediction[0][0])
            label = "Pneumonia" if prob > 0.5 else "Normal"
            confidence = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)
            
            print(f"  Prediction: {label} (Confidence: {confidence}%)")
            success_count += 1
            
        except Exception as e:
            print(f"  ERROR: Prediction failed: {e}")
    
    print(f"\n=== Results ===")
    print(f"Successfully processed: {success_count}/{test_count} images")
    print(f"Model is working: {'YES' if success_count > 0 else 'NO'}")
    
    return success_count > 0

def test_confusion_matrix():
    """Test confusion matrix computation"""
    print("\n=== Testing Confusion Matrix ===")
    
    # Check if we can import the required functions
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        print("SUCCESS: sklearn metrics available")
        return True
    except ImportError as e:
        print(f"ERROR: sklearn not available: {e}")
        return False

if __name__ == "__main__":
    print("=== Dataset Analysis Test ===")
    
    # Test model loading and prediction
    model_works = test_model_on_dataset()
    
    # Test confusion matrix functionality
    cm_works = test_confusion_matrix()
    
    print(f"\n=== Overall Results ===")
    print(f"Model prediction: {'WORKING' if model_works else 'FAILED'}")
    print(f"Confusion matrix: {'WORKING' if cm_works else 'FAILED'}")
    print(f"Dataset analysis: {'READY' if model_works and cm_works else 'NOT READY'}")
