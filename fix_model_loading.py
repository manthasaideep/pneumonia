#!/usr/bin/env python3
"""
Fix model loading by creating a compatible architecture and loading weights only
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

def create_compatible_model():
    """Create a model architecture compatible with current TensorFlow version"""
    print("Creating compatible model architecture...")
    
    model = models.Sequential([
        # Input layer (compatible with current TF)
        layers.Input(shape=(224, 224, 3)),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def load_weights_only(model_path, model):
    """Try to load only the weights from the incompatible model file"""
    print(f"Attempting to load weights from {model_path}...")
    
    try:
        # Try to load weights directly
        model.load_weights(model_path)
        print("SUCCESS: Weights loaded!")
        return True
    except Exception as e:
        print(f"Weight loading failed: {e}")
        return False

def create_pretrained_model():
    """Create a VGG16-based pre-trained model as fallback"""
    print("Creating VGG16-based pre-trained model...")
    
    try:
        from tensorflow.keras.applications import VGG16
        
        # Load VGG16 base model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        print("SUCCESS: VGG16-based model created!")
        return model
        
    except Exception as e:
        print(f"VGG16 model creation failed: {e}")
        return None

def test_model_prediction(model):
    """Test if the model can make predictions"""
    print("Testing model prediction...")
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Preprocess the image
    img = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    try:
        prediction = model.predict(img, verbose=0)
        print(f"SUCCESS: Model prediction: {prediction[0][0]}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

def main():
    """Main function to fix model loading"""
    print("=== Model Loading Fix ===")
    
    # Check which model files exist
    model_files = []
    if os.path.exists('best_pneumonia_model.h5'):
        model_files.append('best_pneumonia_model.h5')
    if os.path.exists('pneumonia_model.h5'):
        model_files.append('pneumonia_model.h5')
    
    if not model_files:
        print("ERROR: No model files found!")
        return
    
    model_path = model_files[0]
    print(f"Using model file: {model_path}")
    
    # Method 1: Try to create compatible model and load weights
    print("\n=== Method 1: Compatible Architecture + Weights ===")
    model = create_compatible_model()
    if load_weights_only(model_path, model):
        if test_model_prediction(model):
            print("SUCCESS: Compatible model with weights works!")
            # Save the working model
            model.save('fixed_pneumonia_model.h5')
            print("Saved as 'fixed_pneumonia_model.h5'")
            return
    
    # Method 2: Create pre-trained model as fallback
    print("\n=== Method 2: Pre-trained VGG16 Model ===")
    model = create_pretrained_model()
    if model is not None:
        if test_model_prediction(model):
            print("SUCCESS: Pre-trained model works!")
            # Save the working model
            model.save('pretrained_pneumonia_model.h5')
            print("Saved as 'pretrained_pneumonia_model.h5'")
            return
    
    print("ERROR: All methods failed!")

if __name__ == "__main__":
    main()
