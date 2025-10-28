#!/usr/bin/env python3
"""
Simple Working Pneumonia Detection Model
Creates a model that can actually predict both Normal and Pneumonia cases
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_dir):
    """Load and preprocess the dataset"""
    images = []
    labels = []
    filenames = []
    
    print("Loading images with labels:")
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, filename)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0  # Normalize
            
            images.append(img)
            filenames.append(filename)
            
            # Label based on filename
            if 'bacteria' in filename.lower() or 'virus' in filename.lower():
                labels.append(1)  # Pneumonia
                print(f"  {filename} -> Pneumonia (1)")
            else:
                labels.append(0)  # Normal
                print(f"  {filename} -> Normal (0)")
                
    return np.array(images), np.array(labels), filenames

def create_simple_model():
    """Create a simple but effective CNN model"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(224, 224, 3)),
        
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_simple_model():
    """Train a simple model that can distinguish between classes"""
    
    # Load data
    print("Loading data...")
    X, y, filenames = load_and_preprocess_data('dataset')
    print(f"Loaded {len(X)} images")
    print(f"Normal cases: {np.sum(y == 0)}")
    print(f"Pneumonia cases: {np.sum(y == 1)}")
    
    # Create model
    print("Creating model...")
    model = create_simple_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # For very small datasets, use all data for training
    print("Training model on all data...")
    
    # Train with all data (no validation split for such small dataset)
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=2,
        verbose=1
    )
    
    # Test predictions
    print("\nTesting predictions:")
    print("=" * 50)
    
    for i, (img, true_label, filename) in enumerate(zip(X, y, filenames)):
        # Make prediction
        pred_proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = 1 if pred_proba > 0.5 else 0
        
        # Calculate confidence
        confidence = round(pred_proba * 100, 2) if pred_label == 1 else round((1 - pred_proba) * 100, 2)
        
        # Display results
        true_class = "Pneumonia" if true_label == 1 else "Normal"
        pred_class = "Pneumonia" if pred_label == 1 else "Normal"
        
        status = "CORRECT" if pred_label == true_label else "WRONG"
        
        print(f"{filename}")
        print(f"  True: {true_class} | Predicted: {pred_class} | {status}")
        print(f"  Confidence: {confidence}% | Probability: {round(pred_proba * 100, 2)}%")
        print()
    
    # Save model
    model.save('pneumonia_model.h5')
    print("Model saved to pneumonia_model.h5")
    
    return model

def test_model():
    """Test the trained model"""
    if not os.path.exists('pneumonia_model.h5'):
        print("Model not found. Please train first.")
        return
    
    # Load model
    model = keras.models.load_model('pneumonia_model.h5')
    
    # Load data
    X, y, filenames = load_and_preprocess_data('dataset')
    
    print("\nModel Testing Results:")
    print("=" * 50)
    
    correct = 0
    total = len(X)
    
    for i, (img, true_label, filename) in enumerate(zip(X, y, filenames)):
        # Make prediction
        pred_proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = 1 if pred_proba > 0.5 else 0
        
        # Calculate confidence
        confidence = round(pred_proba * 100, 2) if pred_label == 1 else round((1 - pred_proba) * 100, 2)
        
        # Display results
        true_class = "Pneumonia" if true_label == 1 else "Normal"
        pred_class = "Pneumonia" if pred_label == 1 else "Normal"
        
        if pred_label == true_label:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"
        
        print(f"{filename}")
        print(f"  True: {true_class} | Predicted: {pred_class} | {status}")
        print(f"  Confidence: {confidence}% | Probability: {round(pred_proba * 100, 2)}%")
        print()
    
    accuracy = (correct / total) * 100
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    print("Simple Working Pneumonia Detection Model")
    print("=" * 50)
    
    # Train model
    model = train_simple_model()
    
    # Test model
    test_model()
    
    print("\nModel training completed!")
    print("The model should now be able to predict both Normal and Pneumonia cases.")



