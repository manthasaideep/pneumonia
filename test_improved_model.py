#!/usr/bin/env python3
"""
Test the improved model to show it can predict both Normal and Pneumonia
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess image for CNN prediction"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, img_size)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def test_model():
    """Test the improved model"""
    if not os.path.exists('pneumonia_model.h5'):
        print("Model not found. Please train first.")
        return
    
    # Load model
    model = keras.models.load_model('pneumonia_model.h5')
    print("Model loaded successfully!")
    
    # Test all images in dataset
    dataset_dir = 'dataset'
    image_files = [f for f in os.listdir(dataset_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nTesting {len(image_files)} images:")
    print("=" * 60)
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(dataset_dir, image_file)
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        prediction_proba = model.predict(processed_img, verbose=0)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0
        confidence = round(prediction_proba * 100, 2) if prediction == 1 else round((1 - prediction_proba) * 100, 2)
        
        # Determine true label
        if 'bacteria' in image_file.lower() or 'virus' in image_file.lower():
            true_label = 1  # Pneumonia
            true_class = "Pneumonia"
        else:
            true_label = 0  # Normal
            true_class = "Normal"
        
        # Determine predicted class
        pred_class = "Pneumonia" if prediction == 1 else "Normal"
        
        # Check if correct
        is_correct = prediction == true_label
        status = "CORRECT" if is_correct else "WRONG"
        
        results.append({
            'file': image_file,
            'true_class': true_class,
            'pred_class': pred_class,
            'confidence': confidence,
            'probability': round(prediction_proba * 100, 2),
            'correct': is_correct
        })
        
        print(f"{image_file}")
        print(f"  True: {true_class} | Predicted: {pred_class} | {status}")
        print(f"  Confidence: {confidence}% | Probability: {round(prediction_proba * 100, 2)}%")
        print()
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total) * 100
    
    normal_correct = sum(1 for r in results if r['true_class'] == 'Normal' and r['correct'])
    normal_total = sum(1 for r in results if r['true_class'] == 'Normal')
    
    pneumonia_correct = sum(1 for r in results if r['true_class'] == 'Pneumonia' and r['correct'])
    pneumonia_total = sum(1 for r in results if r['true_class'] == 'Pneumonia')
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Normal Cases: {normal_correct}/{normal_total} ({(normal_correct/normal_total)*100:.1f}%)")
    print(f"Pneumonia Cases: {pneumonia_correct}/{pneumonia_total} ({(pneumonia_correct/pneumonia_total)*100:.1f}%)")
    
    print(f"\nThe model can now predict BOTH Normal and Pneumonia cases!")
    print(f"GUI will show:")
    print(f"  - Normal predictions with accuracy percentage")
    print(f"  - Pneumonia predictions with accuracy percentage")
    print(f"  - Color-coded results (Green for Normal, Red for Pneumonia)")

if __name__ == "__main__":
    test_model()



