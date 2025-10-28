#!/usr/bin/env python3
"""
Demo script for the Pneumonia Detection System
This script demonstrates how to use the trained model for predictions
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess image for CNN prediction"""
    # Read image with unchanged flag to preserve channels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image")

    # Handle alpha channel and grayscale
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # CLAHE on luminance for robust contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Resize
    img_eq = cv2.resize(img_eq, img_size)

    # Normalize
    img_eq = img_eq.astype('float32') / 255.0

    # Add batch dimension
    img_eq = np.expand_dims(img_eq, axis=0)

    return img_eq

def predict_pneumonia(image_path, model_path='pneumonia_model.h5'):
    """Make pneumonia prediction on a single image"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first using: python train_model.py")
        return None
    
    # Load model
    try:
        # Prefer best model if present in same dir
        if os.path.exists('best_pneumonia_model.h5'):
            model_path = 'best_pneumonia_model.h5'
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file '{image_path}' not found!")
        return None
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Make prediction with simple TTA (hflip)
    img_flipped = processed_img[:, :, ::-1, :]
    batch = np.vstack([processed_img, img_flipped])
    proba_batch = model.predict(batch, verbose=0).reshape(-1)
    prediction_proba = float(np.mean(proba_batch))
    prediction = 1 if prediction_proba > 0.5 else 0
    confidence = round(prediction_proba * 100, 2) if prediction == 1 else round((1 - prediction_proba) * 100, 2)
    
    # Display results
    label = "Pneumonia" if prediction == 1 else "Normal"
    
    print(f"\n🔍 Analysis Results for: {os.path.basename(image_path)}")
    print(f"📊 Prediction: {label}")
    print(f"🎯 Confidence: {confidence}%")
    print(f"📈 Pneumonia Probability: {round(prediction_proba * 100, 2)}%")
    
    return {
        'label': label,
        'confidence': confidence,
        'probability': round(prediction_proba * 100, 2),
        'prediction': prediction
    }

def demo_with_dataset():
    """Demo function to test all images in the dataset"""
    dataset_dir = 'dataset'
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return
    
    print("🫁 Pneumonia Detection System - Demo")
    print("=" * 50)
    
    # Get all image files
    image_files = [f for f in os.listdir(dataset_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No image files found in '{dataset_dir}'")
        return
    
    print(f"📁 Found {len(image_files)} images in dataset")
    print("\n" + "=" * 50)
    
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(dataset_dir, image_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        result = predict_pneumonia(image_path)
        if result:
            results.append({
                'file': image_file,
                **result
            })
    
    # Summary
    if results:
        print("\n" + "=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)
        
        normal_count = sum(1 for r in results if r['label'] == 'Normal')
        pneumonia_count = sum(1 for r in results if r['label'] == 'Pneumonia')
        
        print(f"📊 Total Images Analyzed: {len(results)}")
        print(f"✅ Normal Cases: {normal_count}")
        print(f"⚠️  Pneumonia Cases: {pneumonia_count}")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"🎯 Average Confidence: {round(avg_confidence, 2)}%")
        
        print("\n📝 Detailed Results:")
        for result in results:
            status_icon = "✅" if result['label'] == 'Normal' else "⚠️"
            print(f"  {status_icon} {result['file']}: {result['label']} ({result['confidence']}%)")

def main():
    """Main demo function"""
    print("🫁 Pneumonia Detection System - Demo Script")
    print("=" * 60)
    
    # Check if we should run demo on dataset
    if os.path.exists('dataset'):
        demo_with_dataset()
    else:
        print("❌ Dataset directory not found!")
        print("Please ensure you have images in the 'dataset' folder")
    
    print("\n" + "=" * 60)
    print("💡 To use the web interface, run: python pneumonia.py")
    print("🌐 Then open: http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    main()

