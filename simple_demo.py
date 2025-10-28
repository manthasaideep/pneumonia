#!/usr/bin/env python3
"""
Simple demo to show how the Pneumonia Detection GUI works
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

def predict_pneumonia_demo(image_path):
    """Demo function to show prediction with accuracy"""
    
    # Check if model exists
    if not os.path.exists('pneumonia_model.h5'):
        print("ERROR: Model file 'pneumonia_model.h5' not found!")
        print("Please train the model first using: python train_model.py")
        return None
    
    # Load model
    try:
        model = keras.models.load_model('pneumonia_model.h5')
        print("SUCCESS: Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found!")
        return None
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction_proba = model.predict(processed_img, verbose=0)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0
    confidence = round(prediction_proba * 100, 2) if prediction == 1 else round((1 - prediction_proba) * 100, 2)
    
    # Display results
    label = "Pneumonia" if prediction == 1 else "Normal"
    probability = round(prediction_proba * 100, 2)
    
    print(f"\nPNEUMONIA DETECTION RESULTS")
    print(f"=" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {label}")
    print(f"Accuracy: {confidence}%")
    print(f"Pneumonia Probability: {probability}%")
    print(f"Model: CNN Deep Learning")
    
    if label == "Pneumonia":
        print(f"\nRESULT: PNEUMONIA DETECTED")
        print(f"   Accuracy: {confidence}%")
        print(f"   IMPORTANT: This is for educational purposes only!")
        print(f"   Please consult a medical professional immediately!")
    else:
        print(f"\nRESULT: NORMAL (No Pneumonia)")
        print(f"   Accuracy: {confidence}%")
        print(f"   Good news: No pneumonia detected!")
        print(f"   Always consult a medical professional for proper diagnosis!")
    
    return {
        'label': label,
        'accuracy': confidence,
        'probability': probability,
        'prediction': prediction
    }

def main():
    """Main demo function"""
    print("Pneumonia Detection System - Demo")
    print("=" * 60)
    print("This demo shows how the GUI works:")
    print("1. Upload an X-ray image")
    print("2. System predicts: Normal or Pneumonia")
    print("3. Shows accuracy percentage")
    print("4. If Pneumonia: Shows accuracy percentage prominently")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory '{dataset_dir}' not found!")
        print("Please ensure you have images in the 'dataset' folder")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(dataset_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"ERROR: No image files found in '{dataset_dir}'")
        return
    
    print(f"Found {len(image_files)} images in dataset")
    print("\n" + "=" * 60)
    
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(dataset_dir, image_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        result = predict_pneumonia_demo(image_path)
        if result:
            results.append({
                'file': image_file,
                **result
            })
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        normal_count = sum(1 for r in results if r['label'] == 'Normal')
        pneumonia_count = sum(1 for r in results if r['label'] == 'Pneumonia')
        
        print(f"Total Images Analyzed: {len(results)}")
        print(f"Normal Cases: {normal_count}")
        print(f"Pneumonia Cases: {pneumonia_count}")
        
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        print(f"Average Accuracy: {round(avg_accuracy, 2)}%")
        
        print("\nDetailed Results:")
        for result in results:
            status_icon = "NORMAL" if result['label'] == 'Normal' else "PNEUMONIA"
            print(f"  {result['file']}: {result['label']} (Accuracy: {result['accuracy']}%)")
    
    print("\n" + "=" * 60)
    print("To use the GUI interface, run: python pneumonia_gui.py")
    print("The GUI will show:")
    print("   - Prediction: Normal or Pneumonia")
    print("   - Accuracy: Percentage confidence")
    print("   - Visual indicators and progress bars")
    print("=" * 60)

if __name__ == "__main__":
    main()



