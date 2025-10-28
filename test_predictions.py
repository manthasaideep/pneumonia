#!/usr/bin/env python3
"""
Test script to verify pneumonia detection predictions
"""

import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def test_model_loading():
    """Test if we can load the model"""
    print("Testing model loading...")
    
    model_files = ['best_pneumonia_model.h5', 'pneumonia_model.h5']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Found model: {model_file}")
            try:
                # Try different loading methods
                model = keras.models.load_model(model_file, compile=False)
                print(f"✅ Successfully loaded {model_file}")
                return model, model_file
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue
    
    print("❌ No models could be loaded")
    return None, None

def test_image_preprocessing():
    """Test image preprocessing"""
    print("\nTesting image preprocessing...")
    
    # Look for test images in dataset
    dataset_path = 'dataset'
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    print(f"Testing with image: {image_path}")
                    
                    try:
                        # Test preprocessing
                        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                            
                        # Handle different image formats
                        if img.ndim == 3 and img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            
                        # Apply CLAHE
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l_eq = clahe.apply(l)
                        lab_eq = cv2.merge((l_eq, a, b))
                        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
                        
                        # Resize and normalize
                        img_eq = cv2.resize(img_eq, (224, 224))
                        img_eq = img_eq.astype('float32') / 255.0
                        img_eq = np.expand_dims(img_eq, axis=0)
                        
                        print(f"✅ Successfully preprocessed {file}")
                        return img_eq, image_path
                        
                    except Exception as e:
                        print(f"❌ Failed to preprocess {file}: {e}")
                        continue
    
    print("❌ No images could be preprocessed")
    return None, None

def test_prediction(model, processed_image):
    """Test prediction with the model"""
    print("\nTesting prediction...")
    
    try:
        if model is None:
            print("❌ No model available for prediction")
            return None
            
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        # Determine result
        result = "Pneumonia" if probability > 0.5 else "Normal"
        confidence = round(probability * 100, 2) if result == "Pneumonia" else round((1 - probability) * 100, 2)
        
        print(f"✅ Prediction successful!")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence}%")
        print(f"   Raw Probability: {probability:.4f}")
        
        return {
            'result': result,
            'confidence': confidence,
            'probability': probability
        }
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def main():
    """Main test function"""
    print("Pneumonia Detection Test")
    print("=" * 50)
    
    # Test model loading
    model, model_file = test_model_loading()
    
    # Test image preprocessing
    processed_image, image_path = test_image_preprocessing()
    
    if processed_image is not None:
        # Test prediction
        prediction_result = test_prediction(model, processed_image)
        
        if prediction_result:
            print(f"\n✅ All tests passed!")
            print(f"Model: {model_file}")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Prediction: {prediction_result['result']} ({prediction_result['confidence']}%)")
        else:
            print(f"\n❌ Prediction test failed")
    else:
        print(f"\n❌ Image preprocessing test failed")

if __name__ == "__main__":
    main()


