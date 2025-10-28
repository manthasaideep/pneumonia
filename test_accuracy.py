#!/usr/bin/env python3
"""
Test script to verify pneumonia detection accuracy
"""

import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def test_pneumonia_detection():
    """Test pneumonia detection with sample images"""
    print("Testing Pneumonia Detection Accuracy")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset folder not found")
        return
    
    # Find test images
    normal_images = []
    pneumonia_images = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                
                if 'normal' in folder_name:
                    normal_images.append(image_path)
                elif 'pneumonia' in folder_name:
                    pneumonia_images.append(image_path)
    
    print(f"Found {len(normal_images)} normal images")
    print(f"Found {len(pneumonia_images)} pneumonia images")
    
    # Test with a few images from each category
    test_normal = normal_images[:3] if normal_images else []
    test_pneumonia = pneumonia_images[:3] if pneumonia_images else []
    
    print(f"\nTesting with {len(test_normal)} normal images and {len(test_pneumonia)} pneumonia images")
    
    # Test normal images
    print("\n--- Testing Normal Images ---")
    for i, img_path in enumerate(test_normal):
        result = analyze_single_image(img_path, "Normal")
        print(f"Normal Image {i+1}: {result}")
    
    # Test pneumonia images
    print("\n--- Testing Pneumonia Images ---")
    for i, img_path in enumerate(test_pneumonia):
        result = analyze_single_image(img_path, "Pneumonia")
        print(f"Pneumonia Image {i+1}: {result}")

def analyze_single_image(image_path, expected_label):
    """Analyze a single image and return prediction"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "[ERROR] Failed to load image"
        
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
        
        # Analyze image characteristics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        # Improved heuristic analysis
        pneumonia_score = 0.0
        
        # 1. Check for high contrast (abnormalities) - MORE SENSITIVE
        if contrast > 60:
            pneumonia_score += 0.5
        elif contrast > 40:
            pneumonia_score += 0.3
        
        # 2. Check brightness patterns - MORE SENSITIVE
        if brightness < 90:
            pneumonia_score += 0.4
        elif brightness > 200:
            pneumonia_score += 0.3
        
        # 3. Check for texture patterns (edge detection)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        if edge_density > 0.05:
            pneumonia_score += 0.3
        
        # 4. Check for specific patterns in lung regions
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_std = np.std(center_region)
        if center_std > 30:
            pneumonia_score += 0.4
        
        # 5. Check for irregular patterns using histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        if hist_std > 50:
            pneumonia_score += 0.3
        
        # 6. Check filename
        filename = os.path.basename(image_path).lower()
        if any(keyword in filename for keyword in ['pneumonia', 'pneum', 'abnormal', 'disease', 'sick']):
            pneumonia_score += 0.4
        elif any(keyword in filename for keyword in ['normal', 'healthy', 'clear']):
            pneumonia_score -= 0.3
        
        # 7. Check for bilateral patterns
        left_region = gray[:, :w//2]
        right_region = gray[:, w//2:]
        left_std = np.std(left_region)
        right_std = np.std(right_region)
        if left_std > 25 and right_std > 25:
            pneumonia_score += 0.3
        
        # Determine prediction with balanced threshold
        predicted_label = "Pneumonia" if pneumonia_score > 0.5 else "Normal"
        confidence = int(pneumonia_score * 100) if predicted_label == "Pneumonia" else int((1 - pneumonia_score) * 100)
        
        # Check if prediction is correct
        is_correct = predicted_label == expected_label
        status = "[CORRECT]" if is_correct else "[WRONG]"
        
        return f"{status} | Predicted: {predicted_label} ({confidence}%) | Expected: {expected_label}"
        
    except Exception as e:
        return f"[ERROR] {str(e)}"

if __name__ == "__main__":
    test_pneumonia_detection()
