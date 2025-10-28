#!/usr/bin/env python3
"""
Launcher for Enhanced Pneumonia Detection GUI
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are available"""
    required_packages = ['tkinter', 'tensorflow', 'opencv-python', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                from PIL import Image
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_model():
    """Check if the trained model exists"""
    return os.path.exists('pneumonia_model.h5')

def main():
    print("Enhanced Pneumonia Detection GUI Launcher")
    print("=" * 50)
    
    # Check requirements
    print("Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Please install missing packages:")
        print("pip install tensorflow opencv-python pillow")
        return
    
    print("All required packages are available")
    
    # Check model
    print("Checking trained model...")
    if not check_model():
        print("Trained model not found!")
        print("Please train the model first:")
        print("python train_model.py")
        return
    
    print("Trained model found")
    
    # Launch enhanced GUI
    print("Launching Enhanced GUI application...")
    try:
        import enhanced_pneumonia_gui
        enhanced_pneumonia_gui.main()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("Please check the error and try again")

if __name__ == "__main__":
    main()



