#!/usr/bin/env python3
"""
Quick Start Script for Pneumonia Detection System
This script provides an easy way to get started with the system
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'tensorflow',
        'flask',
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def check_dataset():
    """Check if dataset exists and has images"""
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return False
    
    image_files = [f for f in os.listdir(dataset_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in '{dataset_dir}'")
        return False
    
    print(f"Found {len(image_files)} images in dataset")
    return True

def train_model():
    """Train the pneumonia detection model"""
    print("Training the pneumonia detection model...")
    print("This may take several minutes depending on your system...")
    
    try:
        subprocess.check_call([sys.executable, 'train_model.py'])
        print("Model training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e}")
        return False

def start_web_app():
    """Start the Flask web application"""
    print("Starting the web application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.check_call([sys.executable, 'pneumonia.py'])
    except KeyboardInterrupt:
        print("\nWeb application stopped")
    except subprocess.CalledProcessError as e:
        print(f"Error starting web application: {e}")

def main():
    """Main quick start function"""
    print("Pneumonia Detection System - Quick Start")
    print("=" * 60)
    
    # Step 1: Check requirements
    print("\nStep 1: Checking requirements...")
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        if not install_requirements():
            print("Failed to install packages. Please install manually:")
            print("pip install -r requirements.txt")
            return
    else:
        print("All required packages are installed!")
    
    # Step 2: Check dataset
    print("\nStep 2: Checking dataset...")
    if not check_dataset():
        print("Please ensure you have X-ray images in the 'dataset' folder")
        return
    
    # Step 3: Check if model exists
    print("\nStep 3: Checking trained model...")
    if not os.path.exists('pneumonia_model.h5'):
        print("Trained model not found. Training model...")
        if not train_model():
            print("Model training failed. Please check the error messages above.")
            return
    else:
        print("Trained model found!")
    
    # Step 4: Start web application
    print("\nStep 4: Starting web application...")
    start_web_app()

if __name__ == "__main__":
    main()
