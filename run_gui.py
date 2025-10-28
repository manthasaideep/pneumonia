#!/usr/bin/env python3
"""
Simple launcher for the Pneumonia Detection GUI
"""

import sys
import os
import subprocess

def ensure_packages_installed(packages_to_install):
    """Attempt to install missing packages into the current interpreter."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages_to_install]
        print(f"Attempting to install missing packages: {' '.join(packages_to_install)}")
        subprocess.check_call(cmd)
        return True
    except Exception as install_err:
        print(f"Failed to install packages automatically: {install_err}")
        return False

def check_requirements():
    """Check if required packages are available.

    Returns list of human-readable package specifiers that are missing.
    """
    missing_packages = []

    # tkinter
    try:
        import tkinter  # noqa: F401
    except Exception:
        missing_packages.append('tkinter')

    # tensorflow
    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing_packages.append('tensorflow')

    # Pillow (module is PIL)
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        missing_packages.append('Pillow')

    # OpenCV (module is cv2). Accept either opencv-python or opencv-python-headless
    try:
        import cv2  # noqa: F401
    except Exception:
        # Try a quick install of headless variant as it is more reliable in some setups
        # Do not auto-install here; just mark as missing. Auto-install handled in main().
        missing_packages.append('opencv-python')

    return missing_packages

def check_model():
    """Check if the trained model exists"""
    return os.path.exists('pneumonia_model.h5')

def main():
    # Avoid non-ASCII characters to prevent UnicodeEncodeError on some Windows consoles
    print("Pneumonia Detection GUI Launcher")
    print("=" * 50)
    
    # Check requirements
    print("Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        # Attempt automatic installation
        to_install = []
        if 'tensorflow' in missing:
            # Use a widely compatible TF on Windows/Py311
            to_install.append('tensorflow==2.16.1')
        if 'Pillow' in missing:
            to_install.append('Pillow')
        if 'opencv-python' in missing:
            # Prefer headless for broader compatibility
            to_install.append('opencv-python-headless')

        if to_install and ensure_packages_installed(to_install):
            # Re-check after install
            missing = check_requirements()
        
        if missing:
            print(f"Missing packages after auto-install: {', '.join(missing)}")
            print("Please install missing packages manually:")
            print("pip install tensorflow==2.16.1 opencv-python-headless pillow")
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
    
    # Launch GUI
    print("Launching GUI application...")
    try:
        import simple_working_gui
        simple_working_gui.main()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("Please check the error and try again")
