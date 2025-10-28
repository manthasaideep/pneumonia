#!/usr/bin/env python3
"""
Debug GUI to check model predictions
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os

class DebugGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detection - Debug Mode")
        self.root.geometry("800x600")
        
        self.model = None
        self.current_image_path = None
        
        self.load_model()
        self.create_gui()
        
    def load_model(self):
        try:
            h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
            if h5_files:
                model_file = max(h5_files, key=lambda x: os.path.getsize(x))
                self.model = tf.keras.models.load_model(model_file)
                print(f"Model loaded: {model_file}")
                return True
            return False
        except Exception as e:
            print(f"Model error: {e}")
            return False
    
    def create_gui(self):
        tk.Label(self.root, text="Pneumonia Detection - Debug Mode", 
                font=('Arial', 16, 'bold')).pack(pady=20)
        
        tk.Button(self.root, text="Upload X-Ray Image", font=('Arial', 12), 
                 bg='blue', fg='white', padx=20, pady=10,
                 command=self.upload_image).pack(pady=10)
        
        self.image_label = tk.Label(self.root, text="No image", bg='lightgray', 
                                   width=40, height=15)
        self.image_label.pack(pady=20)
        
        tk.Button(self.root, text="Analyze (Show All Outputs)", font=('Arial', 12), 
                 bg='red', fg='white', padx=20, pady=10,
                 command=self.analyze_debug, state='disabled').pack(pady=10)
        
        self.results_text = tk.Text(self.root, height=15, width=80)
        self.results_text.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Add interpretation buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Interpret as Normal", bg='green', fg='white',
                 command=lambda: self.set_interpretation("NORMAL")).pack(side='left', padx=5)
        tk.Button(button_frame, text="Interpret as Pneumonia", bg='red', fg='white',
                 command=lambda: self.set_interpretation("PNEUMONIA")).pack(side='left', padx=5)
        
        self.analyze_btn = None
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select X-Ray",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                
                image = Image.open(file_path)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Enable analyze button
                for widget in self.root.winfo_children():
                    if isinstance(widget, tk.Button) and "Analyze" in widget.cget("text"):
                        widget.config(state='normal')
                        
            except Exception as e:
                messagebox.showerror("Error", f"Image load failed: {e}")
    
    def analyze_debug(self):
        if not self.current_image_path or not self.model:
            return
        
        try:
            # Preprocess
            img = cv2.imread(self.current_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred = self.model.predict(img, verbose=0)
            
            # Show all debug info
            debug_text = f"""DEBUG ANALYSIS RESULTS:
=========================

Image: {os.path.basename(self.current_image_path)}

Raw Prediction Output:
{pred}

Prediction Shape: {pred.shape}
Number of classes: {pred.shape[1]}

"""
            
            if pred.shape[1] == 1:  # Binary
                score = float(pred[0][0])
                debug_text += f"""BINARY CLASSIFICATION:
Raw score: {score:.6f}

Interpretation Options:
1. If score > 0.5 = Pneumonia: {"PNEUMONIA" if score > 0.5 else "NORMAL"} ({score*100:.1f}%)
2. If score < 0.5 = Normal: {"NORMAL" if score < 0.5 else "PNEUMONIA"} ({(1-score)*100:.1f}%)

Recommended: {"PNEUMONIA" if score > 0.5 else "NORMAL"}
"""
            else:  # Multi-class
                debug_text += f"""MULTI-CLASS CLASSIFICATION:
Class scores: {pred[0]}

Class 0 (likely Normal): {pred[0][0]:.6f} ({pred[0][0]*100:.1f}%)
Class 1 (likely Pneumonia): {pred[0][1]:.6f} ({pred[0][1]*100:.1f}%)

Highest class: {np.argmax(pred[0])}
Recommended: {"NORMAL" if np.argmax(pred[0]) == 0 else "PNEUMONIA"}
"""
            
            debug_text += f"""
=========================
MANUAL INTERPRETATION:
Look at the image and tell the system what it should be by clicking the buttons below.
This will help determine the correct interpretation logic.
"""
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, debug_text)
            
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, f"Error: {e}")
    
    def set_interpretation(self, correct_label):
        current_text = self.results_text.get(1.0, tk.END)
        interpretation = f"\n\nYOUR INTERPRETATION: {correct_label}\n"
        interpretation += "This helps determine if the model logic needs to be inverted.\n"
        
        self.results_text.insert(tk.END, interpretation)

def main():
    root = tk.Tk()
    app = DebugGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()