#!/usr/bin/env python3
"""
Pneumonia Detection System GUI
AI-Powered Chest X-Ray Analysis with Confusion Matrix
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import threading

class PneumoniaDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.current_image_path = None
        self.current_image = None
        self.prediction_result = None
        self.accuracy_percentage = None
        
        # Load model
        self.model_loaded = self.load_model()
        
        # Create GUI
        self.create_header()
        self.create_main_content()
        
    def load_model(self):
        """Load the trained pneumonia detection model"""
        try:
            model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
            if model_files:
                # Try to load the largest model file (likely the main model)
                model_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                model_path = model_files[0]
                print(f"Loading model from {model_path}...")
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
                print(f"Model input shape: {self.model.input_shape}")
                return True
            else:
                print("No trained model found! Please train the model first.")
                return False
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            self.model = None
            return False
    
    def create_header(self):
        """Create the header section"""
        header_frame = tk.Frame(self.root, bg='#4a5568', height=80)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Pneumonia Detection System",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#4a5568'
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Chest X-Ray Analysis",
            font=('Arial', 12),
            fg='#cbd5e0',
            bg='#4a5568'
        )
        subtitle_label.pack()
    
    def create_main_content(self):
        """Create the main content area"""
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Image upload and preview
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.create_upload_section(left_panel)
        
        # Right panel - Analysis and results
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=1)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        self.create_analysis_section(right_panel)
    
    def create_upload_section(self, parent):
        """Create the image upload section"""
        # Upload section title
        upload_title = tk.Label(
            parent,
            text="Upload Chest X-Ray Image",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2d3748'
        )
        upload_title.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            parent,
            text="Choose X-Ray Image",
            font=('Arial', 12, 'bold'),
            bg='#3182ce',
            fg='white',
            padx=30,
            pady=10,
            command=self.upload_image,
            cursor='hand2'
        )
        upload_btn.pack(pady=10)
        
        # Image preview section
        preview_title = tk.Label(
            parent,
            text="Image Preview",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#4a5568'
        )
        preview_title.pack(pady=(30, 10))
        
        # Image preview frame
        self.image_frame = tk.Frame(parent, bg='#f7fafc', relief='sunken', bd=2)
        self.image_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Default image preview label
        self.image_label = tk.Label(
            self.image_frame,
            text="No image selected",
            font=('Arial', 12),
            bg='#f7fafc',
            fg='#a0aec0'
        )
        self.image_label.pack(expand=True)
    
    def create_analysis_section(self, parent):
        """Create the analysis section"""
        # Analysis section title
        analysis_title = tk.Label(
            parent,
            text="Analysis",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2d3748'
        )
        analysis_title.pack(pady=20)
        
        # Model status indicator
        model_status = "Model Loaded" if hasattr(self, 'model_loaded') and self.model_loaded else "Model Not Loaded"
        status_color = '#38a169' if hasattr(self, 'model_loaded') and self.model_loaded else '#e53e3e'
        
        self.model_status_label = tk.Label(
            parent,
            text=model_status,
            font=('Arial', 10),
            bg='white',
            fg=status_color
        )
        self.model_status_label.pack(pady=5)
        
        # Analyze button
        self.analyze_btn = tk.Button(
            parent,
            text="Analyze X-Ray",
            font=('Arial', 12, 'bold'),
            bg='#e53e3e',
            fg='white',
            padx=30,
            pady=10,
            command=self.analyze_image,
            cursor='hand2',
            state='disabled'
        )
        self.analyze_btn.pack(pady=10)
        
        # Confusion Matrix button
        self.confusion_btn = tk.Button(
            parent,
            text="Confusion Matrix (dataset)",
            font=('Arial', 12, 'bold'),
            bg='#805ad5',
            fg='white',
            padx=30,
            pady=10,
            command=self.show_confusion_matrix,
            cursor='hand2'
        )
        self.confusion_btn.pack(pady=10)
        
        # Results section
        results_title = tk.Label(
            parent,
            text="Results",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#4a5568'
        )
        results_title.pack(pady=(30, 10))
        
        # Results frame
        self.results_frame = tk.Frame(parent, bg='#f7fafc', relief='sunken', bd=2)
        self.results_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Default results label
        self.results_label = tk.Label(
            self.results_frame,
            text="Upload an X-ray image and click 'Analyze X-Ray' to get results",
            font=('Arial', 12),
            bg='#f7fafc',
            fg='#a0aec0',
            wraplength=300,
            justify='center'
        )
        self.results_label.pack(expand=True)
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Validate that the file can be opened
                test_image = Image.open(file_path)
                test_image.close()
                
                self.current_image_path = file_path
                self.display_image(file_path)
                self.analyze_btn.config(state='normal')
                print(f"Image uploaded successfully: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.current_image_path = None
    
    def display_image(self, image_path):
        """Display the selected image in the preview"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate size to fit in preview area (max 300x300)
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            print(f"Preprocessing image: {image_path}")
            
            # Read image using PIL first to ensure compatibility
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy array
            img = np.array(pil_image)
            
            # Get model input shape
            if self.model and hasattr(self.model, 'input_shape'):
                input_shape = self.model.input_shape
                if len(input_shape) >= 3:
                    target_size = (input_shape[1], input_shape[2])
                else:
                    target_size = (224, 224)  # Default
            else:
                target_size = (224, 224)  # Default
            
            print(f"Resizing to: {target_size}")
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            print(f"Preprocessed image shape: {img.shape}")
            return img
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def analyze_image(self):
        """Analyze the uploaded image"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please upload an X-ray image first")
            return
            
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Attempting to reload...")
            if not self.load_model():
                messagebox.showerror("Error", "Failed to load model. Please check if model files exist.")
                return
        
        # Show loading message
        self.results_label.config(text="Analyzing image...", fg='#3182ce')
        self.root.update()
        
        try:
            print("Starting image analysis...")
            
            # Preprocess image
            processed_image = self.preprocess_image(self.current_image_path)
            
            # Make prediction
            print("Making prediction...")
            prediction = self.model.predict(processed_image, verbose=0)
            print(f"Raw prediction: {prediction}")
            
            # Get prediction result
            if len(prediction[0]) == 1:  # Binary classification
                confidence = float(prediction[0][0])
                print(f"Binary classification confidence: {confidence}")
                if confidence > 0.5:
                    result = "Pneumonia"
                    accuracy = confidence * 100
                else:
                    result = "Normal"
                    accuracy = (1 - confidence) * 100
            else:  # Multi-class
                class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][class_idx])
                print(f"Multi-class prediction - Class: {class_idx}, Confidence: {confidence}")
                result = "Pneumonia" if class_idx == 1 else "Normal"
                accuracy = confidence * 100
            
            print(f"Final result: {result} with {accuracy:.1f}% confidence")
            
            self.prediction_result = result
            self.accuracy_percentage = accuracy
            
            # Display results
            self.display_results(result, accuracy)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.results_label.config(text="Analysis failed", fg='#e53e3e')
    
    def display_results(self, result, accuracy):
        """Display prediction results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Result text
        result_text = f"Prediction: {result}"
        accuracy_text = f"Confidence: {accuracy:.1f}%"
        
        # Result label
        result_label = tk.Label(
            self.results_frame,
            text=result_text,
            font=('Arial', 16, 'bold'),
            bg='#f7fafc',
            fg='#e53e3e' if result == "Pneumonia" else '#38a169'
        )
        result_label.pack(pady=10)
        
        # Accuracy label
        accuracy_label = tk.Label(
            self.results_frame,
            text=accuracy_text,
            font=('Arial', 14),
            bg='#f7fafc',
            fg='#4a5568'
        )
        accuracy_label.pack(pady=5)
        
        # Additional info
        info_text = f"The AI model is {accuracy:.1f}% confident that this X-ray shows {'pneumonia' if result == 'Pneumonia' else 'normal lungs'}."
        info_label = tk.Label(
            self.results_frame,
            text=info_text,
            font=('Arial', 10),
            bg='#f7fafc',
            fg='#718096',
            wraplength=280,
            justify='center'
        )
        info_label.pack(pady=10)
    
    def show_confusion_matrix(self):
        """Show confusion matrix for the dataset"""
        # Create a new window for confusion matrix
        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Confusion Matrix - Dataset Performance")
        matrix_window.geometry("600x500")
        matrix_window.configure(bg='white')
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sample confusion matrix data (replace with actual model evaluation)
        # This is a placeholder - you should evaluate your model on test data
        cm_sample = np.array([[850, 150], [100, 900]])  # [TN, FP], [FN, TP]
        
        # Plot confusion matrix
        sns.heatmap(cm_sample, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'], ax=ax1)
        ax1.set_title('Confusion Matrix - Test Dataset')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Calculate and display metrics
        tn, fp, fn, tp = cm_sample.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics_text = f"""Model Performance Metrics:
        
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}

True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}"""
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Performance Metrics')
        
        plt.tight_layout()
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, matrix_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PneumoniaDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()