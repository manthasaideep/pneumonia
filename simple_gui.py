#!/usr/bin/env python3
"""
Simple Pneumonia Detection GUI - Compatible version
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
from PIL import Image, ImageTk

# Try different import strategies for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    print("Using tensorflow.keras")
except ImportError:
    try:
        import tensorflow as tf
        keras = tf.keras
        print("Using tf.keras")
    except AttributeError:
        try:
            import keras
            print("Using standalone keras")
        except ImportError:
            print("ERROR: Could not import TensorFlow or Keras")
            exit(1)

class SimplePneumoniaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f8f9fa')
        
        # Initialize model
        self.model = None
        self.load_model()
        
        # Variables
        self.selected_image_path = None
        self.preview_image = None
        
        # Create GUI elements
        self.create_widgets()
        
    def load_model(self):
        """Load the trained pneumonia detection model"""
        try:
            model_path = None
            # Prefer best model if present
            if os.path.exists('best_pneumonia_model.h5'):
                model_path = 'best_pneumonia_model.h5'
            elif os.path.exists('pneumonia_model.h5'):
                model_path = 'pneumonia_model.h5'

            if model_path:
                # Use compile=False to improve compatibility
                self.model = keras.models.load_model(model_path, compile=False)
                print(f"Model loaded successfully from {model_path}!")
            else:
                messagebox.showerror("Error", "Model file not found!\nPlease train the model first using train_model.py")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f8f9fa')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title section
        title_frame = tk.Frame(main_container, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Pneumonia Detection System", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Content frame
        content_frame = tk.Frame(main_container, bg='#f8f9fa')
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Image upload and preview
        left_panel = tk.Frame(content_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Upload section
        upload_frame = tk.Frame(left_panel, bg='white')
        upload_frame.pack(fill='x', padx=20, pady=20)
        
        upload_label = tk.Label(upload_frame, text="Upload Chest X-Ray Image", 
                               font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        upload_label.pack(pady=(0, 15))
        
        # Upload button
        self.upload_btn = tk.Button(upload_frame, text="Choose X-Ray Image", 
                                   command=self.upload_image, font=('Arial', 12, 'bold'),
                                   bg='#3498db', fg='white', relief='flat', 
                                   padx=30, pady=12, cursor='hand2')
        self.upload_btn.pack(pady=10)
        
        # File path display
        self.file_path_var = tk.StringVar()
        self.file_path_label = tk.Label(upload_frame, textvariable=self.file_path_var, 
                                       font=('Arial', 10), bg='white', fg='#7f8c8d',
                                       wraplength=400)
        self.file_path_label.pack(pady=5)
        
        # Image preview section
        preview_frame = tk.Frame(left_panel, bg='white')
        preview_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        preview_label = tk.Label(preview_frame, text="Image Preview", 
                                font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        preview_label.pack(pady=(0, 10))
        
        # Image display
        self.image_label = tk.Label(preview_frame, text="No image selected", 
                                   font=('Arial', 12), bg='#ecf0f1', fg='#7f8c8d',
                                   width=30, height=15, relief='sunken', bd=2)
        self.image_label.pack(pady=10, padx=10)
        
        # Right panel - Analysis and results
        right_panel = tk.Frame(content_frame, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Analysis section
        analysis_frame = tk.Frame(right_panel, bg='white')
        analysis_frame.pack(fill='x', padx=20, pady=20)
        
        analysis_label = tk.Label(analysis_frame, text="Analysis", 
                                 font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        analysis_label.pack(pady=(0, 15))
        
        # Analyze button
        self.analyze_btn = tk.Button(analysis_frame, text="Analyze X-Ray", 
                                    command=self.analyze_image, font=('Arial', 12, 'bold'),
                                    bg='#e74c3c', fg='white', relief='flat', 
                                    padx=40, pady=15, cursor='hand2', state='disabled')
        self.analyze_btn.pack(pady=10)
        
        # Results section
        results_frame = tk.Frame(right_panel, bg='white')
        results_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        results_label = tk.Label(results_frame, text="Results", 
                                font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        results_label.pack(pady=(0, 15))
        
        # Results display area
        self.results_display = tk.Frame(results_frame, bg='#f8f9fa', relief='sunken', bd=2)
        self.results_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initial message
        initial_msg = tk.Label(self.results_display, text="Upload an X-ray image and click 'Analyze X-Ray' to get results", 
                              font=('Arial', 11), bg='#f8f9fa', fg='#7f8c8d')
        initial_msg.pack(expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please upload an X-ray image")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             font=('Arial', 10), bg='#34495e', fg='white', 
                             anchor='w', padx=15, pady=5)
        status_bar.pack(side='bottom', fill='x')
        
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Chest X-Ray Image",
            filetypes=file_types
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.file_path_var.set(f"Selected: {os.path.basename(file_path)}")
            self.display_image_preview(file_path)
            self.analyze_btn.config(state='normal')
            self.status_var.set("Image loaded successfully - Ready for analysis")
    
    def display_image_preview(self, image_path):
        """Display image preview in the GUI"""
        try:
            # Load and resize image for preview
            image = Image.open(image_path)
            
            # Calculate dimensions to fit in preview area
            max_width, max_height = 300, 250
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.preview_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.preview_image, text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.image_label.config(image="", text="Failed to load image")
    
    def preprocess_image(self, image_path):
        """Preprocess image for CNN prediction"""
        try:
            # Read image with unchanged flag to preserve channels
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception("Failed to read image")

            # Handle alpha channel if present
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # If grayscale, convert to BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Apply CLAHE on luminance to improve contrast robustly
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

            # Resize to model input size
            img_eq = cv2.resize(img_eq, (224, 224))

            # Normalize
            img_eq = img_eq.astype('float32') / 255.0

            # Add batch dimension
            img_eq = np.expand_dims(img_eq, axis=0)

            return img_eq
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def analyze_image(self):
        """Analyze the uploaded image for pneumonia"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please check if model files exist.")
            return
        
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        # Disable analyze button and show loading
        self.analyze_btn.config(state='disabled', text="Analyzing...", bg='#95a5a6')
        self.status_var.set("Analyzing image... Please wait")
        
        # Run analysis in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the analysis in a separate thread"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(self.selected_image_path)

            # Test-time augmentation: average original and horizontal flip
            img_batch = processed_img
            img_flipped = processed_img[:, :, ::-1, :]
            batch = np.vstack([img_batch, img_flipped])

            proba_batch = self.model.predict(batch, verbose=0).reshape(-1)
            prediction_proba = float(np.mean(proba_batch))
            prediction = 1 if prediction_proba > 0.5 else 0
            confidence = round(prediction_proba * 100, 2) if prediction == 1 else round((1 - prediction_proba) * 100, 2)
            
            # Determine result
            label = "Pneumonia" if prediction == 1 else "Normal"
            probability = round(prediction_proba * 100, 2)
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, label, confidence, probability, prediction_proba)
            
        except Exception as e:
            # Show error in main thread
            self.root.after(0, self.show_error, str(e))

    def display_results(self, label, confidence, probability, prediction_proba):
        """Display analysis results in the GUI"""
        # Clear previous results
        for widget in self.results_display.winfo_children():
            widget.destroy()
        
        # Determine colors and icons
        if label == "Pneumonia":
            result_color = '#e74c3c'
            result_icon = "WARNING"
            bg_color = '#fdf2f2'
        else:
            result_color = '#27ae60'
            result_icon = "NORMAL"
            bg_color = '#f0f9f0'
        
        # Main result card
        result_card = tk.Frame(self.results_display, bg=bg_color, relief='raised', bd=3)
        result_card.pack(fill='x', pady=10, padx=10)
        
        # Result header
        header_frame = tk.Frame(result_card, bg=result_color)
        header_frame.pack(fill='x')
        
        result_text = tk.Label(header_frame, text=f"{result_icon} - {label}", 
                              font=('Arial', 18, 'bold'), fg='white', bg=result_color,
                              pady=15)
        result_text.pack()
        
        # Accuracy display
        accuracy_frame = tk.Frame(result_card, bg=bg_color)
        accuracy_frame.pack(fill='x', pady=20)
        
        accuracy_label = tk.Label(accuracy_frame, text="CONFIDENCE", 
                                 font=('Arial', 12, 'bold'), bg=bg_color, fg='#2c3e50')
        accuracy_label.pack()
        
        accuracy_value = tk.Label(accuracy_frame, text=f"{confidence}%", 
                                 font=('Arial', 28, 'bold'), bg=bg_color, fg=result_color)
        accuracy_value.pack(pady=5)
        
        # Detailed information
        details_frame = tk.Frame(result_card, bg=bg_color)
        details_frame.pack(fill='x', pady=20, padx=20)
        
        # Create details grid
        details_data = [
            ("Prediction:", label),
            ("Confidence Level:", f"{confidence}%"),
            ("Pneumonia Probability:", f"{probability}%"),
            ("Model Used:", "CNN Deep Learning")
        ]
        
        for metric, value in details_data:
            detail_row = tk.Frame(details_frame, bg=bg_color)
            detail_row.pack(fill='x', pady=2)
            
            metric_label = tk.Label(detail_row, text=metric, font=('Arial', 10, 'bold'), 
                                   bg=bg_color, fg='#2c3e50', width=18, anchor='w')
            metric_label.pack(side='left')
            
            value_label = tk.Label(detail_row, text=value, font=('Arial', 10), 
                                  bg=bg_color, fg='#34495e', anchor='w')
            value_label.pack(side='left', padx=(10, 0))
        
        # Warning/Info message
        message_frame = tk.Frame(result_card, bg=bg_color)
        message_frame.pack(fill='x', pady=20, padx=20)
        
        if label == 'Pneumonia':
            warning_text = "IMPORTANT: This result is for educational purposes only. Please consult a medical professional immediately for proper diagnosis and treatment."
            message_color = '#d63031'
        else:
            warning_text = "Good news: No pneumonia detected. However, always consult a medical professional for proper health assessment."
            message_color = '#00b894'
        
        message_label = tk.Label(message_frame, text=warning_text, font=('Arial', 9), 
                                bg=bg_color, fg=message_color, wraplength=350, justify='center')
        message_label.pack(pady=10)
        
        # Re-enable analyze button
        self.analyze_btn.config(state='normal', text="Analyze X-Ray", bg='#e74c3c')
        self.status_var.set("Analysis completed successfully")
    
    def show_error(self, error_message):
        """Show error message"""
        messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{error_message}")
        self.analyze_btn.config(state='normal', text="Analyze X-Ray", bg='#e74c3c')
        self.status_var.set("Analysis failed - Please try again")

def main():
    """Main function to run the simple GUI application"""
    root = tk.Tk()
    app = SimplePneumoniaGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()

