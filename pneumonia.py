import os
import cv2
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained CNN model (prefer best model if available)
model = None
try:
    model_path = None
    if os.path.exists('best_pneumonia_model.h5'):
        model_path = 'best_pneumonia_model.h5'
    elif os.path.exists('pneumonia_model.h5'):
        model_path = 'pneumonia_model.h5'

    if model_path:
        model = keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully from {model_path}!")
    else:
        print("Warning: Model file not found. Please train the model first using train_model.py")
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    model = None

def preprocess_image(image_path):
    """Preprocess image for CNN prediction"""
    # Read image with unchanged flag to preserve channels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image")

    # Handle alpha channel
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # If grayscale, convert to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # CLAHE on luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Resize
    img_eq = cv2.resize(img_eq, (224, 224))

    # Normalize
    img_eq = img_eq.astype('float32') / 255.0

    # Add batch dimension
    img_eq = np.expand_dims(img_eq, axis=0)

    return img_eq

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please train the model first.", 500
        
    if 'xray' not in request.files:
        return "No file uploaded", 400
    file = request.files['xray']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        processed_img = preprocess_image(filepath)

        # Test-time augmentation: original + horizontal flip average
        img_flipped = processed_img[:, :, ::-1, :]
        batch = np.vstack([processed_img, img_flipped])
        proba_batch = model.predict(batch, verbose=0).reshape(-1)
        prediction_proba = float(np.mean(proba_batch))
        prediction = 1 if prediction_proba > 0.5 else 0
        confidence = round(prediction_proba * 100, 2) if prediction == 1 else round((1 - prediction_proba) * 100, 2)
        
        label = "Pneumonia" if prediction == 1 else "Normal"
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                             image=file.filename, 
                             label=label, 
                             confidence=confidence,
                             probability=round(prediction_proba * 100, 2))
    except Exception as e:
        # Clean up uploaded file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)