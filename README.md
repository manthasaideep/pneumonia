# 🫁 Pneumonia Detection System

A deep learning-based system for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs). This project provides an efficient, accurate, and cost-effective diagnostic tool that can assist medical professionals in identifying pneumonia early, particularly in resource-constrained environments.

## 🌟 Features

- **Deep Learning Model**: Uses a state-of-the-art CNN architecture optimized for medical image analysis
- **Web Interface**: User-friendly Flask web application with modern UI
- **Real-time Analysis**: Fast prediction with confidence scores
- **Data Augmentation**: Robust preprocessing with image augmentation techniques
- **Comprehensive Metrics**: Detailed evaluation using accuracy, precision, recall, and F1-score
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

The system consists of:

1. **CNN Model**: A deep convolutional neural network with:
   - Multiple convolutional layers with batch normalization
   - Max pooling for feature extraction
   - Global average pooling to reduce overfitting
   - Dense layers with dropout for classification
   - Data augmentation layers for improved generalization

2. **Web Application**: Flask-based interface with:
   - Drag-and-drop file upload
   - Real-time image processing
   - Detailed result visualization
   - Responsive design

3. **Data Processing Pipeline**:
   - Image normalization and resizing
   - Data augmentation (rotation, zoom, flip, etc.)
   - Train/validation/test split
   - Comprehensive evaluation metrics

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- Optional: CUDA-compatible GPU for faster training

## 🚀 Installation

1. **Clone or download the project files**

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv pneumonia_env
   source pneumonia_env/bin/activate  # On Windows: pneumonia_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The system is designed to work with chest X-ray images. The current dataset includes:
- Normal chest X-rays
- Pneumonia-affected chest X-rays (bacterial and viral)

**Dataset Structure**:
```
dataset/
├── IM-0001-0001.jpeg      # Normal case
├── IM-0007-0001.jpeg      # Normal case
├── person100_bacteria_475.jpeg  # Pneumonia case
└── person101_bacteria_486.jpeg  # Pneumonia case
```

## 🎯 Usage

### Step 1: Train the Model

Before using the web application, you need to train the CNN model:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Create and compile the CNN model
- Train the model with data augmentation
- Evaluate performance using multiple metrics
- Save the trained model as `pneumonia_model.h5`
- Generate training history plots

### Step 2: Run the Web Application

After training the model, start the Flask application:

```bash
python pneumonia.py
```

The application will be available at: `http://localhost:5000`

### Step 3: Use the Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload an X-ray image** by:
   - Dragging and dropping the image onto the upload area, or
   - Clicking "Choose File" and selecting an image
3. **Click "Analyze X-Ray"** to get the prediction
4. **View the results** including:
   - Prediction (Normal/Pneumonia)
   - Confidence level
   - Detailed analysis information

## 📈 Model Performance

The CNN model provides comprehensive evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔧 Technical Details

### Model Architecture

```python
# Input: 224x224x3 RGB images
# Data Augmentation: Random flip, rotation, zoom
# Convolutional Layers: 32, 64, 128, 256 filters
# Batch Normalization: After each conv layer
# Pooling: Max pooling for feature reduction
# Global Average Pooling: Reduces overfitting
# Dense Layers: 512, 256, 1 neurons
# Dropout: 0.5, 0.3 for regularization
# Output: Sigmoid activation for binary classification
```

### Preprocessing Pipeline

1. **Image Loading**: RGB conversion and resizing to 224x224
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Data Augmentation**: 
   - Random horizontal flip
   - Random rotation (±20°)
   - Random zoom (±20%)
   - Random shear (±20%)
   - Random width/height shift (±20%)

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: Early stopping, learning rate reduction
- **Batch Size**: 8 (adjustable based on available memory)

## 📁 Project Structure

```
pneumonia/
├── dataset/                 # X-ray images
├── templates/              # HTML templates
│   ├── index.html         # Upload interface
│   └── result.html        # Results display
├── static/                # Static files (created automatically)
├── pneumonia.py           # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This system is designed for educational and research purposes only.

2. **Not for Medical Diagnosis**: The predictions should NOT be used as a substitute for professional medical diagnosis.

3. **Consult Healthcare Professionals**: Always consult qualified medical professionals for proper diagnosis and treatment.

4. **Limited Dataset**: The current model is trained on a small dataset and may not generalize well to all cases.

5. **Continuous Improvement**: The model should be retrained with larger, more diverse datasets for clinical use.

## 🔮 Future Enhancements

- [ ] Integration with larger medical datasets
- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Model ensemble for improved accuracy
- [ ] API endpoints for integration with hospital systems
- [ ] Real-time video analysis capabilities
- [ ] Mobile application development
- [ ] Cloud deployment options

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure you've run `train_model.py` first
   - Check that `pneumonia_model.h5` exists in the project directory

2. **Memory issues during training**:
   - Reduce batch size in `train_model.py`
   - Use smaller image size (e.g., 128x128 instead of 224x224)

3. **Poor model performance**:
   - Increase the dataset size
   - Adjust hyperparameters (learning rate, epochs)
   - Try different model architectures

4. **Web application not starting**:
   - Check if port 5000 is available
   - Ensure all dependencies are installed
   - Verify Flask installation

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Medical Image Analysis Papers](https://paperswithcode.com/task/medical-image-classification)

## 📄 License

This project is for educational purposes. Please ensure compliance with medical device regulations if used in clinical settings.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Remember**: This tool is for educational purposes only. Always consult medical professionals for proper diagnosis and treatment.

