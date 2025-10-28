#!/usr/bin/env python3
"""
Improved Pneumonia Detection Model Training
Better handling for small datasets and improved model architecture
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImprovedPneumoniaDetector:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the dataset with better labeling"""
        images = []
        labels = []
        
        print("Loading images with labels:")
        
        # Define class mapping based on filename patterns
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_dir, filename)
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0  # Normalize
                
                images.append(img)
                
                # Improved labeling logic
                if 'bacteria' in filename.lower() or 'virus' in filename.lower():
                    labels.append(1)  # Pneumonia
                    print(f"  {filename} -> Pneumonia (1)")
                else:
                    labels.append(0)  # Normal
                    print(f"  {filename} -> Normal (0)")
                    
        return np.array(images), np.array(labels)
    
    def create_improved_model(self):
        """Create an improved CNN model for pneumonia detection"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile the model with improved optimizer and loss function"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def train_model_with_improved_augmentation(self, model, X_train, y_train, X_val, y_val, epochs=100, batch_size=2):
        """Train the model with improved data augmentation for small datasets"""
        
        # Create more aggressive data augmentation for small datasets
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            shear_range=0.3,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        # Improved callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_pneumonia_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)
        
        print(f"Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")
        
        self.history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return metrics"""
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0.0
        
        print(f"\nModel Performance:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))
        except ValueError:
            print("Classification report not available for single test sample")
            print(f"Test sample prediction: {y_pred[0]} (0=Normal, 1=Pneumonia)")
            print(f"Test sample actual: {y_test[0]} (0=Normal, 1=Pneumonia)")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'confusion_matrix': cm
        }
    
    def test_predictions(self, model, X, y, filenames):
        """Test predictions on all images to verify model behavior"""
        print("\nTesting predictions on all images:")
        print("=" * 60)
        
        for i, (img, true_label, filename) in enumerate(zip(X, y, filenames)):
            # Make prediction
            pred_proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            pred_label = 1 if pred_proba > 0.5 else 0
            
            # Calculate confidence
            confidence = round(pred_proba * 100, 2) if pred_label == 1 else round((1 - pred_proba) * 100, 2)
            
            # Display results
            true_class = "Pneumonia" if true_label == 1 else "Normal"
            pred_class = "Pneumonia" if pred_label == 1 else "Normal"
            
            status = "✓" if pred_label == true_label else "✗"
            
            print(f"{status} {filename}")
            print(f"   True: {true_class} | Predicted: {pred_class}")
            print(f"   Confidence: {confidence}% | Probability: {round(pred_proba * 100, 2)}%")
            print()
    
    def save_model(self, model, model_path='pneumonia_model.h5'):
        """Save the trained model"""
        model.save(model_path)
        print(f"Model saved to {model_path}")

def main():
    # Initialize the detector
    detector = ImprovedPneumoniaDetector(img_size=(224, 224))
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = detector.load_and_preprocess_data('dataset')
    print(f"\nLoaded {len(X)} images")
    print(f"Normal cases: {np.sum(y == 0)}")
    print(f"Pneumonia cases: {np.sum(y == 1)}")
    
    # Get filenames for testing
    filenames = [f for f in os.listdir('dataset') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # For very small datasets, use all data for training with validation split
    if len(X) <= 6:
        print("\nSmall dataset detected. Using all data for training with validation split.")
        # For very small datasets, don't use stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        X_test, y_test = X_val, y_val
    else:
        # Split the data normally
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and compile model
    print("\nCreating improved model...")
    model = detector.create_improved_model()
    model = detector.compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Train the model with improved parameters
    print("\nTraining model with improved parameters...")
    epochs = 100  # More epochs for better learning
    batch_size = 2  # Very small batch size for small dataset
    
    detector.train_model_with_improved_augmentation(
        model, X_train, y_train, X_val, y_val, 
        epochs=epochs, batch_size=batch_size
    )
    
    # Test predictions on all images
    detector.test_predictions(model, X, y, filenames)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = detector.evaluate_model(model, X_test, y_test)
    
    # Save the model
    detector.save_model(model)
    
    print("\nTraining completed successfully!")
    print("The model should now be able to predict both Normal and Pneumonia cases correctly.")

if __name__ == "__main__":
    main()
