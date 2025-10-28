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

class PneumoniaDetector:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the dataset"""
        images = []
        labels = []
        
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
                
                # Label based on filename
                if 'bacteria' in filename.lower() or 'virus' in filename.lower():
                    labels.append(1)  # Pneumonia
                else:
                    labels.append(0)  # Normal
                    
        return np.array(images), np.array(labels)
    
    def create_model(self):
        """Create a CNN model for pneumonia detection"""
        model = keras.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Global Average Pooling instead of Flatten to reduce overfitting
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile the model with appropriate optimizer and loss function"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Train the model with data augmentation"""
        
        # Create data generators for augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)
        
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
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model, model_path='pneumonia_model.h5'):
        """Save the trained model"""
        model.save(model_path)
        print(f"Model saved to {model_path}")

def main():
    # Initialize the detector
    detector = PneumoniaDetector(img_size=(224, 224))
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = detector.load_and_preprocess_data('dataset')
    print(f"Loaded {len(X)} images")
    print(f"Normal cases: {np.sum(y == 0)}")
    print(f"Pneumonia cases: {np.sum(y == 1)}")
    
    # Split the data - handle small datasets
    if len(X) < 6:
        print("Warning: Small dataset detected. Using simple train/test split.")
        # For very small datasets, use simple split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        X_val, X_test, y_val, y_test = X_test, X_test, y_test, y_test
    else:
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
    print("Creating model...")
    model = detector.create_model()
    model = detector.compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Train the model - adjust parameters for small dataset
    print("Training model...")
    epochs = 20 if len(X) < 10 else 50  # Fewer epochs for small datasets
    batch_size = min(4, len(X_train))  # Smaller batch size for small datasets
    print(f"Training with {epochs} epochs and batch size {batch_size}")
    detector.train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = detector.evaluate_model(model, X_test, y_test)
    
    # Plot training history
    detector.plot_training_history()
    
    # Save the model
    detector.save_model(model)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
