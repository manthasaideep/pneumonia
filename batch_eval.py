#!/usr/bin/env python3
"""
Batch evaluation utility for Pneumonia Detector.

Usage:
  python batch_eval.py [dataset_dir]

Behavior:
- If dataset_dir has subfolders whose names contain 'NORMAL' and 'PNEUMONIA',
  those are used as ground truth labels (0=Normal, 1=Pneumonia).
- Otherwise, labels are inferred from filenames:
  - contains 'normal' -> Normal (0)
  - contains 'bacteria' or 'virus' -> Pneumonia (1)
  Files without a clear label are skipped.

Outputs:
- Saves confusion_matrix.png in the dataset_dir
- Prints a short classification report to stdout
"""

import os
import sys
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
# Force using tf.keras consistently
keras = tf.keras


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def gather_with_subfolders(root_dir):
    paths, labels = [], []
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        name = sub.lower()
        if 'normal' in name:
            label = 0
        elif 'pneumonia' in name:
            label = 1
        else:
            continue
        for f in os.listdir(sub_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                paths.append(os.path.join(sub_path, f))
                labels.append(label)
    return paths, labels


def gather_with_filenames(root_dir):
    paths, labels = [], []
    for f in os.listdir(root_dir):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        fl = f.lower()
        if 'normal' in fl:
            label = 0
        elif 'bacteria' in fl or 'virus' in fl or 'pneumonia' in fl:
            label = 1
        else:
            # unknown label -> skip
            continue
        paths.append(os.path.join(root_dir, f))
        labels.append(label)
    return paths, labels


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    if not os.path.exists('pneumonia_model.h5'):
        print("ERROR: Model file 'pneumonia_model.h5' not found.")
        sys.exit(1)
    model = keras.models.load_model('pneumonia_model.h5', compile=False)

    # Gather labeled images
    paths, labels = gather_with_subfolders(dataset_dir)
    if not paths:
        paths, labels = gather_with_filenames(dataset_dir)
    if not paths:
        print("ERROR: No labeled images found. Provide subfolders NORMAL/PNEUMONIA or filename keywords.")
        sys.exit(1)

    # Predict
    y_true, y_pred = [], []
    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = []
        for p in batch:
            imgs.append(preprocess_image(p)[0])
        arr = np.stack(imgs, axis=0)
        proba = model.predict(arr, verbose=0)[:, 0]
        preds = (proba > 0.5).astype(int)
        y_pred.extend(preds.tolist())
        y_true.extend(labels[i:i+batch_size])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Pneumonia'])
    ax.set_yticklabels(['Normal', 'Pneumonia'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center', color='#2c3e50', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(dataset_dir, 'confusion_matrix.png')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'], digits=3)
    print(f"Saved confusion matrix to: {out_path}")
    print("\nClassification report:\n")
    print(report)


if __name__ == '__main__':
    main()


