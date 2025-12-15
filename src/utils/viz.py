import pandas as pd
import numpy as np
import os, random
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import matplotlib.patches as patches

def plot_samples(img_dir, label_dir, class_names, n=3):
    """Plot sample images with bounding boxes from YOLO format labels"""
    # Get list of image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return
    
    # Sample random images
    n_samples = min(n, len(img_files))
    sampled_files = random.sample(img_files, n_samples)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_samples, figsize=(6*n_samples, 6))
    if n_samples == 1:
        axes = [axes]
    
    for ax, img_file in zip(axes, sampled_files):
        # Read image
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Read corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Draw bounding boxes if label exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert YOLO format (normalized) to pixel coordinates
                        x_center = cx * width
                        y_center = cy * height
                        box_w = w * width
                        box_h = h * height
                        
                        # Calculate corner coordinates
                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)
                        
                        # Draw rectangle and label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Add class label
                        label_text = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                        cv2.putText(img, label_text, (x1, max(y1-10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Error processing label in {label_file}: {e}")
                        continue
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_file, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def count_classes(label_dir, n_classes):
    counts = np.zeros(n_classes, dtype=int)
    for txt in [f for f in os.listdir(label_dir) if f.endswith('.txt')]:
        with open(os.path.join(label_dir, txt)) as f:
            for line in f:
                # Safely convert class ID to int, handling 'X.0' format
                counts[int(float(line.split()[0]))] += 1
    return counts

