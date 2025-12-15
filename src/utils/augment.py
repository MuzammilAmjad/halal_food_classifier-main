import os
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil


def clip_bbox(bbox):
    """
    Clip YOLO format bounding box to valid [0, 1] range.
    
    Args:
        bbox: [cx, cy, w, h] in normalized coordinates
        
    Returns:
        Clipped [cx, cy, w, h] or None if invalid
    """
    cx, cy, w, h = bbox
    
    # Calculate corners
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    
    # Clip to [0, 1]
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))
    
    # Recalculate center and dimensions
    new_w = x_max - x_min
    new_h = y_max - y_min
    
    # Check if box is still valid (minimum 1% of image in each dimension)
    if new_w < 0.01 or new_h < 0.01:
        return None
    
    new_cx = (x_min + x_max) / 2
    new_cy = (y_min + y_max) / 2
    
    return [new_cx, new_cy, new_w, new_h]


def augment_dataset(train_img_dir, train_label_dir, class_names, train_counts, 
                    balance_threshold=2.0, target_multiplier=1.0, verbose=True):
    """
    Augment minority classes in object detection dataset with YOLO format labels.
    
    Args:
        train_img_dir: Path to training images directory
        train_label_dir: Path to training labels directory
        class_names: List of class names
        train_counts: Array of current sample counts per class
        balance_threshold: Ratio threshold to trigger augmentation (default: 2.0)
        target_multiplier: Multiply mean count by this to set target (default: 1.0)
        verbose: Print detailed progress information
        
    Returns:
        dict: Statistics about augmentation process
    """
    
    # Calculate class imbalance
    min_c = train_counts.min()
    max_c = train_counts.max()
    ratio = max_c / min_c if min_c > 0 else float('inf')
    
    stats = {
        'imbalance_ratio': ratio,
        'min_count': min_c,
        'max_count': max_c,
        'mean_count': train_counts.mean(),
        'augmented': False,
        'total_created': 0,
        'per_class': {}
    }
    
    if verbose:
        print(f"Imbalance ratio (max/min): {ratio:.2f}")
        print(f"Class distribution: min={min_c}, max={max_c}, mean={train_counts.mean():.1f}")
    
    if ratio <= balance_threshold:
        if verbose:
            print("\n✓ Dataset is balanced – no augmentation needed")
        return stats
    
    if verbose:
        print("\n🔄 Dataset is imbalanced → Starting augmentation for minority classes\n")
    
    stats['augmented'] = True
    
    # Setup augmentation directories
    aug_img_dir = Path(train_img_dir).parent / 'aug_images'
    aug_lbl_dir = Path(train_label_dir).parent / 'aug_labels'
    aug_img_dir.mkdir(exist_ok=True)
    aug_lbl_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"Augmentation directories:")
        print(f"  Images: {aug_img_dir}")
        print(f"  Labels: {aug_lbl_dir}\n")
    
    # Define augmentation pipeline
    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ColorJitter(p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_visibility=0.2,  # Keep boxes with at least 20% visible
        clip=True  # Attempt to clip boxes
    ))
    
    # Target number of samples per class
    target = int(train_counts.mean() * target_multiplier)
    if verbose:
        print(f"Target samples per class: {target}\n")
    
    # Track statistics
    aug_stats = {cls: 0 for cls in range(len(class_names))}
    total_created = 0
    
    # Augment each minority class
    for cls_id in range(len(class_names)):
        current_count = train_counts[cls_id]
        
        if current_count >= target:
            if verbose:
                print(f"✓ Class {cls_id} ({class_names[cls_id]}): {current_count} samples - sufficient")
            continue
        
        needed = target - current_count
        if verbose:
            print(f"⚠ Class {cls_id} ({class_names[cls_id]}): {current_count} samples → need {needed} more")
        
        # Find all images containing this class
        candidate_images = []
        for lbl_file in os.listdir(train_label_dir):
            if not lbl_file.endswith('.txt'):
                continue
            
            lbl_path = os.path.join(train_label_dir, lbl_file)
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            if int(float(parts[0])) == cls_id:
                                base_name = lbl_file.replace('.txt', '')
                                candidate_images.append(base_name)
                                break
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Error reading {lbl_file}: {e}")
                continue
        
        if not candidate_images:
            if verbose:
                print(f"  ⚠ No images found for class {cls_id}\n")
            continue
        
        if verbose:
            print(f"  Found {len(candidate_images)} candidate images")
        
        # Generate augmented samples
        created = 0
        attempts = 0
        max_attempts = needed * 5  # Allow more attempts
        failed_count = 0
        
        while created < needed and attempts < max_attempts:
            attempts += 1
            
            # Select random source image
            base_name = random.choice(candidate_images)
            
            # Find image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = os.path.join(train_img_dir, base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path is None:
                continue
            
            lbl_path = os.path.join(train_label_dir, base_name + '.txt')
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Parse YOLO labels
            bboxes = []
            labels = []
            
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        label = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Validate and clip original coordinates
                        clipped = clip_bbox([cx, cy, w, h])
                        if clipped:
                            bboxes.append(clipped)
                            labels.append(label)
            except Exception:
                continue
            
            if not bboxes:
                continue
            
            # Apply augmentation
            try:
                transformed = aug_pipeline(image=img, bboxes=bboxes, class_labels=labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                # Double-check and clip all bounding boxes after augmentation
                valid_bboxes = []
                valid_labels = []
                
                for bbox, label in zip(aug_bboxes, aug_labels):
                    clipped = clip_bbox(bbox)
                    if clipped:
                        valid_bboxes.append(clipped)
                        valid_labels.append(label)
                
                if not valid_bboxes:
                    failed_count += 1
                    continue
                
                aug_bboxes = valid_bboxes
                aug_labels = valid_labels
                
            except Exception as e:
                failed_count += 1
                continue
            
            # Save augmented image and label
            aug_name = f"aug_cls{cls_id}_{created}_{base_name}"
            aug_img_path = aug_img_dir / f"{aug_name}.jpg"
            aug_lbl_path = aug_lbl_dir / f"{aug_name}.txt"
            
            # Write image
            success = cv2.imwrite(str(aug_img_path), aug_img)
            if not success:
                continue
            
            # Write labels
            try:
                with open(aug_lbl_path, 'w') as f:
                    for label, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{int(label)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            except Exception:
                aug_img_path.unlink()  # Remove orphaned image
                continue
            
            created += 1
            aug_stats[cls_id] += 1
            total_created += 1
        
        if verbose:
            print(f"  ✓ Created {created} augmented samples (attempts: {attempts}, failed: {failed_count})\n")
        
        stats['per_class'][cls_id] = {
            'created': created,
            'attempts': attempts,
            'failed': failed_count
        }
    
    # Sanity checks
    if verbose:
        print("\n" + "="*60)
        print("SANITY CHECKS")
        print("="*60)
    
    # Check 1: Verify directories exist
    if verbose:
        print(f"\n1. Directory existence:")
        print(f"   Aug images dir exists: {aug_img_dir.exists()}")
        print(f"   Aug labels dir exists: {aug_lbl_dir.exists()}")
    
    # Check 2: Count files
    aug_images = list(aug_img_dir.glob('*.jpg')) + list(aug_img_dir.glob('*.png'))
    aug_labels = list(aug_lbl_dir.glob('*.txt'))
    
    stats['files_created'] = {
        'images': len(aug_images),
        'labels': len(aug_labels)
    }
    
    if verbose:
        print(f"\n2. File counts:")
        print(f"   Augmented images: {len(aug_images)}")
        print(f"   Augmented labels: {len(aug_labels)}")
        print(f"   Expected: {total_created}")
    
    if len(aug_images) != len(aug_labels):
        if verbose:
            print(f"   ⚠ WARNING: Mismatch between images and labels!")
    
    # Check 3: Verify file pairs
    orphaned_images = []
    orphaned_labels = []
    
    for img_path in aug_images:
        lbl_path = aug_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            orphaned_images.append(img_path.name)
    
    for lbl_path in aug_labels:
        img_exists = False
        for ext in ['.jpg', '.png']:
            if (aug_img_dir / f"{lbl_path.stem}{ext}").exists():
                img_exists = True
                break
        if not img_exists:
            orphaned_labels.append(lbl_path.name)
    
    stats['orphaned'] = {
        'images': len(orphaned_images),
        'labels': len(orphaned_labels)
    }
    
    if verbose:
        print(f"\n3. Verifying image-label pairs...")
        if orphaned_images:
            print(f"   ⚠ Orphaned images (no label): {len(orphaned_images)}")
        if orphaned_labels:
            print(f"   ⚠ Orphaned labels (no image): {len(orphaned_labels)}")
        if not orphaned_images and not orphaned_labels:
            print(f"   ✓ All files properly paired")
    
    # Check 4: Augmentation statistics
    if verbose:
        print(f"\n4. Augmentation statistics by class:")
        for cls_id, count in aug_stats.items():
            if count > 0:
                print(f"   Class {cls_id} ({class_names[cls_id]}): +{count} samples")
    
    # Merge augmented data
    if total_created > 0:
        if verbose:
            print(f"\n5. Merging augmented data into training set...")
        
        merged_imgs = 0
        merged_lbls = 0
        
        # Copy images
        for img_path in aug_images:
            dest = Path(train_img_dir) / img_path.name
            shutil.copy2(img_path, dest)
            merged_imgs += 1
        
        # Copy labels
        for lbl_path in aug_labels:
            dest = Path(train_label_dir) / lbl_path.name
            shutil.copy2(lbl_path, dest)
            merged_lbls += 1
        
        stats['merged'] = {
            'images': merged_imgs,
            'labels': merged_lbls
        }
        
        if verbose:
            print(f"   ✓ Merged {merged_imgs} images")
            print(f"   ✓ Merged {merged_lbls} labels")
        
        # Verify merge
        final_train_count = len([f for f in os.listdir(train_img_dir) 
                                if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        final_label_count = len([f for f in os.listdir(train_label_dir) if f.endswith('.txt')])
        
        stats['final_counts'] = {
            'images': final_train_count,
            'labels': final_label_count
        }
        
        if verbose:
            print(f"\n6. Post-merge verification:")
            print(f"   Total training images: {final_train_count}")
            print(f"   Total training labels: {final_label_count}")
            print("\n" + "="*60)
            print(f"✓ AUGMENTATION COMPLETE: {total_created} samples created and merged")
            print("="*60)
    else:
        if verbose:
            print("\n⚠ No augmented samples were created")
    
    stats['total_created'] = total_created
    
    return stats
