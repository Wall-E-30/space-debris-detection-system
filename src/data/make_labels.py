import os
import json
import shutil

# 1. Define paths
raw_train_dir = 'data/raw/SwissCube/training'
out_images_dir = 'data/processed/images/train'
out_labels_dir = 'data/processed/labels/train'

# Ensure output directories exist
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

# 2. Image dimensions (Assuming 1024x1024, adjust if your images are 640x480, etc.)
IMG_WIDTH = 1024.0
IMG_HEIGHT = 1024.0

print("Starting dataset flattening and conversion...")

# 3. Loop through every folder in the training directory
for seq_folder in os.listdir(raw_train_dir):
    seq_path = os.path.join(raw_train_dir, seq_folder)
    
    # Only process directories that start with 'seq_'
    if not os.path.isdir(seq_path) or not seq_folder.startswith('seq_'):
        continue
        
    gt_info_path = os.path.join(seq_path, 'scene_gt_info.json')
    rgb_folder = os.path.join(seq_path, 'rgb')
    
    if not os.path.exists(gt_info_path):
        # BOP format often nests files under a '000000' subdirectory
        nested_path = os.path.join(seq_path, '000000')
        if os.path.isdir(nested_path):
            gt_info_path = os.path.join(nested_path, 'scene_gt_info.json')
            rgb_folder = os.path.join(nested_path, 'rgb')
            
    if not os.path.exists(gt_info_path):
        continue
        
    # Open the JSON for this specific sequence
    with open(gt_info_path, 'r') as f:
        gt_info = json.load(f)
        
    # 4. Loop through every image in the JSON
    for image_id_str, objects in gt_info.items():
        # BOP format usually zero-pads image filenames to 6 digits (e.g., 000001.png)
        img_id_int = int(image_id_str)
        orig_img_name = f"{img_id_int:06d}.png" 
        orig_img_path = os.path.join(rgb_folder, orig_img_name)
        
        # Create a unique name to prevent overwriting (e.g., seq_000001_000001)
        unique_base_name = f"{seq_folder}_{img_id_int:06d}"
        new_img_path = os.path.join(out_images_dir, f"{unique_base_name}.png")
        new_label_path = os.path.join(out_labels_dir, f"{unique_base_name}.txt")
        
        # Copy the image over to the processed folder
        if os.path.exists(orig_img_path):
            shutil.copy(orig_img_path, new_img_path)
        else:
            # Fallback just in case the images are .jpg instead of .png
            orig_img_name_jpg = f"{img_id_int:06d}.jpg"
            orig_img_path_jpg = os.path.join(rgb_folder, orig_img_name_jpg)
            if os.path.exists(orig_img_path_jpg):
                shutil.copy(orig_img_path_jpg, new_img_path.replace('.png', '.jpg'))
            else:
                print(f"Warning: Could not find image for {orig_img_path}. Skipping.")
                continue 
        
        # 5. Generate and write the YOLO label
        with open(new_label_path, 'w') as label_file:
            for obj in objects:
                bbox = obj['bbox_visib']
                x_min, y_min, width, height = bbox
                
                # Convert to YOLO center format and normalize
                x_center = (x_min + (width / 2.0)) / IMG_WIDTH
                y_center = (y_min + (height / 2.0)) / IMG_HEIGHT
                norm_w = width / IMG_WIDTH
                norm_h = height / IMG_HEIGHT
                
                # Write: class_id x_center y_center width height
                label_file.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

print(f"Done! All images and labels are securely flattened into {out_images_dir} and {out_labels_dir}")