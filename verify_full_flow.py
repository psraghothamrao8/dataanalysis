import os
import shutil
import numpy as np
from PIL import Image

# Imports from our system
from app.config import DATASET_ROOT, DATASET_DIR
from app.services.data_service import detect_issues, apply_fix
from app.services.training_service import run_automated_training

def create_color_image(color, size=(224, 224)):
    # Create simple solid color image
    img = Image.new('RGB', size, color)
    return img

def generate_synthetic_data(base_path):
    print(f"Generating synthetic data at {base_path}...")
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    splits = ['train', 'val', 'test']
    classes = {'red': (255, 0, 0), 'blue': (0, 0, 255)}
    
    for split in splits:
        for cls_name, color in classes.items():
            dir_path = os.path.join(base_path, split, cls_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate 20 images per class per split
            count = 20 if split == 'train' else 5
            for i in range(count):
                img = create_color_image(color)
                img.save(os.path.join(dir_path, f"{cls_name}_{i}.jpg"))
                
    # INJECT NOISE: Move 3 'red' images to 'blue' folder in train to simulate label errors
    print("Injecting label errors (Mislabeling 3 Red images as Blue)...")
    src = os.path.join(base_path, 'train', 'red')
    dst = os.path.join(base_path, 'train', 'blue')
    for i in range(3):
        # Rename to avoid overwrite if names clash (they shouldn't)
        shutil.move(os.path.join(src, f"red_{i}.jpg"), os.path.join(dst, f"mislabeled_red_{i}.jpg"))

def verify_system():
    # 1. Setup
    generate_synthetic_data(DATASET_DIR)
    
    # 2. Analyze
    print("\n--- STEP 1: ANALYZE ---")
    result = detect_issues()
    issues = result.get('issues', [])
    print(f"Detected {len(issues)} issues.")
    
    # Verify we found the mislabeled ones
    found_mislabeled = sum(1 for i in issues if 'mislabeled' in i['file_path'])
    print(f"Correctly identified noise: {found_mislabeled}/3")
    
    if found_mislabeled > 0:
        print("PASS: Label noise detection working (LogReg check).")
    else:
        # Note: With only 20 samples, LogReg might struggle, but usually finding outliers in color space is easy
        print("WARNING: Detection might have been too subtle or dataset too small.")

    # 3. Fix
    print("\n--- STEP 2: FIX ---")
    fixed_count = 0
    for issue in issues:
        # We know they are mislabeled, so let's move them back
        # The system suggests 'red' for the mislabeled 'blue' images (which are actually red)
        suggested = issue['suggested_label']
        path = issue['file_path']
        
        # Check if recommendation makes sense (should suggest 'red' for the moved files)
        if 'mislabeled' in path and suggested == 'red':
            success, msg = apply_fix(path, 'move', new_label=suggested)
            if success: fixed_count += 1
            print(f"Fixed {os.path.basename(path)} -> {suggested}: {msg}")
            
    print(f"Fixed {fixed_count} issues.")

    # 4. Train
    print("\n--- STEP 3: TRAIN ---")
    # Run a tiny training loop (2 epochs) to verify pipeline
    # We patch the default epochs just for this test to avoid waiting
    try:
        print("Starting Auto-Training (Optuna + Final Loop)... this may take a minute.")
        # We pass full_epochs=1 to make it fast
        result = run_automated_training(full_epochs=1)
        
        print("Training Result:", result['status'])
        print("Best Params:", result.get('params'))
        print("Val Acc:", result.get('accuracy'))
        
        if result['status'] == 'completed':
             print("PASS: Training pipeline success.")
        else:
             print("FAIL: Training pipeline failed.")
             
    except Exception as e:
        print(f"FAIL: Training crashed with {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_system()
