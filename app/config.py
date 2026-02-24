import os
import pathlib

# Centralized BASE_DIR resolution
# Centralized BASE_DIR resolution
# app/config.py -> app -> agent-ai-2035 (Root)
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()

DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "path", "Solution")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure base dirs exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)

def get_data_paths():
    """
    Dynamically resolve the current dataset paths.
    Useful because the dataset folder name might change after a zip upload.
    """
    dataset_name = "mlcc" # Default
    
    if os.path.exists(DATASET_ROOT):
        # Find the first valid subdirectory that isn't hidden
        subdirs = [d for d in os.listdir(DATASET_ROOT) 
                   if os.path.isdir(os.path.join(DATASET_ROOT, d)) 
                   and not d.startswith('.') and d != '__pycache__']
        if subdirs:
            # If 'mlcc' exists, prefer it (backward comapt), else take the first one
            dataset_name = "mlcc" if "mlcc" in subdirs else subdirs[0]
            
    dataset_dir = os.path.join(DATASET_ROOT, dataset_name)
    
    return {
        "dataset_root": DATASET_ROOT,
        "dataset_dir": dataset_dir,
        "train": os.path.join(dataset_dir, "train"),
        "val": os.path.join(dataset_dir, "val"),
        "test": os.path.join(dataset_dir, "test")
    }

# DL Process Wrapper Configuration
# Default to a placeholder. User should set this env var or update this file.
DL_PROCESS_WRAPPER_PATH = os.environ.get("DL_PROCESS_WRAPPER_PATH", r"C:\Program Files\Samsung Electro-Mechanics\SEM DL Kit\SEM_DL_Kit\Scripts\DLProcessWrapper2.4\DLProcessWrapper.exe")

def get_latest_model_name():
    """Finds the highest Model_X folder in Solution/Train."""
    train_dir = os.path.join(MODELS_DIR, "Train")
    if not os.path.exists(train_dir):
        return "Model_1"
    
    max_num = 0
    for d in os.listdir(train_dir):
        if d.startswith("Model_") and os.path.isdir(os.path.join(train_dir, d)):
            try:
                num = int(d.split("_")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return f"Model_{max_num}" if max_num > 0 else "Model_1"

def get_model_paths(model_name=None):
    """Returns dynamic paths, optionally for a specific model_name, otherwise latest."""
    if model_name is None:
        model_name = get_latest_model_name()
    
    train_model_dir = os.path.join(MODELS_DIR, "Train", model_name)
    test_model_dir = os.path.join(MODELS_DIR, "Test", model_name)
    
    # Ensure they exist (during initial boot)
    os.makedirs(train_model_dir, exist_ok=True)
    os.makedirs(test_model_dir, exist_ok=True)
    
    return {
        "model_name": model_name,
        "train_dir": train_model_dir,
        "test_dir": test_model_dir,
        "training_json": os.path.join(train_model_dir, "Training.json"),
        "evaluation_json": os.path.join(train_model_dir, "Evaluation.json"),
        "testing_json": os.path.join(test_model_dir, "Testing.json"),
        "status_file": os.path.join(train_model_dir, "Status.txt")
    }

# Remove legacy module-level variables like MODEL_CONFIG_DIR and TRAINING_JSON_PATH
# Files needing these will call get_model_paths() directly.