import os
import pathlib

# Centralized BASE_DIR resolution
# Centralized BASE_DIR resolution
# app/config.py -> app -> agent-ai-2035 (Root)
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()

DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
# MODELS_DIR = r"E:\raghotham\plato_new\frontierr\sxc\Train"
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

# Model Configuration Directory - Assuming it's inside the project root
MODEL_CONFIG_DIR = os.path.join(MODELS_DIR, "Model_1")
os.makedirs(MODEL_CONFIG_DIR, exist_ok=True)

MAX_CAPTION_LENGTH = 100
TRAINING_JSON_PATH = os.path.join(MODEL_CONFIG_DIR, "Training.json")
EVALUATION_JSON_PATH = os.path.join(MODEL_CONFIG_DIR, "Evaluation.json")
TESTING_JSON_PATH = os.path.join(MODEL_CONFIG_DIR, "Testing.json")
# TESTING_JSON_PATH moved to dynamic logic in training_service.py
STATUS_FILE_PATH = os.path.join(MODEL_CONFIG_DIR, "Status.txt")