import os
import copy
import time
import torch
import csv
import logging 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

try:
    import optuna
except ImportError:
    pass

from app.config import (
    MODELS_DIR, LOGS_DIR, get_data_paths, DATASET_ROOT,
    DL_PROCESS_WRAPPER_PATH, BASE_DIR, get_model_paths, get_latest_model_name
)
from app.logger_config import setup_logger
from app.services.data_service import CustomImageDataset, train_transform, val_transform
import subprocess
import json
import re
import glob
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX 1: Initialize logger broadly here, but we will CHANGE the file handler per run ---
# We point this to a generic 'startup.log' initially so we don't create empty training logs at boot.
logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "startup.log"), mode='a')



# --- Dataset Scanning & JSON Update Logic ---
def scan_dataset_and_update_configs():
    """
    Scans the dataset directory (Train/Val/Test) and updates 
    Training.json and Testing.json with the current file lists and class definitions.
    """
    logger.info("Scanning dataset and updating JSON configurations...")
    
    paths = get_data_paths()
    train_dir = paths['train']
    val_dir = paths['val']
    test_dir = paths['test']
    
    # 1. Identify Classes (Canonical List)
    # We look at train_dir for the source of truth for classes
    classes = []
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        logger.warning("No classes found in Train directory. Skipping JSON update.")
        return False

    # Map class name to index (0-based)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    # 2. Build classlst and classBin
    # Check for user-defined class mapping metadata
    meta_path = os.path.join(DATASET_ROOT, "dataset_meta.json")
    user_ok_classes = set()
    user_ng_classes = set()
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                user_ok_classes = {c.lower() for c in meta.get("ok_classes", [])}
                user_ng_classes = {c.lower() for c in meta.get("ng_classes", [])}
        except Exception as e:
            logger.warning(f"Failed to read dataset metadata: {e}")

    class_lst = []
    class_bin = []
    
    for cls_name in classes:
        # classlst entry
        class_lst.append({
            "ClassName": cls_name,
            "iClassWeights": 1
        })
        
        # classBin entry
        c_lower = cls_name.lower()
        
        if c_lower in user_ok_classes:
            bin_name = "OK"
        elif c_lower in user_ng_classes:
            bin_name = "NG"
        else:
            # Fallback Logic: If name is "OK" (case-insensitive) -> "OK", else "NG"
            bin_name = "OK" if c_lower == "ok" else "NG"
            
        class_bin.append({
            "classBinName": bin_name
        })
        

    # 3. Helper to scan images
    def scan_images(directory, dataset_id=1):
        img_list = []
        if not os.path.exists(directory):
            return img_list
            
        for cls_name in classes:
            cls_path = os.path.join(directory, cls_name)
            if not os.path.exists(cls_path):
                continue
                
            label_idx = class_to_idx.get(cls_name, 0)
            
            # Recursive scan for images
            # Using glob for simplicity
            pattern = os.path.join(cls_path, "**", "*")
            files = glob.glob(pattern, recursive=True)
            
            for f_path in files:
                if f_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_list.append({
                        "ImageName": os.path.abspath(f_path),
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": dataset_id,
                        "CaptionsSentence": None,
                        "TabularInfo": None
                    })
        return img_list

    train_imgs = scan_images(train_dir)
    val_imgs = scan_images(val_dir)
    test_imgs = scan_images(test_dir)
    
    logger.info(f"Found {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")
    logger.info(f"Classes identified: {classes}")

    # 4. Update Training.json
    paths_dict = get_model_paths()
    training_json_path = paths_dict["training_json"]
    model_name = paths_dict["model_name"]

    # Ensure Training.json exists by copying from template if needed
    if not os.path.exists(training_json_path):
        template_path = os.path.join(BASE_DIR, "training.json")
        if os.path.exists(template_path):
            try:
                shutil.copy(template_path, training_json_path)
                logger.info(f"Created {training_json_path} from template.")
            except Exception as e:
                logger.error(f"Failed to create Training.json from template: {e}")
                return False
        else:
             logger.error(f"Training.json not found and no template at {template_path}")
             return False

    if os.path.exists(training_json_path):
        try:
            with open(training_json_path, 'r') as f:
                t_data = json.load(f)
            
            # Update fields
            t_data["classlst"] = class_lst
            t_data["classBin"] = class_bin
            t_data["trainImglst"] = train_imgs
            t_data["ValImgList"] = val_imgs
            
            # Update Model counts
            if "Model" in t_data:
                t_data["Model"]["iTrainImgCount"] = len(train_imgs)
                t_data["Model"]["iValidationImgCount"] = len(val_imgs)
                t_data["Model"]["iTotalClasses"] = len(classes)
                # Ensure Paths are Local and Correct
                t_data["Model"]["SolutionDir"] = str(MODELS_DIR) + os.sep 
                t_data["Model"]["ModelDir"] = "Train" # Forever Train, as that's where model is saved
                t_data["Model"]["name"] = model_name
                
            with open(training_json_path, 'w') as f:
                json.dump(t_data, f, indent=2)
            logger.info(f"Updated {training_json_path}")
        except Exception as e:
            logger.error(f"Failed to update Training.json: {e}")
            return False
            
    # 5. Update Testing.json
    # Create it if it doesn't exist? Or only update if exists?
    # Uses TESTING_JSON_PATH
            
    # Logic: Read SolutionDir and ModelName from Training.json to find the REAL Testing.json path
    target_test_json = paths_dict["testing_json"]
    evaluation_json_path = paths_dict["evaluation_json"]

    # Create it if it doesn't exist? Or only update if exists?
    # User says Test and Eval formats are similar, unlike Training.
    if not os.path.exists(target_test_json):
         if os.path.exists(evaluation_json_path):
             try:
                 shutil.copy(evaluation_json_path, target_test_json)
                 logger.info(f"Created {target_test_json} from Evaluation.json template.")
             except Exception as e:
                 logger.warning(f"Failed to create Testing.json from Evaluation.json: {e}")
         elif os.path.exists(training_json_path):
             # Fallback to Training only if really needed, but might be wrong format.
             # Better to skip and warn? User said "Train is different". 
             # Let's avoid copying Training.json if possible.
             logger.warning(f"Testing.json missing at {target_test_json} and Evaluation.json not found. "
                            "Skipping creation to avoid using Training.json (incompatible format).")
    
    
    if os.path.exists(target_test_json):
        try:
            with open(target_test_json, 'r') as f:
                test_data = json.load(f)
            
            test_data["classlst"] = class_lst
            test_data["classBin"] = class_bin
            test_data["testImglst"] = test_imgs
            test_data["trainImglst"] = []
            test_data["ValImgList"] = []
            
            if "Model" in test_data:
                 test_data["Model"]["iTestImgCount"] = len(test_imgs)
                 test_data["Model"]["iTotalClasses"] = len(classes)
                 test_data["Model"]["iTrainImgCount"] = 0
                 test_data["Model"]["iValidationImgCount"] = 0
                 test_data["Model"]["name"] = model_name
                 # Don't update SolutionDir as requested
                 # test_data["Model"]["SolutionDir"] = str(MODELS_DIR) + os.sep
                 # test_data["Model"]["ModelDir"] = "Train"
            
            with open(target_test_json, 'w') as f:
                json.dump(test_data, f, indent=2)
            logger.info(f"Updated {target_test_json}")
        except Exception as e:
            logger.error(f"Failed to update Testing.json: {e}")
            return False

    # 6. Update Evaluation.json (In Place, preserving format)
    if os.path.exists(evaluation_json_path):
        try:
            with open(evaluation_json_path, 'r') as f:
                eval_data = json.load(f)
            
            # Update Images and Classes
            eval_data["classlst"] = class_lst
            eval_data["classBin"] = class_bin
            eval_data["testImglst"] = val_imgs + train_imgs # Evaluation runs on Train + Validation images
            eval_data["trainImglst"] = []
            eval_data["ValImgList"] = []
            
            eval_data["valImgList"] = []
            
            if "Model" in eval_data:
                 # User Request: Update iTrainImgCount instead of iTestImgCount in Evaluation.json
                 # Even though we use testImglst.
                 eval_data["Model"]["iTrainImgCount"] = len(val_imgs) + len(train_imgs)
                 eval_data["Model"]["iTestImgCount"] = 0 
                 eval_data["Model"]["iTotalClasses"] = len(classes)
                 eval_data["Model"]["iValidationImgCount"] = 0
                 eval_data["Model"]["name"] = model_name
                 # Don't update SolutionDir as requested
                 # eval_data["Model"]["SolutionDir"] = str(MODELS_DIR) + os.sep
                 # eval_data["Model"]["ModelDir"] = "Train"
                 
            with open(evaluation_json_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            logger.info(f"Updated {evaluation_json_path}")
        except Exception as e:
            logger.error(f"Failed to update Evaluation.json: {e}")
            # Don't return False here? Evaluation is secondary to Training config? 
            # Better to return False if it fails to ensure consistency.
            return False

    return True


def create_new_model_version():
    """
    Creates a new Model_X directory and associated configurations.
    Triggered when a new dataset is uploaded.
    """
    # Get current latest model (e.g., Model_1)
    current_model = get_latest_model_name()
    current_num = 1
    if current_model.startswith("Model_"):
        try:
            current_num = int(current_model.split("_")[1])
        except ValueError:
            pass
            
    # Calculate next model
    next_num = current_num + 1
    next_model = f"Model_{next_num}"
    
    # Get paths for curr and next
    curr_paths = get_model_paths(current_model)
    next_paths = get_model_paths(next_model)  # This will os.makedirs internally
    
    # We should copy config JSONs from current to next to preserve structure
    for key in ["training_json", "evaluation_json"]:
        if os.path.exists(curr_paths[key]):
            try:
                with open(curr_paths[key], 'r') as f:
                    data = json.load(f)
                if "Model" in data:
                    data["Model"]["name"] = next_model
                with open(next_paths[key], 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to copy {key} to {next_model}: {e}")
                
    if os.path.exists(curr_paths["testing_json"]):
        try:
            with open(curr_paths["testing_json"], 'r') as f:
                data = json.load(f)
            if "Model" in data:
                data["Model"]["name"] = next_model
            with open(next_paths["testing_json"], 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to copy testing_json to {next_model}: {e}")
            
    # Create empty Stopbuffer texts if they don't exist
    for f in [os.path.join(next_paths["train_dir"], "StopbufferTrain.txt"),
              os.path.join(next_paths["train_dir"], "StopbufferTest.txt"),
              os.path.join(next_paths["test_dir"], "StopbufferTest.txt")]:
        try:
            with open(f, 'w') as fh:
                fh.write("1")
        except Exception as e:
            logger.error(f"Failed to create stop buffer file {f}: {e}")
            
    logger.info(f"Successfully created new model version: {next_model}")
    return next_model


# --- DL Process Wrapper Integration ---

def generate_config_json(config_path, mode="Train", params=None):
    """Updates the existing JSON configuration file for DLProcessWrapper."""
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return False
        
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return False

    # Update Parameters from Training Service calls
    if params and "Model" in config_data:
        if "epochs" in params:
            config_data["Model"]["epochs"] = int(params["epochs"])
        if "batch_size" in params:
            config_data["Model"]["iBatchSize"] = int(params["batch_size"])
        if "lr" in params:
            config_data["Model"]["fBaseLR"] = float(params["lr"])
            
    # Write Config back
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Updated config file at: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write config file: {e}")
        return False

def parse_status_log(log_path):
    """
    Parses the Status.txt file for training progress.
    Returns the last parsed epoch info.
    """
    if not os.path.exists(log_path):
        return None

    last_info = None
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # [2025/12/17 10:25:12]epoch: 0, val_loss: 2.1864, train_loss: 1.8939, best_epoch: 0, best_loss: 2.1864, val_acc: 28.04%
                match = re.search(r"epoch:\s*(\d+),\s*val_loss:\s*([\d.]+),\s*train_loss:\s*([\d.]+),\s*best_epoch:\s*(\d+),\s*best_loss:\s*([\d.]+),\s*val_acc:\s*([\d.]+?)%", line)
                if match:
                    last_info = {
                        "epoch": int(match.group(1)),
                        "val_loss": float(match.group(2)),
                        "train_loss": float(match.group(3)),
                        "best_epoch": int(match.group(4)),
                        "best_loss": float(match.group(5)),
                        "val_acc": float(match.group(6))
                    }
    except Exception as e:
        logger.error(f"Error parsing status log: {e}")
        
    return last_info

def parse_confusion_matrix_file(file_path, ok_classes=None):
    """
    Parses ConfMatrixTest.txt or ConfMatrixEval.txt.
    Calculates Miss Rate and Overkill Rate using provided ok_classes as good 
    and other classes (except Unknown) as not good (NG).
    Ignores Unknown class in calculations.
    
    Args:
        file_path (str): Path to the confusion matrix file.
        ok_classes (list): List of class names considered "OK" (Good). 
                           If None, defaults to ["OK"].
    """
    if not os.path.exists(file_path):
        logger.warning(f"Confusion matrix file not found: {file_path}")
        return None

    if ok_classes is None:
        ok_classes = ["OK"]
        
    # Normalize ok_classes for easier comparison
    ok_classes_norm = [c.lower() for c in ok_classes]

    results = {}
    total_metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # --- 1. Calculate Miss/Overkill from Matrix Table ---
        miss_count = 0
        total_defects = 0
        overkill_count = 0
        total_ok = 0
        
        matrix_start_idx = -1
        for i, line in enumerate(lines):
            if "Confusion Matrix :" in line:
                matrix_start_idx = i
                break
        
        if matrix_start_idx != -1 and matrix_start_idx + 1 < len(lines):
            # Parse Header
            header_line = lines[matrix_start_idx + 1]
            headers = [h.strip() for h in header_line.split() if h.strip()]
            
            # Map headers to indices
            header_map = {h.lower(): idx for idx, h in enumerate(headers)}
            
            # Identify which columns correspond to OK classes
            ok_col_indices = []
            for h, idx in header_map.items():
                if h in ok_classes_norm:
                    ok_col_indices.append(idx)
            
            unknown_col_idx = header_map.get("unknown", -1)
            
            # Parse Rows
            current_idx = matrix_start_idx + 2
            while current_idx < len(lines):
                line = lines[current_idx].strip()
                if not line or "," in line: # End of matrix or start of CSV section
                    break
                    
                parts = [p.strip() for p in line.split() if p.strip()]
                if not parts:
                    current_idx += 1
                    continue
                    
                row_label = parts[0]
                # Parse counts - skip label
                try:
                    counts = [int(x) for x in parts[1:]]
                except ValueError:
                    logger.debug(f"Skipping line due to non-integer counts: {line}")
                    current_idx += 1 # unexpected format
                    continue

                # Ignore Unknown Row
                if row_label.lower() == "unknown":
                    current_idx += 1
                    continue
                    
                # Calculate row totals excluding Unknown column predictions
                row_total = sum(counts)
                unknown_pred_count = 0
                if unknown_col_idx != -1 and unknown_col_idx < len(counts):
                    unknown_pred_count = counts[unknown_col_idx]
                
                valid_row_total = row_total - unknown_pred_count
                
                # Determine if this row is an OK class or NG class
                is_ok_row = row_label.lower() in ok_classes_norm
                
                # Calculate how many predictions were "OK"
                pred_ok_total = 0
                for idx in ok_col_indices:
                    if idx < len(counts):
                        pred_ok_total += counts[idx]
                        
                if is_ok_row:
                    # Valid Good Class (OK)
                    # Overkill = Good predicted as Defect (Predicted != OK and != Unknown)
                    # So, Overkill = Valid Total - Predicted OK
                    row_overkill = valid_row_total - pred_ok_total
                    
                    overkill_count += row_overkill
                    total_ok += valid_row_total
                else:
                    # Valid Defect Class (NG)
                    # Miss = Defect predicted as OK
                    miss_count += pred_ok_total
                    total_defects += valid_row_total
                    
                current_idx += 1

        # Calculate Rates
        miss_rate = (miss_count / total_defects * 100.0) if total_defects > 0 else 0.0
        overkill_rate = (overkill_count / total_ok * 100.0) if total_ok > 0 else 0.0

        # --- 2. Parse CSV-like Section ---
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    row_label = parts[0].strip()
                    
                    if row_label.lower() == "unknown":
                        continue
                        
                    import math
                    def safe_float(v):
                        try:
                            f = float(v)
                            return 0.0 if math.isnan(f) or math.isinf(f) else f
                        except:
                            return 0.0
                            
                    metrics = {
                        "total": int(parts[1]),
                        "correct": int(parts[2]),
                        "incorrect": int(parts[3]),
                        "accuracy": safe_float(parts[4]),
                        "error_rate": safe_float(parts[5])
                    }
                    if row_label.lower() == "sum":
                        total_metrics = metrics
                        # Inject calculated rates
                        total_metrics["miss_rate"] = miss_rate
                        total_metrics["overkill_rate"] = overkill_rate
                    else:
                        results[row_label] = metrics
                except ValueError:
                    continue 

    except Exception as e:
        logger.error(f"Error parsing confusion matrix file: {e}")
        return None
        
    return {"class_metrics": results, "total_metrics": total_metrics}

def run_dl_process_wrapper(config_path, mode="Train"):
    """
    Executes the DLProcessWrapper.exe with the given config and mode.
    Modes: "Train", "Evaluation", "Test"
    """
    exe_path = DL_PROCESS_WRAPPER_PATH
    if not os.path.exists(exe_path):
        logger.error(f"DLProcessWrapper.exe not found at {exe_path}. Please check config.py or DL_PROCESS_WRAPPER_PATH env var.")
        return False

    cmd = [exe_path, config_path, mode]
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # We use Popen to run it and wait for completion.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # This will block until the process terminates.
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"DLProcessWrapper failed with code {process.returncode}")
            logger.error(f"Stderr: {stderr.decode('utf-8')}")
            return False
            
        logger.info(f"DLProcessWrapper finished successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to run DLProcessWrapper: {e}")
        return False

def poll_for_status(file_path, timeout=600):
    """
    Polls the Status.txt file until "SUCCESS", "TENSORRT", or "End Learning" is found, or timeout.
    Returns the parsed status if successful, {"status": "error", "message": ...} if error found, None otherwise.
    """
    start_time = time.time()
    logger.info(f"Polling {file_path} for completion status...")
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    if "SUCCESS" in content or "TENSORRT" in content or "End Learning" in content:
                        logger.info("Training process reported SUCCESS/TENSORRT/End Learning.")
                        return parse_status_log(file_path)
                        
                    content_upper = content.upper()
                    if "ERROR" in content_upper or "EXCEPTION" in content_upper or "FAIL" in content_upper:
                        logger.error(f"Error detected in Status.txt: {content.strip()}")
                        return {"status": "error", "message": content.strip()}
                     
            except Exception as e:
                pass # Retry on read error
        
        time.sleep(2)
        
    logger.warning(f"Timeout waiting for training completion in {file_path}")
    return None

def poll_for_file(file_path, timeout=300):
    """
    Polls until the file exists and is not empty.
    Returns True if found, False on timeout.
    """
    start_time = time.time()
    logger.info(f"Polling for file: {file_path}")
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) > 0:
                    logger.info(f"File found and ready: {file_path}")
                    return True
            except:
                pass
        time.sleep(2)
        
    logger.warning(f"Timeout waiting for file: {file_path}")
    return False

def run_automated_training(full_epochs=1, custom_params=None, custom_dataset_path=None):
    """
    Replaced Run Automated Training to use DLProcessWrapper.
    """
    logger.info("Starting Automated Training via DLProcessWrapper")
    
    # --- Step 0: Scan Dataset and Update JSONs ---
    # This ensures Training.json and Testing.json have correct file lists and class info
    if not scan_dataset_and_update_configs():
         return {"status": "error", "message": "Failed to scan dataset and update configurations."}
    
    paths_dict = get_model_paths()
    training_json = paths_dict["training_json"]
    eval_json = paths_dict["evaluation_json"]
    status_file = paths_dict["status_file"]
    model_name = paths_dict["model_name"]  # Added to fix "model_name is not defined" error
    
    # 0. Clean up stale files to ensure we don't read old data
    try:
        if os.path.exists(status_file):
            os.remove(status_file)
        
        # Cleanup eval/test matrices if possible (path guessing)
        # We can't easily guess them all without parsing configs, but we can try basic locations if needed.
        # For now, deleting Status.txt is the most critical for Training completion check.
    except Exception as e:
        logger.warning(f"Failed to clean up stale files: {e}")

    # Remove auto-calculation of num_classes as requested by user ("don't change")
    
    # --- Optuna Hyperparameter Tuning ---
    if custom_params and len(custom_params) > 0:
        logger.info(f"Using provided custom hyperparameters: {custom_params}")
        best_params = custom_params
        if 'epochs' not in best_params:
            best_params['epochs'] = full_epochs
    else:
        logger.info("No hyperparameters provided. Starting Optuna tuning...")
        # attempt to run optuna tuning
        try:
            import optuna
            
            def objective(trial):
                trial_lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                trial_batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
                trial_params = {"lr": trial_lr, "batch_size": trial_batch_size, "epochs": full_epochs}
                
                logger.info(f"Optuna Trial {trial.number}: Testing params {trial_params}")
                
                if not generate_config_json(training_json, mode="Train", params=trial_params):
                    raise optuna.exceptions.TrialPruned("Failed to generate training configuration.")
                
                if os.path.exists(status_file):
                    try: os.remove(status_file)
                    except: pass
                    
                # Ensure StopbufferTrain.txt and StopbufferTest.txt exist before Training
                train_dir = os.path.dirname(training_json)
                try:
                    with open(os.path.join(train_dir, "StopbufferTrain.txt"), 'w') as f:
                        f.write("1")
                    with open(os.path.join(train_dir, "StopbufferTest.txt"), 'w') as f:
                        f.write("1")
                except Exception as e:
                    logger.warning(f"Failed to create Stopbuffer files in Train dir: {e}")
                
                logger.info(f"Launching Training Process for Trial {trial.number}...")
                if not run_dl_process_wrapper(training_json, "Train"):
                     raise optuna.exceptions.TrialPruned("DLProcessWrapper execution failed.")
                     
                status = poll_for_status(status_file, timeout=60000)
                
                if status and isinstance(status, dict) and status.get("status") == "error":
                     logger.warning(f"Trial {trial.number}: detected error: {status.get('message')}")
                     raise optuna.exceptions.TrialPruned(f"Training failed: {status.get('message')}")
                     
                if not status:
                    status = parse_status_log(status_file)
                    
                if status and "val_loss" in status:
                    logger.info(f"Trial {trial.number} finished with val_loss: {status['val_loss']}")
                    return status["val_loss"]
                elif status and "best_loss" in status:
                    logger.info(f"Trial {trial.number} finished with best_loss: {status['best_loss']}")
                    return status["best_loss"]
                else:
                    logger.warning(f"Trial {trial.number}: Could not read loss from Status.txt. Pruning.")
                    raise optuna.exceptions.TrialPruned("Could not read loss from Status.txt")
            
            optuna.logging.set_verbosity(optuna.logging.INFO)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=3)
            
            best_params = study.best_params
            best_params["epochs"] = full_epochs
            logger.info(f"Optuna tuning completed. Best params found: {best_params}")
            
        except ImportError:
            logger.warning("Optuna is not installed. Using default parameters.")
            best_params = {"epochs": full_epochs}
        except Exception as e:
            logger.error(f"Optuna tuning encountered an error: {e}. Falling back to defaults.")
            best_params = {"epochs": full_epochs}
            
    # Update Training.json with the best parameters before final training
    logger.info(f"Generating final Training.json with params: {best_params}")
    if not generate_config_json(training_json, mode="Train", params=best_params):
         return {"status": "error", "message": "Failed to generate Training.json"}

    # --- identify OK classes from training.json for metrics calculation ---
    ok_classes = ["OK"] # default fallback
    try:
        with open(training_json, 'r') as f:
            t_data = json.load(f)
            class_lst = t_data.get("classlst", [])
            class_bin = t_data.get("classBin", [])
            
            if class_lst and class_bin and len(class_lst) == len(class_bin):
                found_ok = []
                for i, c_item in enumerate(class_lst):
                    c_name = c_item.get("ClassName")
                    if i < len(class_bin):
                         bin_name = class_bin[i].get("classBinName")
                         if bin_name == "OK" and c_name:
                             found_ok.append(c_name)
                
                if found_ok:
                    ok_classes = found_ok
                    logger.info(f"identified OK classes from config: {ok_classes}")
    except Exception as e:
        logger.warning(f"Failed to parse OK classes from training.json: {e}")

    # 3. Run FINAL Training
    if os.path.exists(status_file):
        try: os.remove(status_file)
        except: pass

    # Ensure StopbufferTrain.txt and StopbufferTest.txt exist before Training
    train_dir = os.path.dirname(training_json)
    try:
        with open(os.path.join(train_dir, "StopbufferTrain.txt"), 'w') as f:
            f.write("1")
        with open(os.path.join(train_dir, "StopbufferTest.txt"), 'w') as f:
            f.write("1")
    except Exception as e:
        logger.warning(f"Failed to create Stopbuffer files in Train dir: {e}")

    logger.info("Launching Final Training Process with best parameters...")
    if not run_dl_process_wrapper(training_json, "Train"):
        logger.error("Final Training process failed execution (exit code).")
        return {"status": "error", "message": "Training process failed execution"}
    
    # 4. Poll for Training Completion
    # We poll Status.txt for "SUCCESS" or "TENSORRT"
    last_status = poll_for_status(status_file, timeout=60000) # Long timeout for training
    
    if last_status and isinstance(last_status, dict) and last_status.get("status") == "error":
        logger.error(f"Final training failed: {last_status.get('message')}")
        return last_status
        
    if not last_status:
        # Fallback: maybe it finished but didn't write SUCCESS? 
        # But user said "wait till SUCCESS or TENSORRT comes".
        # We will log a warning but try to proceed if we see *some* status, or abort?
        # User implies strict wait.
        logger.warning("Training completion marker not found. Proceeding with caution or check Status.txt content manually.")
        last_status = parse_status_log(status_file) # Try to get whatever is there
    
    logger.info(f"Training Completed. Last Status: {last_status}")

    # 5. Run Evaluation -- ONLY after training success
    
    # Prepare Evaluation Config
    # Update Evaluation.json IN PLACE to ensure it points to the correct model
    if os.path.exists(eval_json):
        try:
            with open(eval_json, 'r') as f:
                e_data = json.load(f)
            
            # Update Model Paths only
            if "Model" in e_data:
                # Don't update SolutionDir as requested
                # e_data["Model"]["SolutionDir"] = str(MODELS_DIR) + os.sep
                # e_data["Model"]["ModelDir"] = "Train"
                # Keep other fields as is, respecting user's "different format"
                pass
                
            with open(eval_json, 'w') as f:
                json.dump(e_data, f, indent=2)
            logger.info(f"Updated {eval_json} for Training flow.")
        except Exception as e:
             logger.warning(f"Failed to update Evaluation.json in training flow: {e}")
    
    # shutil.copy(training_json, eval_json)

    # Cleanup old Eval Matrix if exists
    eval_matrix_path = os.path.join(os.path.dirname(training_json), "ConfMatrixEvaluation.txt")
    if os.path.exists(eval_matrix_path):
        try: os.remove(eval_matrix_path)
        except: pass
    
    # Ensure StopbufferTest.txt exists before Evaluation
    stop_buffer_path = os.path.join(os.path.dirname(training_json), "StopbufferTest.txt")
    try:
        with open(stop_buffer_path, 'w') as f:
            f.write("1") # Ensure it has content
    except Exception as e:
        logger.warning(f"Failed to create StopbufferTest.txt: {e}")

    logger.info("Launching Evaluation Process...")
    if not run_dl_process_wrapper(eval_json, "Evaluate"):
        logger.error("Evaluation process failed execution. Aborting Testing.")
        return {
            "status": "error", 
            "message": "Evaluation process failed execution",
            "last_training_status": last_status
        }
    
    # Poll for Eval Matrix
    if not poll_for_file(eval_matrix_path, timeout=600):
         logger.error("Timeout waiting for Evaluation results.")
    
    # 6. Parse Eval Results
    eval_results = parse_confusion_matrix_file(eval_matrix_path, ok_classes=ok_classes)
    
    # 7. Run Testing - ONLY after Eval success
    # Test directory comes dynamically from get_model_paths()
    test_dir = paths_dict["test_dir"]
    os.makedirs(test_dir, exist_ok=True)

    
    real_test_json_path = paths_dict["testing_json"]
    
    # Copy training config to this new location for Testing -> NO, User says different format.
    # Update Testing.json IN PLACE
    if os.path.exists(real_test_json_path):
        try:
            with open(real_test_json_path, 'r') as f:
                t_data = json.load(f)
            
            if "Model" in t_data:
                # Don't update SolutionDir as requested
                # t_data["Model"]["SolutionDir"] = str(MODELS_DIR) + os.sep
                # t_data["Model"]["ModelDir"] = "Train"
                pass
                
            with open(real_test_json_path, 'w') as f:
                json.dump(t_data, f, indent=2)
            logger.info(f"Updated {real_test_json_path} for Training flow.")
        except Exception as e:
            logger.warning(f"Failed to update Testing.json in training flow: {e}")
            
    # shutil.copy(training_json, real_test_json_path)
    
    # Cleanup old Test Matrix if exists
    test_matrix_path = os.path.join(test_dir, "ConfMatrixTest.txt")
    if os.path.exists(test_matrix_path):
        try: os.remove(test_matrix_path)
        except: pass
    
    # Ensure StopbufferTest.txt exists before Testing
    stop_buffer_test_path = os.path.join(test_dir, "StopbufferTest.txt")
    try:
        with open(stop_buffer_test_path, 'w') as f:
            f.write("1") 
    except Exception as e:
        logger.warning(f"Failed to create StopbufferTest.txt in Test dir: {e}")

    logger.info(f"Launching Testing Process with config at: {real_test_json_path}")
    if not run_dl_process_wrapper(real_test_json_path, "Test"):
         logger.error("Testing process failed execution.")
         return {
            "status": "error", 
            "message": "Testing process failed execution",
            "last_training_status": last_status,
            "eval_results": eval_results
        }
    
    # Poll for Test Matrix
    if not poll_for_file(test_matrix_path, timeout=600):
        logger.error("Timeout waiting for Testing results.")
    
    # 8. Parse Test Results
    test_results = parse_confusion_matrix_file(test_matrix_path, ok_classes=ok_classes)
    
    return {
        "status": "success",
        "last_training_status": last_status,
        "eval_results": eval_results,
        "test_results": test_results,
        "model_name": model_name
    }

def run_inference():
    """
    Runs (Evaluation -> Testing) sequence without Training.
    This corresponds to the User's "Inference" request.
    """
    logger.info("Starting Inference (Evaluation & Testing) via DLProcessWrapper")
    
    # --- Step 0: Scan Dataset and Update JSONs ---
    # Updates Training.json and default Testing.json with current file lists
    # This also populates the file lists in memory if we need them, but 
    # scan_dataset_and_update_configs returns True/False.
    # We need the image lists. Let's re-scan or rely on the JSONs being updated.
    # scan_dataset_and_update_configs updates Testing.json! But we need to update Evaluation.json too.
    if not scan_dataset_and_update_configs():
         return {"status": "error", "message": "Failed to scan dataset and update configurations."}
    
    # We need to know where the model is.
    # Based on User input: ModelDir = "Train", SolutionDir = MODELS_DIR
    model_dir_name = "Train"
    solution_dir = str(MODELS_DIR) + os.sep

    # Get Test Images (we need to populate Evaluation.json with them)
    # scan_dataset_and_update_configs wrote them to Testing.json, maybe we can read from there?
    # Or just re-scan. Let's read from Testing.json since it was just updated.
    
    # Resolve Testing.json path (same logic as scan_dataset)
    target_test_json = TESTING_JSON_PATH
    t_data_for_paths = {}
    
    # Try to find real Testing.json if it exists (it should have been created/updated by scan_dataset)
    # We'll use the one scan_dataset used.
    # But wait, scan_dataset logic for 'real' path depends on Training.json content.
    # Let's assume scan_dataset did its job and updated the Testing.json at target_test_json 
    # OR at the specific Model path.
    # Let's try to locate the most likely Testing.json to get the image list.
    
    test_imgs = []
    classes = []
    
    # Read Training.json to find model name and get context
    training_json = TRAINING_JSON_PATH
    if os.path.exists(training_json):
        try:
            with open(training_json, 'r') as f:
                t_data = json.load(f)
                if "Model" in t_data:
                     # Check if scan_dataset updated SolutionDir/ModelDir in Training.json 
                     # (it does in my previous edit)
                     pass
        except: pass

    # We need the test and val images list. 
    # Let's re-scan test/val dir to be safe and independent.
    paths = get_data_paths()
    test_dir_path = paths['test']
    val_dir_path = paths['val']
    
    # Quick scan for classes (needed for label indexing)
    train_dir = paths['train']
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Helper scan
    def scan_imgs_local(d):
        imgs = []
        if not os.path.exists(d): return imgs
        for c in classes:
            cp = os.path.join(d, c)
            if not os.path.exists(cp): continue
            lbl = class_to_idx.get(c, 0)
            pat = os.path.join(cp, "**", "*")
            files = glob.glob(pat, recursive=True)
            for fp in files:
                if fp.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    imgs.append({
                        "ImageName": os.path.abspath(fp),
                        "Label": lbl,
                        "MaskPath": "",  # Empty as per user snippet
                        "Captions": None,
                        "DatasetId": 1,
                        "CaptionsSentence": None,
                        "TabularInfo": None
                    })
        return imgs

    test_imgs = scan_imgs_local(test_dir_path)
    val_imgs = scan_imgs_local(val_dir_path)
    train_imgs = scan_imgs_local(train_dir)
    logger.info(f"Scanned {len(test_imgs)} test images, {len(val_imgs)} val images, and {len(train_imgs)} train images for inference.")

    # --- 1. Run Evaluation ---
    eval_json = EVALUATION_JSON_PATH
    
    # Update Evaluation.json IN PLACE
    if not os.path.exists(eval_json):
        logger.error(f"Evaluation.json not found at {eval_json}. Cannot run evaluation.")
        return {"status": "error", "message": "Evaluation configuration file missing."}

    try:
        with open(eval_json, 'r') as f:
            e_data = json.load(f)
        
        # Update Model Paths
        if "Model" in e_data:
            # Don't update SolutionDir as requested
            # e_data["Model"]["SolutionDir"] = solution_dir
            # e_data["Model"]["ModelDir"] = model_dir_name # "Train"
            
            # User Request: Update iTrainImgCount instead of iTestImgCount in Evaluation.json
            e_data["Model"]["iTrainImgCount"] = len(val_imgs) + len(train_imgs)
            e_data["Model"]["iTestImgCount"] = 0
            e_data["Model"]["iTotalClasses"] = len(classes)
            e_data["Model"]["iValidationImgCount"] = 0
        
        # Update Image List - User snippet shows Evaluation using 'testImglst'
        e_data["testImglst"] = val_imgs + train_imgs
        e_data["trainImglst"] = []
        e_data["ValImgList"] = []
        # e_data["ValImgList"] = [] # formatting in user snippet had this empty
        
        with open(eval_json, 'w') as f:
            json.dump(e_data, f, indent=2)
        logger.info(f"Updated {eval_json} for Inference.")
        
    except Exception as e:
        logger.error(f"Failed to update Evaluation.json: {e}")
        return {"status": "error", "message": "Failed to update Evaluation configuration."}

    # Cleanup old Eval Matrix
    eval_matrix_path = os.path.join(os.path.dirname(training_json), "ConfMatrixEvaluation.txt")
    if os.path.exists(eval_matrix_path):
        try: os.remove(eval_matrix_path)
        except: pass
    
    # Ensure StopbufferTest.txt exists 
    stop_buffer_path = os.path.join(os.path.dirname(training_json), "StopbufferTest.txt")
    try:
        with open(stop_buffer_path, 'w') as f:
            f.write("1") 
    except Exception as e:
        logger.warning(f"Failed to create StopbufferTest.txt: {e}")

    logger.info("Launching Evaluation Process...")
    if not run_dl_process_wrapper(eval_json, "Evaluate"):
        logger.error("Evaluation process failed execution.")
        return {
            "status": "error", 
            "message": "Evaluation process failed execution"
        }
    
    # Poll for Eval Matrix
    if not poll_for_file(eval_matrix_path, timeout=600):
         logger.error("Timeout waiting for Evaluation results.")
    
    # Parse Eval Results
    # Identify OK classes logic (reused)
    ok_classes = ["OK"] 
    # ... (same logic as before to find ok_classes from Training.json or hardcoded) ...
    # Simplified: check classes list or Training.json
    # We can try to read Training.json again or just rely on default.
    try:
        with open(training_json, 'r') as f:
            t_data = json.load(f)
            class_lst = t_data.get("classlst", [])
            class_bin = t_data.get("classBin", [])
            if class_lst and class_bin:
                found_ok = [c.get("ClassName") for i, c in enumerate(class_lst) 
                            if i < len(class_bin) and class_bin[i].get("classBinName") == "OK"]
                if found_ok: ok_classes = found_ok
    except: pass

    eval_results = parse_confusion_matrix_file(eval_matrix_path, ok_classes=ok_classes)
    
    # --- 2. Run Testing ---
    
    # We update Testing.json IN PLACE
    # Note: scan_dataset_and_update_configs might have already updated a 'Testing.json' 
    # but we need to ensure it's the right one and has the right ModelDir.
    
    # Resolve Real Test JSON Path
    model_name = "Model_1"
    try:
         with open(training_json, 'r') as f:
            t_data = json.load(f)
            if "Model" in t_data:
                model_name = t_data["Model"].get("name", "Model_1")
    except: pass
    
    # Construct path: SolutionDir/Test/ModelName/Testing.json -> This is where Scan Dataset puts it?
    # Actually scan_dataset puts it in `sol_dir/Test/model_name/Testing.json`.
    # Let's use that path.
    test_dir_real = os.path.join(MODELS_DIR, "Test", model_name)
    os.makedirs(test_dir_real, exist_ok=True)
    real_test_json_path = os.path.join(test_dir_real, "Testing.json")
    
    # If it doesn't exist, check default TESTING_JSON_PATH
    if not os.path.exists(real_test_json_path) and os.path.exists(TESTING_JSON_PATH):
        try:
             shutil.copy(TESTING_JSON_PATH, real_test_json_path)
        except: pass
        
    if not os.path.exists(real_test_json_path):
         logger.error("Testing.json not found and could not be created.")
         return {"status": "error", "message": "Testing configuration missing."}
         
    # Update Testing.json
    try:
        with open(real_test_json_path, 'r') as f:
            test_data = json.load(f)
            
        if "Model" in test_data:
            # Don't update SolutionDir as requested
            # test_data["Model"]["SolutionDir"] = solution_dir
            # test_data["Model"]["ModelDir"] = model_dir_name # "Train"
            
            test_data["Model"]["iTestImgCount"] = len(test_imgs)
            test_data["Model"]["iTotalClasses"] = len(classes)
            test_data["Model"]["iTrainImgCount"] = 0
            test_data["Model"]["iValidationImgCount"] = 0
            
        test_data["testImglst"] = test_imgs
        test_data["trainImglst"] = []
        test_data["ValImgList"] = []
        
        with open(real_test_json_path, 'w') as f:
             json.dump(test_data, f, indent=2)
        logger.info(f"Updated {real_test_json_path} for Inference.")
             
    except Exception as e:
         logger.error(f"Failed to update Testing.json: {e}")
         return {"status": "error", "message": "Failed to update Testing configuration."}

    # Cleanup old Test Matrix
    test_matrix_path = os.path.join(test_dir_real, "ConfMatrixTest.txt")
    if os.path.exists(test_matrix_path):
        try: os.remove(test_matrix_path)
        except: pass
    
    # Ensure StopbufferTest.txt in Test Dir
    stop_buffer_test_path = os.path.join(test_dir_real, "StopbufferTest.txt")
    try:
        with open(stop_buffer_test_path, 'w') as f:
            f.write("1") 
    except Exception as e:
        logger.warning(f"Failed to create StopbufferTest.txt in Test dir: {e}")

    logger.info(f"Launching Testing Process with config at: {real_test_json_path}")
    if not run_dl_process_wrapper(real_test_json_path, "Test"):
         logger.error("Testing process failed execution.")
         return {
            "status": "error", 
            "message": "Testing process failed execution",
            "eval_results": eval_results
        }
    
    # Poll for Test Matrix
    if not poll_for_file(test_matrix_path, timeout=600):
        logger.error("Timeout waiting for Testing results.")
    
    # Parse Test Results
    test_results = parse_confusion_matrix_file(test_matrix_path, ok_classes=ok_classes)
    
    return {
        "status": "success",
        "eval_results": eval_results,
        "test_results": test_results,
        "model_name": model_name
    }
