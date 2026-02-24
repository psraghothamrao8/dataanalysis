import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import zipfile
import io
import os
import threading
import json

# Minimal Imports
from app.config import BASE_DIR, MODELS_DIR, get_data_paths, DATASET_ROOT
from app.services.agent_service import analyze_situation_and_decide
from app.services.data_service import apply_fix, get_dataset_stats
from app.services.training_service import run_automated_training, run_inference, create_new_model_version

app = FastAPI(title="AutoML Agent (n8n)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dataset
paths = get_data_paths()
app.mount("/dataset", StaticFiles(directory=paths["dataset_root"]), name="dataset")

# --- State ---
training_state = {"status": "idle", "result": None, "error": None}
state_lock = threading.Lock()

# --- Models ---
class FixRequest(BaseModel):
    file_path: str
    action: str
    new_label: Optional[str] = None

class FixItem(BaseModel):
    file_path: str
    issue_type: str
    suggested_label: Optional[str] = None
    split: Optional[str] = None

class BatchFixRequest(BaseModel):
    status: Optional[str] = None
    total_issues_found: Optional[int] = None
    all_issues: List[FixItem]

# NEW: Model for Training Request
class TrainingRequest(BaseModel):
    dataset_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class UploadPathRequest(BaseModel):
    file_path: str
    ok_classes: Optional[List[str]] = None
    ng_classes: Optional[List[str]] = None


@app.post("/api/train")
def start_train(req: Optional[TrainingRequest] = None):
    """
    Step 3: Train model.
    Optional JSON Body:
    {
      "dataset_path": "/path/to/custom/dataset",
      "hyperparameters": { "lr": 0.005, "batch_size": 16, "epochs": 50 }
    }
    """
    # 1. Check if already running
    if training_state["status"] == "running": 
        raise HTTPException(400, "Already running")
    
    # Extract params if provided
    dataset_path = req.dataset_path if req else None
    custom_params = req.hyperparameters if req else None

    # 2. Update state to running
    with state_lock: 
        training_state.update({"status": "running", "result": None, "error": None})

    try:
        # 3. RUN TRAINING
        res = run_automated_training(
            custom_params=custom_params,
            custom_dataset_path=dataset_path
        )
        
        # 4. Success
        with state_lock: 
            training_state.update({"status": "completed", "result": res})
        
        return {"status": "completed", "result": res}

    except Exception as e:
        # 5. Failure
        with state_lock: 
            training_state.update({"status": "failed", "error": str(e)})
        raise HTTPException(500, detail=f"Training failed: {str(e)}")

@app.post("/api/inference")
def inference():
    """Runs inference on the validation/test set."""
    # 1. Check if already running (Train or Inference)
    if training_state["status"] == "running": 
        raise HTTPException(status_code=400, detail="A process (Training or Inference) is already running")
    
    # 2. Update state to running
    with state_lock: 
        training_state.update({"status": "running", "result": None, "error": None})

    try:
        result = run_inference()
        
        # 3. Handle Result
        if result["status"] == "error":
             with state_lock:
                 training_state.update({"status": "failed", "error": result["message"]})
             raise HTTPException(status_code=500, detail=result["message"]) # 500 or 404? 500 implies execution failure.
        
        with state_lock:
             training_state.update({"status": "completed", "result": result})
             
        return result

    except Exception as e:
        with state_lock:
            training_state.update({"status": "failed", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/upload")
def upload(req: UploadPathRequest):
    """
    Load dataset from a local file path (Zip only).
    Optionally accepts ok_classes and ng_classes to define class mapping.
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    
    if not req.file_path.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    try:
        if os.path.exists(DATASET_ROOT):
            shutil.rmtree(DATASET_ROOT)
        os.makedirs(DATASET_ROOT)

        with zipfile.ZipFile(req.file_path, 'r') as z:
            z.extractall(DATASET_ROOT)
            
        # --- Save Class Mapping Metadata ---
        if req.ok_classes or req.ng_classes:
            meta_path = os.path.join(DATASET_ROOT, "dataset_meta.json")
            meta_data = {
                "ok_classes": req.ok_classes or [],
                "ng_classes": req.ng_classes or []
            }
            try:
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2)
            except Exception as e:
                print(f"Failed to save dataset metadata: {e}") # Non-critical failure

        # --- Create New Model Version Folder ---
        new_model_name = create_new_model_version()

        return {
            "status": "success", 
            "message": f"Dataset extracted from {req.file_path}. Created {new_model_name}",
            "files_extracted": len(z.namelist()),
            "meta_saved": bool(req.ok_classes or req.ng_classes),
            "new_model_name": new_model_name
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The provided file is not a valid zip archive.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/api/download_csv")
def download_csv(file_path: str):
    """
    Downloads a CSV file given its file path.
    Useful for retrieving the 'misclassifications.csv' generated during training/inference.
    
    Usage: GET /api/download_csv?file_path=/absolute/path/to/logs/file.csv
    """
    # 1. Basic validation
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # 2. Ensure it is a CSV (Security best practice)
    if not file_path.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files can be downloaded via this endpoint.")
        
    # 3. Return the file as a downloadable attachment
    return FileResponse(
        path=file_path, 
        filename=os.path.basename(file_path), 
        media_type='text/csv'
    )
    

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)