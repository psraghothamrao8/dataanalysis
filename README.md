# AutoML Agent for n8n

An autonomous AI Agent designed to clean, optimize, and train image classification models with minimal human intervention. It exposes a simple API optimized for **n8n workflows**.

## 🚀 Key Features

-   **AutoML Training**: Fully automated training loop (Tune -> Train -> Save).
-   **Smart Optimization**: Uses **Optuna** to find the best Learning Rate and Batch Size (20 trials).
-   **Hybrid Data Cleaning**:
    -   **Training Data**: Validated via Logistic Regression (to avoid bias).
    -   **Validation/Test Data**: Validated via **Weighted Ensemble** (85% Current Best Model + 15% LogReg).
-   **Robustness**: Automatic Early Stopping (Patience 20), Min Epochs (30), and Max Epochs (300).
-   **Dynamic Dataset**: Automatically detects your dataset folder name after upload.

---

## 🛠️ Setup

1.  **Install Dependencies**
    ```bash
    pip install torch torchvision cleanlab optuna fastapi uvicorn scikit-learn ollama
    ```

2.  **Run the Server**
    ```bash
    # Run from the root directory
    python -m app.main
    ```
    *Server will start on `http://0.0.0.0:8000`*

3.  **Ollama (Optional)**
    Ensure [Ollama](https://ollama.com/) is running with `llama3` for the "Analyze" step natural language recommendations.
    ```bash
    ollama run llama3
    ```
    *(If not available, the system falls back to rule-based logic).*

---

## 🔌 API Integration (n8n)

This backend is designed to be the "Brain" of your n8n workflow.

### 1. Analyze Dataset
*Triggers the AI to scan dataset health and recommend actions.*
-   **Endpoint**: `GET /api/analyze`
-   **Response**:
    ```json
    {
      "recommended_action": "data_cleaning",
      "issues_list": [
        {"file_path": "...", "issue_type": "hybrid_label_issue", "suggested_label": "cat"}
      ]
    }
    ```

### 2. Apply Fixes
*Applies the fixes (Delete/Move) approved by you or the agent.*
-   **Endpoint**: `POST /api/batch_fix`
-   **Body**:
    ```json
    {
      "file_paths": ["path/to/image1.jpg", ...],
      "action": "move",  // or "delete"
      "new_label": "cat" // required if action is "move"
    }
    ```

### 3. Start Training
*Launches the background Optuna Search + Training job.*
-   **Endpoint**: `POST /api/train`
-   **Response**: `{"status": "started"}`

### 4. Check Status
*Poll this to check training progress or dataset stats.*
-   **Endpoint**: `GET /api/status`
-   **Response**:
    ```json
    {
      "dataset": {"train": {"count": 1000}, ...},
      "training": {
          "status": "completed",
          "result": {"accuracy": 0.95, "params": {"lr": 0.001}}
      }
    }
    ```

### 5. Upload Dataset (Optional)
*Upload a new `.zip` file to reset the dataset.*
-   **Endpoint**: `POST /api/upload`
-   **Format**: `multipart/form-data`, key=`file`.
-   **Logic**: Automatically extracts and detects the root folder name.

---

## 📂 Project Structure

```
app/
├── main.py                 # The API Entry Point
├── config.py               # Paths & Dynamic Folder Detection
├── logger_config.py        # Centralized Logger
└── services/
    ├── data_service.py     # Hybrid Filtering Logic (Cleanlab + Ensemble)
    ├── training_service.py # Optuna + Training Loop Logic
    └── agent_service.py    # LLM Decision Logic
```
# dataanalysis
