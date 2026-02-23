# n8n Integration Guide

This project is fully compatible with **n8n**, allowing you to automate the entire machine learning lifecycle (Analyze -> Fix -> Train) visually.

## 1. Prerequisites
-   [n8n](https://n8n.io/) installed and running (locally or cloud).
-   This Python server running on `http://localhost:8000`.

## 2. Import Workflow
1.  Open your n8n dashboard.
2.  Click **"Workflows"** -> **"Add Workflow"**.
3.  Click the **(...)** menu in the top right -> **"Import from File"**.
4.  Select the `n8n_workflow.json` file from this project.

## 3. Workflow Logic
The imported workflow performs the following steps automatically:

1.  **Analyze Dataset**: Calls `GET /api/analyze` to check for label errors.
2.  **Check Issues**: Uses an `If` node to see if any issues were found.
    -   **If Issues Found**:
        1.  **Prepare Fixes**: Automatically converts the "Suggested Labels" into a batch fix request.
        2.  **Apply Fixes**: Calls `POST /api/batch_fix` to move mislabeled images to their correct folders.
        3.  **Start Training**: Proceeds to training.
    -   **If No Issues**:
        1.  **Start Training**: Directly calls `POST /api/train`.

## 4. Customization
-   **Manual approval**: You can add a "Wait for Approval" node (e.g., Email or Slack button) before the "Apply Fixes" step if you want to review changes first.
-   **Notifications**: Add Slack/Discord nodes to notify you when Training completes.
