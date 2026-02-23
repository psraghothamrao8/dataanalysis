# app/services/agent_service.py

import os
import csv
import time
from app.config import LOGS_DIR
from app.services.data_service import get_dataset_stats, detect_issues

def save_issues_to_csv(issues):
    """
    Saves the list of detected issues to a CSV file in the LOGS_DIR.
    """
    if not issues:
        return None
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"analysis_issues_{timestamp}.csv"
    csv_path = os.path.join(LOGS_DIR, csv_filename)
    
    try:
        keys = issues[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(issues)
        return csv_path
    except Exception as e:
        print(f"Error saving issues CSV: {e}")
        return None

def analyze_situation_and_decide():
    """
    Gather dataset statistics and potential issues.
    Returns raw data so the orchestration layer (n8n) can decide what to do.
    """
    # 1. Get High-level Stats
    stats = get_dataset_stats()
    
    # 2. Run Deep Analysis
    detection_results = detect_issues()
    issues = detection_results.get("issues", [])
    
    # 3. Save to CSV
    csv_path = save_issues_to_csv(issues)
    
    # 4. Return Context for n8n
    return {
        "status": "analysis_completed",
        "dataset_summary": stats,
        "total_issues_found": len(issues),
        "issues_csv_path": csv_path,  # NEW: Path to the generated CSV
        "all_issues": issues
    }

def diagnose_after_exploration(results):
    """
    Analyzes training results to see if we need to loop back.
    """
    best = results.get("best_result", {})
    val_acc = best.get("val_acc", 0.0)
    
    status = "success" if val_acc > 0.85 else "needs_improvement"
    
    return {
        "diagnosis": status, 
        "best_accuracy": val_acc,
        "details": "Model performed well." if status == "success" else "Model underperforming."
    }