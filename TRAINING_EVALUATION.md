# Training Service with Integrated Evaluation

## Summary

The training service has been enhanced to automatically display comprehensive evaluation metrics after training completes, including:

1. **Confusion Matrix** - Displayed in a formatted table
2. **Standard Metrics** - Accuracy, Precision, Recall, F1-Score (per-class)
3. **Miss Rate** - Percentage of actual defects classified as OK
4. **Overkill Rate** - Percentage of actual OK items classified as defects

## Changes Made

### Modified File: `app/services/training_service.py`

#### 1. Added Imports
```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
```

#### 2. New Functions Added

**`get_predictions_and_labels(model, loader)`**
- Collects all predictions and true labels from a DataLoader
- Used for comprehensive evaluation

**`calculate_miss_overkill_rates(y_true, y_pred, class_names, ok_class="OK")`**
- Calculates miss and overkill rates
- **Miss Rate**: (Defects classified as OK) / (Total defects) × 100%
- **Overkill Rate**: (OK items classified as defects) / (Total OK) × 100%
- Treats classification as binary: OK vs NOT-OK (all other classes)

**`print_confusion_matrix(cm, class_names)`**
- Formats and logs confusion matrix in a readable table format

**`evaluate_and_log_metrics(model, val_loader, class_names, ok_class="OK")`**
- Main evaluation function that:
  - Calculates confusion matrix
  - Computes accuracy and per-class metrics
  - Calculates miss/overkill rates if OK class exists
  - Logs everything to the training log

#### 3. Modified `run_automated_training()`

After training completes, the function now:
1. Creates a validation DataLoader
2. Automatically detects the OK class (if present)
3. Calls `evaluate_and_log_metrics()` to display all metrics
4. Returns additional metrics in the result dictionary

## Example Output

When training completes, you'll see output like this in the logs:

```
============================================================
FINAL MODEL EVALUATION
============================================================

============================================================
CONFUSION MATRIX
============================================================
             |       OK |     NONE |   RETEST |
------------------------------------------------------------
          OK |       98 |        8 |        6 |
        NONE |       19 |       44 |        3 |
      RETEST |       26 |        6 |       11 |
============================================================

============================================================
VALIDATION METRICS
============================================================
Accuracy: 66.23%

Per-class Metrics:
          OK: Precision=68.53%, Recall=87.50%, F1=76.86%
        NONE: Precision=75.86%, Recall=66.67%, F1=71.00%
      RETEST: Precision=55.00%, Recall=25.58%, F1=34.92%

============================================================
MISS AND OVERKILL RATES
============================================================
Miss Rate:     41.28% (45/109)
  -> 45 defects incorrectly classified as OK
Overkill Rate: 12.50% (14/112)
  -> 14 OK items incorrectly classified as defects
============================================================

Model saved to: models/best_model.pth
```

## How Miss and Overkill Rates Work

### Definitions

**Miss Rate (Type II Error - False Negative)**
- **What it measures**: How many actual defects slip through as "OK"
- **Formula**: (Defects classified as OK) / (Total actual defects) × 100%
- **From confusion matrix**: Sum of defect rows in the OK column
- **Business impact**: CRITICAL - Defective products reach customers!

**Overkill Rate (Type I Error - False Positive)**
- **What it measures**: How many OK items are rejected as defects
- **Formula**: (OK classified as defects) / (Total actual OK) × 100%
- **From confusion matrix**: Sum of OK row in defect columns
- **Business impact**: COSTLY - Good products wasted unnecessarily

### Example Calculation

Given this confusion matrix:

```
            |   OK | NONE | RETEST
----------------------------------------
OK          |  98 |   8  |   6        <- 8+6=14 Overkill
NONE        |  19 |  44  |   3        <- 19 Miss
RETEST      |  26 |   6  |  11        <- 26 Miss
```

**Miss Rate Calculation:**
- Defects classified as OK: 19 (NONE→OK) + 26 (RETEST→OK) = 45
- Total actual defects: 66 (NONE) + 43 (RETEST) = 109
- **Miss Rate = 45/109 = 41.28%**

**Overkill Rate Calculation:**
- OK classified as defects: 8 (OK→NONE) + 6 (OK→RETEST) = 14
- Total actual OK: 112
- **Overkill Rate = 14/112 = 12.50%**

## Usage

### Normal Training
Just run training as usual via the API or n8n workflow:

```python
result = run_automated_training(full_epochs=300)
```

The evaluation metrics will automatically be logged after training completes.

### Accessing Results
The function now returns additional fields:

```python
{
    "status": "completed",
    "accuracy": 0.6623,
    "params": {"lr": 0.001, "batch_size": 32},
    "confusion_matrix": [[98, 8, 6], [19, 44, 3], [26, 6, 11]],
    "miss_rate": 41.28,
    "overkill_rate": 12.50
}
```

### Viewing Logs
All detailed metrics are logged to `logs/pluto.log`:

```bash
type logs\pluto.log    # Windows
cat logs/pluto.log     # Linux/Mac
```

## OK Class Detection

The system automatically tries to find a class named "OK" (case-insensitive):
- If found: Calculates miss/overkill rates
- If not found: Skips miss/overkill calculation and logs a message

To ensure miss/overkill rates are calculated:
1. Make sure one of your dataset classes is named "OK" (any case: ok, OK, Ok)
2. Or modify the `ok_class` parameter in the code

## Requirements

- scikit-learn (already required for the project)
- No additional dependencies needed!

## Testing

A test script is provided: `test_training_eval.py`

```bash
python test_training_eval.py
```

This runs a quick 5-epoch training to verify the evaluation integration.

## Future Enhancements

Possible improvements:
1. Save confusion matrix as a plot image
2. Export metrics to JSON file
3. Add ROC curves and AUC scores
4. Track metrics across training iterations
5. Add threshold tuning for optimizing miss vs overkill trade-off
