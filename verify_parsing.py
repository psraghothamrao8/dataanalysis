import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.services.training_service import parse_confusion_matrix_file

def create_dummy_matrix(filename):
    content = """
Confusion Matrix :
   OK    NG   UNKNOWN
   90    10    5
   5     95    2
   0     0     100
   
OK, 105, 90, 15, 0.85, 0.15
NG, 102, 95, 7, 0.93, 0.07
UNKNOWN, 100, 0, 100, 0.0, 1.0
Sum, 307, 185, 122, 0.60, 0.40
    """
    with open(filename, "w") as f:
        f.write(content)

def test_parse():
    filename = "dummy_matrix.txt"
    create_dummy_matrix(filename)
    
    print("Testing with OK=['OK']...")
    results = parse_confusion_matrix_file(filename, ok_classes=["OK"])
    
    # Expected:
    # OK row: 90 OK, 10 NG, 5 Unknown. Valid Total = 90+10=100. Overkill = 100 - 90 = 10.
    # NG row: 5 OK, 95 NG, 2 Unknown. Valid Total = 5+95=100. Miss = 5.
    
    # Total OK = 100.
    # Total Defects = 100.
    
    # Overkill Rate = 10/100 = 10%
    # Miss Rate = 5/100 = 5%
    
    miss_rate = results['total_metrics']['miss_rate']
    overkill_rate = results['total_metrics']['overkill_rate']
    
    print(f"Miss Rate: {miss_rate}% (Expected 5.0%)")
    print(f"Overkill Rate: {overkill_rate}% (Expected 10.0%)")
    
    assert abs(miss_rate - 5.0) < 0.001, f"Miss Rate correct {miss_rate}"
    assert abs(overkill_rate - 10.0) < 0.001, f"Overkill Rate incorrect {overkill_rate}"
    
    print("\nTesting with OK=['NG'] (Inverted logic just for test)...")
    # If NG is the "OK class":
    # NG row (now treated as OK/Good): 5 OK(pred), 95 NG(pred). 
    #   "OK" col index is now the NG column (index 1).
    #   Predicted "Good" count = 95. Valid Total = 100. 
    #   Overkill = 100 - 95 = 5.
    
    # OK row (now treated as Defect): 90 OK(pred), 10 NG(pred).
    #   Predicted "Good" count = 10. Valid Total = 100.
    #   Miss = 10.
    
    results_inverted = parse_confusion_matrix_file(filename, ok_classes=["NG"])
    
    miss_rate_inv = results_inverted['total_metrics']['miss_rate'] # 10/100 = 10%
    overkill_rate_inv = results_inverted['total_metrics']['overkill_rate'] # 5/100 = 5%
    
    print(f"Miss Rate (Inverted): {miss_rate_inv}% (Expected 10.0%)")
    print(f"Overkill Rate (Inverted): {overkill_rate_inv}% (Expected 5.0%)")
    
    assert abs(miss_rate_inv - 10.0) < 0.001
    assert abs(overkill_rate_inv - 5.0) < 0.001
    
    print("\nAll tests passed!")
    
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_parse()
