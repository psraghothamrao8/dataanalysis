import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import upload, UploadPathRequest

req = UploadPathRequest(
    file_path=r"c:\Users\rag\Pictures\data_analysis\path2\dataset.zip",
    ok_classes=["OK"],
    ng_classes=["NG", "RETEST", "NONE"]
)

try:
    res = upload(req)
    print("SUCCESS:", json.dumps(res, indent=2))
except Exception as e:
    print("FAILED:", str(e))
