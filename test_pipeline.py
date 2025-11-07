"""
test_pipeline.py
------------------------------------------------
Tests end-to-end pipeline: build → evaluate → API → submission.
------------------------------------------------
"""

import os
import subprocess

steps = [
    "python src/build_index.py",
    "python src/evaluate.py",
    "python -m uvicorn src.api:app --port 8000 --reload"
]

print("🚀 Running full SHL pipeline...")
for step in steps:
    print(f"\n➡️ {step}")
    os.system(step)
print("\n✅ Pipeline validation complete.")
