import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n==================================================")
    print(f"Running: {os.path.basename(script_path)}")
    print(f"==================================================")
    
    # Run using the same python executable that called this runner
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n[ERROR] {os.path.basename(script_path)} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # Make sure we use absolute paths based on this script's directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    pipeline_steps = [
        os.path.join(root_dir, "src", "train_model.py"),
        os.path.join(root_dir, "src", "tune_xgboost.py"),
        os.path.join(root_dir, "src", "cost_model.py"),
        os.path.join(root_dir, "src", "explain_shap.py")
    ]
    
    for step in pipeline_steps:
        run_script(step)
        
    print("\n==================================================")
    print("Success: Full pipeline completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
