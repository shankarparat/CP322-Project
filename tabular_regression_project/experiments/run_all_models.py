"""
Run All Models Script
Runs XGBoost, TabNet, FT-Transformer on both datasets, then SHAP analysis.
"""

from pathlib import Path
import subprocess
import sys

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
METRICS_DIR = ROOT / "artifacts" / "metrics"
RESULTS_DIR = ROOT / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_xgboost():
    print("\n" + "=" * 60)
    print("Running XGBoost + Linear Regression Training...")
    print("=" * 60)
    xgboost_script = SRC_DIR / "train_xgboost.py"
    subprocess.run([sys.executable, str(xgboost_script)], check=True, cwd=str(SRC_DIR))


def run_tabnet():
    print("\n" + "=" * 60)
    print("Running TabNet Training...")
    print("=" * 60)
    tabnet_script = SRC_DIR / "train_tabnet.py"
    subprocess.run([sys.executable, str(tabnet_script)], check=True, cwd=str(SRC_DIR))


def run_fttransformer():
    print("\n" + "=" * 60)
    print("Running FT-Transformer Training...")
    print("=" * 60)
    fttransformer_script = SRC_DIR / "train_fttransformer.py"
    subprocess.run([sys.executable, str(fttransformer_script)], check=True, cwd=str(SRC_DIR))


def run_shap():
    print("\n" + "=" * 60)
    print("Running SHAP Analysis...")
    print("=" * 60)
    shap_script = SRC_DIR / "shap_analysis.py"
    subprocess.run([sys.executable, str(shap_script)], check=True, cwd=str(SRC_DIR))


def merge_results():
    print("\n" + "=" * 60)
    print("Merging Results...")
    print("=" * 60)
    
    xgboost_path = METRICS_DIR / "xgboost_results.csv"
    tabnet_path = METRICS_DIR / "tabnet_results.csv"
    fttransformer_path = METRICS_DIR / "fttransformer_results.csv"
    
    results = []
    
    # Load XGBoost results (wide format)
    if xgboost_path.exists():
        df_xgb = pd.read_csv(xgboost_path)
        print(f"  Loaded: {xgboost_path}")
        for _, row in df_xgb.iterrows():
            results.append({"Model": row["Model"], "Dataset": "house", "RMSE": row["House_RMSE"], "R2": row["House_R2"]})
            results.append({"Model": row["Model"], "Dataset": "energy", "RMSE": row["Energy_RMSE"], "R2": row["Energy_R2"]})
    
    # Load TabNet results (long format)
    if tabnet_path.exists():
        df_tabnet = pd.read_csv(tabnet_path)
        print(f"  Loaded: {tabnet_path}")
        for _, row in df_tabnet.iterrows():
            results.append({"Model": row["Model"], "Dataset": row["Dataset"], "RMSE": row["RMSE"], "R2": row["R2"]})
    
    # Load FT-Transformer results (long format)
    if fttransformer_path.exists():
        df_ftt = pd.read_csv(fttransformer_path)
        print(f"  Loaded: {fttransformer_path}")
        for _, row in df_ftt.iterrows():
            results.append({"Model": row["Model"], "Dataset": row["Dataset"], "RMSE": row["RMSE"], "R2": row["R2"]})
    
    df_combined = pd.DataFrame(results)
    
    combined_path_1 = METRICS_DIR / "final_results.csv"
    combined_path_2 = RESULTS_DIR / "final_results.csv"
    
    df_combined.to_csv(combined_path_1, index=False)
    df_combined.to_csv(combined_path_2, index=False)
    
    print(f"\n  Combined results saved to:")
    print(f"    - {combined_path_1}")
    print(f"    - {combined_path_2}")
    
    return df_combined


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "RUN ALL MODELS" + " " * 18 + "#")
    print("#" * 60)
    
    # Train all models
    run_xgboost()
    run_tabnet()
    run_fttransformer()
    
    # Run SHAP analysis (on XGBoost models)
    run_shap()
    
    # Merge all results
    df_final = merge_results()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(df_final.to_string(index=False))
    
    print("\n" + "#" * 60)
    print("#" + " " * 16 + "ALL MODELS COMPLETE" + " " * 15 + "#")
    print("#" * 60)


if __name__ == "__main__":
    main()
