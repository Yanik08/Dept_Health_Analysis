from pathlib import Path
import joblib
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression
from src.evaluation import evaluate_logit


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # --------------------- 1. Build processed data ---------------------
    print("\n[1] Building processed datasets (WEO panel + merged WEO–crisis)...")
    weo_panel, merged_df = build_and_save_panels(project_root)
    print(f"    WEO panel shape:   {weo_panel.shape}")
    print(f"    Merged panel shape:{merged_df.shape}")

    merged_path = project_root / "data" / "processed" / "merged_weo_crisis.csv"

    # --------------------- 2. Load merged dataset ---------------------
    print(f"\n[2] Loading merged dataset from: {merged_path}")
    df = load_model_dataset(merged_path)
    print(f"    Modelling DataFrame shape: {df.shape}")

    # --------------------- 3. Train logistic regression ---------------------
    print("\n[3] Training Logistic Regression model...")
    model, X_test, y_test = train_logistic_regression(df)
    print("    Model trained.")

    # --------------------- 4. Evaluate & save results ---------------------
    print("\n[4] Evaluating Logistic Regression model...")

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    accuracy, roc_auc, cm = evaluate_logit(model, X_test, y_test, results_dir)
    print(f"    Test accuracy: {accuracy:.3f}")
    print(f"    Test ROC AUC: {roc_auc:.3f}")
    print("    Confusion Matrix:")
    print(cm)
    
    # --------------------- 5. Save model ---------------------
    model_path = results_dir / "logit_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[5] Saved trained model to: {model_path}")

    print("\n :) main.py finished successfully :)")


if __name__ == "__main__":
    main()
