from pathlib import Path
import joblib
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression, train_random_forest, train_xgboost
from src.evaluation import evaluate_logit, evaluate_rf, evaluate_xgb


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
                        
                        #### Logistic Regression Model #####
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

    # --------------------- 5. Save Logistic Regression model ---------------------
    model_path = results_dir / "logit_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[5] Saved trained Logistic Regression model to: {model_path}")

                        #### Random Forest Model #####
    # --------------------- 6. Train Random Forest -------------------------------
    print("\n[6] Training Random Forest model...")
    rf_model, X_test_rf, y_test_rf = train_random_forest(df)
    print("    Random Forest model trained.")

    # --------------------- 7. Evaluate & save RF results ------------------------
    rf_accuracy, rf_roc_auc, rf_cm = evaluate_rf(rf_model, X_test_rf, y_test_rf, results_dir)
    print(f"    RF Test accuracy: {rf_accuracy:.3f}")
    print(f"    RF Test ROC AUC: {rf_roc_auc:.3f}")
    print("    RF Confusion Matrix:")
    print(rf_cm)

    # --------------------- 8. Save Random Forest model --------------------------
    rf_model_path = results_dir / "rf_model.joblib"
    joblib.dump(rf_model, rf_model_path)
    print(f"    Saved RF model to: {rf_model_path}")


                        #### XGBoost Model #####
    # --------------------- 9. Train XGBoost -------------------------------
    print("\n[8] Training XGBoost model...")
    xgb_model, X_test_xgb, y_test_xgb = train_xgboost(df)
    print("    XGBoost model trained.")

    # --------------------- 10. Evaluate & save XGBoost results ---------------------
    print("\n[9] Evaluating XGBoost model...")
    xgb_accuracy, xgb_roc_auc, xgb_cm = evaluate_xgb(
        xgb_model,
        X_test_xgb,
        y_test_xgb,
        results_dir)
    print(f"    XGB Test accuracy: {xgb_accuracy:.3f}")
    print(f"    XGB Test ROC AUC: {xgb_roc_auc:.3f}")
    print("   XGB Confusion Matrix:")
    print(xgb_cm)

    # --------------------- 11. Save XGBoost model --------------------------
    xgb_model_path = results_dir / "xgb_model.joblib"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"    Saved XGB model to: {xgb_model_path}")
    
    
    print("        \U0001F60E main.py finished successfully \U0001F918\U0001F9A7")
if __name__ == "__main__":
    main()