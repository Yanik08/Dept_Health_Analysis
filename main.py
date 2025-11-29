from pathlib import Path
import joblib
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression, train_random_forest, train_xgboost, get_feature_columns
from src.evaluation import evaluate_logit, evaluate_rf, evaluate_xgb
import pandas as pd


Variable_Labels: dict[str, str] = {
    "BCA_NGDPD": "Current account balance (% of GDP)",
    "GGXONLB_NGDP": "Net lending/borrowing of gov. (% of GDP)",
    "GGXWDG_NGDP": "Government gross debt (% of GDP)",
    "LP": "Labour productivity / output index (LP)",
    "LUR": "Unemployment rate (%)",
    "NGDPD": "Nominal GDP (in billions of USD)",
    "NGDP_RPCH": "Real GDP growth (% change)",
    "PCPIPCH": "Inflation (CPI, % change)",
}

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
    
                            #### Predicton model #####
    # ------------------- 12. Custom scenario prediction ---------------------
    #
    answer = input(
        "\n  Would you like to predict dept situation based on macrovariables?\n    -(Yes/n): "
    ).strip().lower()

    if answer in ["Yes" , "yes" , "YES"]:
        target = "sovereign_crisis"
        feature_cols = get_feature_columns(df, target)

        print("\nEnter the values for the following macroeconomic variables:")
        print("(Use numeric values, e.g. 123.45)\n")

        user_data: dict[str, float] = {} # To store user inputs

        for col in feature_cols: 
            # To make it user-friendly, get variable label if available
            label = Variable_Labels.get(col, col)

            while True:
                # Show friendly label + technical name in brackets
                raw_val = input(f"  {label} [{col}]: ").strip() # Get user input
                try:
                    val = float(raw_val) # Convert to float
                    user_data[col] = val # Store in dictionary
                    break
                except ValueError: # Handle invalid input
                    print("    XXXX Invalid number. Please enter a numeric value.\n")

        # Create a one-row DataFrame in the correct feature order
        X_new = pd.DataFrame([user_data], columns=feature_cols)

        print("\n  Predictions for your custom scenario:\n")

        # Logistic Regression
        logit_proba = model.predict_proba(X_new)[0, 1] # Probability of crisis
        logit_pred = int(logit_proba >= 0.5) # crisis threshold at 0.5

        # Random Forest
        rf_proba = rf_model.predict_proba(X_new)[0, 1] # Probability of crisis
        rf_pred = int(rf_proba >= 0.5) # crisis threshold at 0.5

        # XGBoost
        try:
            xgb_proba = xgb_model.predict_proba(X_new)[0, 1]
            xgb_pred = int(xgb_proba >= 0.5)
            has_xgb = True # Flag to indicate XGB model is available
        except NameError:
            has_xgb = False # XGB model not available
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print(f" Logistic Regression →  p = {logit_proba:.3f}   crisis = {logit_pred}") 
        print(f" Random Forest       →  p = {rf_proba:.3f}   crisis = {rf_pred}") 
        if has_xgb:
            print(f" XGBoost             →  p = {xgb_proba:.3f}   crisis = {xgb_pred}")
        else:
            print(" XGBoost             →  (model not available in this run)") 
        print("------------------------------------------------------------\n")
        print("------------------------------------------------------------")

        print(" Interpretation:") 
        print("  - 'p' is the predicted probability of a sovereign crisis.")
        print("  - 'crisis' = 1 means the model predicts a crisis (p ≥ 0.5).")
        print("  - 'crisis' = 0 means the model predicts no crisis.\n")
    
    print("\n\n        \U0001F60E main.py finished successfully \U0001F918\U0001F9A7 \n\n")

if __name__ == "__main__":
    main()