from pathlib import Path
import pandas as pd
import joblib
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression, train_random_forest, train_xgboost
from src.evaluation import evaluate_logit, evaluate_rf, evaluate_xgb

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

HORIZONS = [3, 5, 10]  # Prediction horizons in years

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
    print("Targets in df:", [c for c in df.columns if c.startswith("crisis_h")]) # added for debug, had issues before, handy to see targets

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
                        
                        #### Logistic Regression Model #####
    # --------------------- 3. Train logistic regression ---------------------
    print("\n[3] Training Logistic Regression model for within-3-years crisis...")
    logit_model, X_test, y_test = train_logistic_regression(df, target="crisis_h3")
    print("    Model trained.")

    # --------------------- 4. Evaluate & save results ---------------------
    print("\n[4] Evaluating Logistic Regression model...")
    accuracy, roc_auc, cm = evaluate_logit(logit_model, X_test, y_test, results_dir)
    print(f"    Test accuracy: {accuracy:.3f}")
    print(f"    Test ROC AUC: {roc_auc:.3f}")
    print("    Confusion Matrix:")
    print(cm)

    # --------------------- 5. Save Logistic Regression model ---------------------
    logit_model_path = results_dir / "logit_model.joblib"
    joblib.dump(logit_model, logit_model_path)
    print(f"\n[5] Saved trained Logistic Regression model to: {logit_model_path}")

                        #### Random Forest Model #####
    # --------------------- 6. Train Random Forest -------------------------------
    print("\n[6] Training Random Forest model for within-3-years crisis...")
    rf_model, rf_X_test, rf_y_test = train_random_forest(df, target="crisis_h3")
    print("    Random Forest model trained.")

    # --------------------- 7. Evaluate & save RF results ------------------------
    print("\n[7] Evaluating Random Forest model...")
    rf_accuracy, rf_roc_auc, rf_cm = evaluate_rf(rf_model, rf_X_test, rf_y_test, results_dir)
    print(f"    RF Test accuracy: {rf_accuracy:.3f}")
    print(f"    RF Test ROC AUC:  {rf_roc_auc:.3f}")
    print("    RF Confusion Matrix:")
    print(rf_cm)

    rf_model_path = results_dir / "rf_model.joblib"
    joblib.dump(rf_model, rf_model_path)
    print(f"    Saved RF model to: {rf_model_path}")

    # --------------------- 8. Train RF for horizons 3/5/10 ----------------------
    print("\n[8] Training Random Forest early-warning models for horizons 3, 5, 10...")
    rf_h_models: dict[int, object] = {}

    for h in HORIZONS:
        print(f"    Training Random Forest for within-{h}-years horizon...")
        model_h, _, _ = train_random_forest(df, target=f"crisis_h{h}")
        rf_h_models[h] = model_h


                      #### XGBoost Model #####
    # --------------------- 9. Train XGBoost -------------------------------
    print("\n[9] Training XGBoost model for within-3-years crisis...")
    xgb_model, xgb_X_test, xgb_y_test = train_xgboost(df, target="crisis_h3")
    print("    XGBoost model trained.")

    # --------------------- 10. Evaluate & save XGBoost results ---------------------
    print("\n[10] Evaluating XGBoost model...")
    xgb_accuracy, xgb_roc_auc, xgb_cm = evaluate_xgb(
        xgb_model,
        xgb_X_test,
        xgb_y_test,
        results_dir)
    print(f"    XGB Test accuracy: {xgb_accuracy:.3f}")
    print(f"    XGB Test ROC AUC: {xgb_roc_auc:.3f}")
    print("   XGB Confusion Matrix:")
    print(xgb_cm)

    xgb_model_path = results_dir / "xgb_model.joblib"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"    Saved XGB model to: {xgb_model_path}")

    # ----------------------11. Train XGB for horizons 3, 5, 10---------------- 
    print("\n[11] Training XGBoost models for horizons 3, 5, 10...")    
    xgb_h_model: dict[int, object] = {3: xgb_model}  # Store models for each horizon, starting with h=3

    for h in [5, 10]:
        print(f"    Training XGBoost for within-{h}-years horizon...")
        m_h, _, _ = train_xgboost(df, target=f"crisis_h{h}")
        xgb_h_model[h] = m_h

    print("    All XGBoost models trained.")

                            #### Predicton model #####
    # ------------------- 12. Custom scenario prediction ---------------------

    answer = input(
        "\n  Would you like to predict dept situation based on macrovariables?\n    -(Yes/no): "
    ).strip().lower()

    if answer in ["Yes" , "yes" , "YES", "Yeah", "yeah", "Y", "y"]:
        feature_cols = list(Variable_Labels.keys())  # Use all macroeconomic variables as features

        print("\nEnter the values for the following macroeconomic variables:")
        user_data: dict[str, float] = {}


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

        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("\n Crisis probabilities (within 3, 5, 10 years):\n")
        
        crisis_flags: list[int] = []
        probs: dict[int, float] = {}

        for h in HORIZONS:
            model_h = xgb_h_model[h]
            p = float(model_h.predict_proba(X_new)[0, 1])
            crisis = int(p >= 0.5)
            probs[h] = p
            crisis_flags.append(crisis)
            print(f" Within {h} years: p = {p:.3f} --> crisis = {crisis}")

        print("------------------------------------------------------------\n")
        print("------------------------------------------------------------")
        print(" Interpretation:")

        if any(crisis_flags):
            print(f"\n \u26A0\uFE0F CRISIS ALERT: Crisis likely within {max(HORIZONS)} years.\n")
            print("\n -> Policy Advice: Strengthen fiscal and monetary policies, build reserves, seek IMF support early. \n")
        elif any(p > 0.30 for p in probs.values()):
            print(f"\n \U0001F6A8 ELEVATED RISK: Monitor closely, crisis possible within {max(HORIZONS)} years.\n")
            print("\n -> Policy Advice: Enhance surveillance, consider preemptive measures to bolster economic stability. \n")
        else:
            print("\n \U00002705 LOW RISK: No crisis signals based on macrovariables.\n")
            print("\n -> Policy Advice: Maintain prudent fiscal and monetary policies, continue monitoring economic indicators. \n")

        print("\n ------------------------------------------------------------\n")
    print("\n\n        \U0001F60E main.py finished successfully \U0001F918\U0001F9A7 \n\n")

if __name__ == "__main__":
    main()