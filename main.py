from pathlib import Path
from xml.parsers.expat import model
import pandas as pd
import joblib
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression, train_random_forest, train_xgboost, train_xgboost_with_val
from src.evaluation import evaluate_logit, evaluate_rf, evaluate_xgb, choose_threshold_min_fn, choose_threshold_min_fn

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

SPLITS = ["chronological", "random"]
HORIZONS = [3, 5, 10]  # Prediction horizons in years

def main() -> None:
    project_root = Path(__file__).resolve().parent

    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    chrono_dir = results_root / "chrono"
    random_dir = results_root / "random"
    chrono_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

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

    # Results directories (chrono = main, random = robustness) 
    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    chrono_dir = results_root / "chrono"
    random_dir = results_root / "random"
    chrono_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    print("Targets in df:", [c for c in df.columns if c.startswith("crisis_h")]) # added for debug, had issues before, handy to see targets

                        #### Logistic Regression Model #####
    # --------------------- 3. Train logistic regression ---------------------
    print("\n[3] Training Logistic Regression model for within-3-years crisis...")
    logit_model, X_test, y_test = train_logistic_regression(df, target="crisis_h3", split_method="chronological")
    print("    Model trained.")

    # --------------------- 4. Evaluate & save results ---------------------
    print("\n[4] Evaluating Logistic Regression model...")
    accuracy, roc_auc, cm = evaluate_logit(logit_model, X_test, y_test, chrono_dir)
    print(f"    Test accuracy: {accuracy:.3f}")
    print(f"    Test ROC AUC: {roc_auc:.3f}")
    print("    Confusion Matrix:")
    print(cm)

    # --------------------- 5. Save Logistic Regression model ---------------------
    logit_model_path = chrono_dir / "logit_model.joblib"
    joblib.dump(logit_model, logit_model_path)
    print(f"\n[5] Saved trained Logistic Regression model to: {logit_model_path}")

                        #### Random Forest Model #####
    # --------------------- 6. Train Random Forest -------------------------------
    print("\n[6] Training Random Forest model for within-3-years crisis...")
    rf_model, rf_X_test, rf_y_test = train_random_forest(df, target="crisis_h3", split_method="chronological")

    # --------------------- 7. Evaluate & save RF results ------------------------
    print("\n[7] Evaluating Random Forest model...")
    rf_accuracy, rf_roc_auc, rf_cm = evaluate_rf(rf_model, rf_X_test, rf_y_test, chrono_dir)
    print(f"    RF Test accuracy: {rf_accuracy:.3f}")
    print(f"    RF Test ROC AUC:  {rf_roc_auc:.3f}")
    print("    RF Confusion Matrix:")
    print(rf_cm)

    rf_model_path = chrono_dir / "rf_model.joblib"
    joblib.dump(rf_model, rf_model_path)
    print(f"    Saved RF model to: {rf_model_path}")

    # --------------------- 8. Train RF for horizons 3/5/10 ----------------------
    print("\n[8] Training Random Forest early-warning models for horizons 3, 5, 10...")
    rf_h_models: dict[int, object] = {}

    for h in HORIZONS:
        print(f"    Training Random Forest for within-{h}-years horizon...")
        model_h, _, _ = train_random_forest(df, target=f"crisis_h{h}", split_method="chronological")
        rf_h_models[h] = model_h


                      #### XGBoost Model #####
    # --------------------- 9. Train XGBoost -------------------------------
    print("\n[9] Training XGBoost model for within-3-years crisis...")
    xgb_model, xgb_X_test, xgb_y_test = train_xgboost(
        df, target="crisis_h3", split_method="chronological"
    )
    print("    XGBoost model trained.")

    # --------------------- 10. Evaluate & save XGBoost results ---------------------
    print("\n[10] Evaluating XGBoost model...")
    xgb_accuracy, xgb_roc_auc, xgb_cm = evaluate_xgb(
        xgb_model, xgb_X_test, xgb_y_test, chrono_dir, threshold=0.20
    )

    print(f"    XGB Test accuracy: {xgb_accuracy:.3f}")
    print(f"    XGB Test ROC AUC:  {xgb_roc_auc:.3f}")
    print("    XGB Confusion Matrix:")
    print(xgb_cm)

    xgb_model_path = chrono_dir / "xgb_model.joblib"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"    Saved XGB model to: {xgb_model_path}")
    # ----------------------11. Train XGB for horizons 3, 5, 10---------------- 
    print("\n[11] Training + evaluating XGBoost models for horizons 3, 5, 10...")

    # store everything you need
    xgb_h_model: dict[int, object] = {}
    xgb_h_metrics: dict[int, tuple[float, float]] = {}  # (accuracy, roc_auc)
    xgb_h_thr_pess: dict[int, float] = {}

    for h in [3, 5, 10]:
        print(f"    Training XGBoost for within-{h}-years horizon...")

        # Train
        m_h, X_val_h, y_val_h, X_test_h, y_test_h = train_xgboost_with_val(
            df, target=f"crisis_h{h}", split_method="chronological"
        )

        xgb_h_model[h] = m_h

        probas_val_h = m_h.predict_proba(X_val_h)[:, 1]
        thr_h = choose_threshold_min_fn(y_val_h, probas_val_h)
        xgb_h_thr_pess[h] = thr_h      

        # Evaluate + save in a horizon-specific folder (no overwriting)
        out_dir_h = chrono_dir / f"xgb_h{h}"
        acc_h, auc_h, cm_h = evaluate_xgb(
            m_h, X_test_h, y_test_h, out_dir_h, threshold=thr_h
        )
        xgb_h_metrics[h] = (acc_h, auc_h)

        # Save model
        model_path_h = out_dir_h / f"xgb_model_h{h}.joblib"
        joblib.dump(m_h, model_path_h)
        print(f"    Saved XGB model to: {model_path_h}")

    print("    All XGBoost models trained + evaluated.")

                            #### Predicton model #####
    # ------------------- 12. Custom scenario prediction ---------------------
    
    # originally, i had used another split method, the chronological split is more realistic
    # but for robustness testing, I will also do random split below
    print("\n[ROBUSTNESS] Training + evaluating models with RANDOM split (within-3)...")

    # Logit robustness
    logit_r, X_test_r, y_test_r = train_logistic_regression(df, target="crisis_h3", split_method="random")
    evaluate_logit(logit_r, X_test_r, y_test_r, random_dir)
    joblib.dump(logit_r, random_dir / "logit_model.joblib")

    # RF robustness
    rf_r, rf_X_test_r, rf_y_test_r = train_random_forest(df, target="crisis_h3", split_method="random")
    evaluate_rf(rf_r, rf_X_test_r, rf_y_test_r, random_dir)
    joblib.dump(rf_r, random_dir / "rf_model.joblib")

    # XGB robustness
    xgb_r, xgb_X_test_r, xgb_y_test_r = train_xgboost(df, target="crisis_h3", split_method="random")
    evaluate_xgb(xgb_r, xgb_X_test_r, xgb_y_test_r, random_dir)
    joblib.dump(xgb_r, random_dir / "xgb_model.joblib")
    
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
        
        probs: dict[int, float] = {}
        crisis_flags_opt: list[int] = []
        crisis_flags_pess: list[int] = []
        OPTIMISTIC_THR = 0.50
        
        for h in HORIZONS:
            model_h = xgb_h_model[h]
            p = float(model_h.predict_proba(X_new)[0, 1])
            probs[h] = p

            thr_pess = float(xgb_h_thr_pess[h])

            crisis_opt = int(p >= OPTIMISTIC_THR)
            crisis_pess = int(p >= thr_pess)

            crisis_flags_opt.append(crisis_opt)
            crisis_flags_pess.append(crisis_pess)

            print(
                f" Within {h} years: p={p:.3f} | "
                f"optimistic={crisis_opt} (thr={OPTIMISTIC_THR:.2f}) | "
                f"pessimistic={crisis_pess} (thr={thr_pess:.2f})"
            )

        print("------------------------------------------------------------\n")
        print("------------------------------------------------------------")
        print(" Interpretation:")
        print("\nInterpretation:")

        # ---------------- Optimistic (0.50) ----------------
        print("\n[Optimistic rule]")
        if any(crisis_flags_opt):
            print(f"\n\u26A0\uFE0F  CRISIS ALERT (optimistic): Crisis likely within {max(HORIZONS)} years.")
            print("-> Policy Advice: Strengthen fiscal and monetary policies, build reserves, seek IMF support early.")
        elif any(p > 0.30 for p in probs.values()):
            print(f"\n\U0001F6A8  ELEVATED RISK (optimistic): Monitor closely, crisis possible within {max(HORIZONS)} years.")
            print("-> Policy Advice: Enhance surveillance, consider preemptive measures, communicate clearly.")
        else:
            print("\n\U00002705  LOW RISK (optimistic): No crisis signals based on macrovariables.")
            print("-> Policy Advice: Maintain prudent fiscal and monetary policies.")

        # ---------------- Pessimistic (val-tuned) ----------------
        print("\n[Pessimistic rule]")
        if any(crisis_flags_pess):
            print(f"\n\u26A0\uFE0F  CRISIS ALERT (pessimistic): Early-warning flag within {max(HORIZONS)} years.")
            print("-> Policy Advice: Prepare contingency plans, increase buffers, coordinate with international partners.")
        elif any(p > 0.30 for p in probs.values()):
            print(f"\n\U0001F6A8  ELEVATED RISK (pessimistic): Monitor closely, crisis possible within {max(HORIZONS)} years.")
            print("-> Policy Advice: Tighten monitoring, stress-test public finances, consider precautionary funding options.")
        else:
            print("\n\U00002705  LOW RISK (pessimistic): No crisis signals based on macrovariables.")
            print("-> Policy Advice: Stay vigilant; pessimistic rule is stricter but no alarm triggered.")
        print("\n ------------------------------------------------------------\n")
    print("\n\n        \U0001F60E main.py finished successfully \U0001F918\U0001F9A7 \n\n")

if __name__ == "__main__":
    main()