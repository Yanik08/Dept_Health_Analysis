from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from src.data_loader import build_and_save_panels
from src.models import load_model_dataset, train_logistic_regression, train_random_forest, train_xgboost, train_xgboost_with_val
from src.evaluation import evaluate_logit, evaluate_rf, evaluate_xgb, choose_threshold_min_fn, xgb_risk_drivers_report

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

    # Descriptive statistics function
    def describe_merged_panel(df: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # basic dimensions
        n_rows = len(df)
        n_cols = df.shape[1]

        # core identifiers (adapt names if needed)
        cc_col = "country_code"
        year_col = "year"

        n_countries = df[cc_col].dropna().nunique() if cc_col in df.columns else np.nan
        min_year = df[year_col].min() if year_col in df.columns else np.nan
        max_year = df[year_col].max() if year_col in df.columns else np.nan

        # missingness (top 10)
        miss = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
        miss_top10 = miss.head(10)

        # duplicate country-year check
        dup_ctry_year = (
            df.duplicated(subset=[cc_col, year_col]).sum()
            if (cc_col in df.columns and year_col in df.columns)
            else np.nan
        )

        summary = pd.DataFrame({
            "metric": [
                "rows", "columns", "countries",
                "min_year", "max_year",
                "duplicate_country_year_rows"
            ],
            "value": [
                n_rows, n_cols, n_countries,
                min_year, max_year,
                dup_ctry_year
            ]
        })

        # save files for report
        summary.to_csv(out_path.with_suffix(".csv"), index=False)
        miss_top10.to_frame("missing_%").to_csv(out_path.parent / (out_path.stem + "_missing_top10.csv"))

        # console print (so it appears when you run main)
        print("\n --- Merged Panel Overview (descriptive statistics)---")
        print(summary.to_string(index=False))
        print("\nTop 10 missingness (%):")
        print(miss_top10.to_string())

    # call it right after you create/load the merged df
    describe_merged_panel(merged_df, project_root / "results" / "merged_panel_overview")



    chrono_dir = results_root / "chrono"
    random_dir = results_root / "random"
    chrono_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    print("Targets in df:", [c for c in df.columns if c.startswith("crisis_h")]) # added for debug, had issues before, handy to see targets

    ### ----------------Chronological split model training + evaluation ----------------

    print("\n[CHRONO] Training + evaluating every model by horizon (3, 5, 10)...")
    # ---------------------- LOGIT: horizons 3/5/10 ----------------------
    print("\n[3] LOGIT: Training + evaluating for horizons 3, 5, 10...")
    logit_h_model: dict[int, object] = {}

    for h in HORIZONS:
        print("------------------------------------------------------------")
        print(f"    [LOGIT] Horizon h={h}")
        out_dir_h = chrono_dir / f"logit_h{h}"
        out_dir_h.mkdir(parents=True, exist_ok=True)

        logit_model_h, X_test_h, y_test_h = train_logistic_regression(
            df, target=f"crisis_h{h}", split_method="chronological"
        )

        acc_h, auc_h, cm_h = evaluate_logit(logit_model_h, X_test_h, y_test_h, out_dir_h)
        print(f"    LOGIT h={h} | accuracy={acc_h:.3f} | roc_auc={auc_h:.3f}")
        print("    Confusion matrix:")
        print(cm_h)

        model_path_h = out_dir_h / f"logit_model_h{h}.joblib"
        joblib.dump(logit_model_h, model_path_h)
        print(f"    Saved LOGIT model to: {model_path_h}")

        logit_h_model[h] = logit_model_h

    print("    All LOGIT models trained + evaluated.\n")


    # ---------------------- RF: horizons 3/5/10 ----------------------
    print("\n[6] RF: Training + evaluating for horizons 3, 5, 10...")
    rf_h_model: dict[int, object] = {}

    for h in HORIZONS:
        print("------------------------------------------------------------")
        print(f"    [RF] Horizon h={h}")
        out_dir_h = chrono_dir / f"rf_h{h}"
        out_dir_h.mkdir(parents=True, exist_ok=True)

        rf_model_h, rf_X_test_h, rf_y_test_h = train_random_forest(
            df, target=f"crisis_h{h}", split_method="chronological"
        )

        acc_h, auc_h, cm_h = evaluate_rf(rf_model_h, rf_X_test_h, rf_y_test_h, out_dir_h)
        print(f"    RF h={h} | accuracy={acc_h:.3f} | roc_auc={auc_h:.3f}")
        print("    Confusion matrix:")
        print(cm_h)

        model_path_h = out_dir_h / f"rf_model_h{h}.joblib"
        joblib.dump(rf_model_h, model_path_h)
        print(f"    Saved RF model to: {model_path_h}")

        rf_h_model[h] = rf_model_h

    print("    All RF models trained + evaluated.\n")


    # ---------------------- XGB: horizons 3/5/10 (VAL threshold) ----------------------
    print("\n[9] XGB: Training + evaluating for horizons 3, 5, 10 (chrono + val threshold)...")

    xgb_h_model: dict[int, object] = {}
    xgb_h_metrics: dict[int, tuple[float, float]] = {}  # (accuracy, roc_auc)
    xgb_h_thr_pess: dict[int, float] = {}               # val-tuned threshold (min FN) per horizon

    for h in HORIZONS:
        print("------------------------------------------------------------")
        print(f"    [XGB] Horizon h={h}")
        out_dir_h = chrono_dir / f"xgb_h{h}"
        out_dir_h.mkdir(parents=True, exist_ok=True)

        m_h, X_val_h, y_val_h, X_test_h, y_test_h = train_xgboost_with_val(
            df, target=f"crisis_h{h}", split_method="chronological"
        )

        probas_val_h = m_h.predict_proba(X_val_h)[:, 1]
        thr_h = choose_threshold_min_fn(y_val_h, probas_val_h)
        xgb_h_thr_pess[h] = float(thr_h)

        acc_h, auc_h, cm_h = evaluate_xgb(m_h, X_test_h, y_test_h, out_dir_h, threshold=xgb_h_thr_pess[h])
        xgb_h_metrics[h] = (acc_h, auc_h)

        # --- XGB interpretability output for report (SHAP) ---
        # Uses the same X_test_h that you evaluate on (so columns match the model)
        xgb_drivers_h = xgb_risk_drivers_report(
            model=m_h,
            X=X_test_h,
            results_dir=out_dir_h,
            variable_labels=Variable_Labels,
            top_k=12,
            prefix=f"xgb_h{h}",
            make_beeswarm=True
        )

        print("\nTop XGBoost risk drivers (SHAP) for horizon", h)
        print(xgb_drivers_h.to_string(index=False))

        print(f"    XGB h={h} | accuracy={acc_h:.3f} | roc_auc={auc_h:.3f} | thr_pess={xgb_h_thr_pess[h]:.2f}")
        print("    Confusion matrix:")
        print(cm_h)

        model_path_h = out_dir_h / f"xgb_model_h{h}.joblib"
        joblib.dump(m_h, model_path_h)
        print(f"    Saved XGB model to: {model_path_h}")

        xgb_h_model[h] = m_h

    print("\n[CHRONO] Done. \n")

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
        elif any(p > 0.10 for p in probs.values()):
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