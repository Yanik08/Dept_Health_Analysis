"""Model evaluation and visualization."""
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def choose_threshold_min_fn(y_val, proba_val) -> float:
    """
    Choose the threshold that maximizes recall (minimizes false negatives) on validation.
    If multiple thresholds give the same recall, pick the highest one (fewer false positives).
    """
    best_thr = 0.01
    best_recall = -1.0

    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (proba_val >= thr).astype(int)
        rec = recall_score(y_val, y_pred, zero_division=0)

        # maximize recall; tie-breaker: higher thr (less FP)
        if (rec > best_recall) or (rec == best_recall and thr > best_thr):
            best_recall = rec
            best_thr = float(thr)

    print(f"    Threshold chosen (min FN / max recall): {best_thr:.2f} | recall={best_recall:.3f}")
    return best_thr

##### Evaluate Logistic Regression model #####
def evaluate_logit(model, X_test, y_test, results_dir: Path | None = None):
    """
    Evaluate the logistic regression model and save results.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred) # accuracy calculation
    roc_auc = roc_auc_score(y_test, y_proba) # ROC AUC calculation
    cm = confusion_matrix(y_test, y_pred)  # confusion matrix calculation
    # Extra metrics for imbalanced crises
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()

    print(f"Precision (Logit): {precision:.3f}")
    print(f"Recall    (Logit): {recall:.3f}")
    print(f"F1        (Logit): {f1:.3f}")
    print(f"Bal Acc   (Logit): {bal_acc:.3f}")
    print(f"PR AUC    (Logit): {pr_auc:.3f}")
    print(f"Accuracy (Logit): {accuracy:.3f}") # accuracy printout
    print(f"ROC AUC (Logit): {roc_auc:.3f}") # ROC AUC printout
    print("Confusion Matrix:") # confusion matrix header printout
    print(cm) # confusion matrix printout

    # Plot ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test) # plot ROC curve
    plt.title("ROC Curve") # title for ROC curve plot

    if results_dir is not None: # if results_dir is provided, save results
        results_dir.mkdir(parents=True, exist_ok=True) # ensure results directory exists

        # Save ROC curve figure
        roc_path = results_dir / "logit_roc_curve.png" # path for ROC curve figure
        plt.savefig(roc_path, bbox_inches="tight") # save ROC curve figure
        print(f"Saved ROC curve to: {roc_path}") # printout of saved ROC curve path

        #Save confusion matrix as plot
        plt.figure(figsize=(6, 4)) # create new figure for confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") # plot confusion matrix heatmap
        plt.title("Confusion Matrix (Logitic Regression)") # title for confusion matrix plot
        plt.tight_layout() # adjust layout
        plt.savefig(results_dir / "logit_confusion_matrix.png", bbox_inches="tight") # save confusion matrix figure


        # Save metrics
        metrics_df = pd.DataFrame(
            {
                "metric": ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp"],
                "value":  [accuracy,  roc_auc,  pr_auc,  precision,  recall,  f1,  bal_acc,             tn,   fp,   fn,   tp],
            }
        )

        metrics_path = results_dir / "logit_metrics.csv" # path for metrics CSV
        metrics_df.to_csv(metrics_path, index=False) # save metrics to CSV without index
        print(f"Saved metrics to: {metrics_path}") # printout of saved metrics path 

        # Save predictions (index aligned with X_test / y_test)
        preds_df = pd.DataFrame(
            {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,}  # predictions dataframe
        )
        preds_path = results_dir / "logit_predictions.csv" # path for predictions CSV
        preds_df.to_csv(preds_path, index=False) # save predictions to CSV without index
        print(f"Saved predictions to: {preds_path}") # printout of saved predictions path

    # Show the ROC figure in interactive runs
    plt.show()
    return accuracy, roc_auc, cm

##### Evaluate Random Forest model ##### literally copy-paste from above with minor changes

def evaluate_rf(model, X_test, y_test, results_dir: Path | None = None):
    """
    Evaluate the Random Forest model and save results.
    """

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()

    print(f"Precision (RF): {precision:.3f}")
    print(f"Recall    (RF): {recall:.3f}")
    print(f"F1        (RF): {f1:.3f}")
    print(f"Bal Acc   (RF): {bal_acc:.3f}")
    print(f"PR AUC    (RF): {pr_auc:.3f}")
    print(f"Accuracy (RF): {accuracy:.3f}")
    print(f"ROC AUC (RF): {roc_auc:.3f}")
    print("Confusion Matrix (RF):")
    print(cm)

    # Plot ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve (Random Forest)")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save ROC curve figure
        roc_path = results_dir / "rf_roc_curve.png"
        plt.savefig(roc_path, bbox_inches="tight")
        print(f"Saved RF ROC curve to: {roc_path}")

        # Save confusion matrix as plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Random Forest)")
        plt.tight_layout()
        plt.savefig(results_dir / "rf_confusion_matrix.png", bbox_inches="tight")
        print(f"Saved RF confusion matrix to: {results_dir / 'rf_confusion_matrix.png'}")

        # Save metrics
        metrics_df = pd.DataFrame(
            {
                "metric": ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp"],
                "value":  [accuracy,  roc_auc,  pr_auc,  precision,  recall,  f1,  bal_acc,             tn,   fp,   fn,   tp],
            }
        )

        metrics_path = results_dir / "rf_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved RF metrics to: {metrics_path}")

        # Save predictions (index aligned with X_test / y_test)
        preds_df = pd.DataFrame(
            {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}
        )
        preds_path = results_dir / "rf_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"Saved RF predictions to: {preds_path}")

    plt.show()

    return accuracy, roc_auc, cm

##### Evaluate XGBoost model ##### Again, similar to above

def evaluate_xgb(model, X_test, y_test, results_dir: Path | None = None, threshold: float = 0.5):
    """
    Evaluate the XGBoost model and optionally save results.

    threshold:
        probability cutoff for predicting class 1 (crisis). Default 0.5.
        Lower it (e.g., 0.2) to catch more crises (higher recall).
    """
    # Probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    # Apply threshold to get class predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)  # IMPORTANT: AUC uses probabilities
    ap = average_precision_score(y_test, y_proba)  # PR-AUC (good for imbalanced)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Threshold (XGB): {threshold:.2f}")
    print(f"Accuracy  (XGB): {accuracy:.3f}")
    print(f"ROC AUC   (XGB): {roc_auc:.3f}")
    print(f"PR AUC/AP (XGB): {ap:.3f}")
    print(f"Precision (XGB): {precision:.3f}")
    print(f"Recall    (XGB): {recall:.3f}")
    print(f"F1        (XGB): {f1:.3f}")
    print(f"Bal Acc   (XGB): {bal_acc:.3f}")
    print("Confusion Matrix (XGB):")
    print(cm)

    # Plot ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve (XGBoost)")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save ROC curve
        roc_path = results_dir / "xgb_roc_curve.png"
        plt.savefig(roc_path, bbox_inches="tight")
        print(f"Saved XGB ROC curve to: {roc_path}")

        # Save confusion matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (XGBoost) thr={threshold:.2f}")
        plt.tight_layout()
        cm_path = results_dir / "xgb_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        print(f"Saved XGB confusion matrix to: {cm_path}")

        # Save metrics
        metrics_df = pd.DataFrame(
            {
                "metric": ["threshold", "accuracy", "roc_auc", "pr_auc_ap", "precision", "recall", "f1", "balanced_accuracy"],
                "value": [threshold, accuracy, roc_auc, ap, precision, recall, f1, bal_acc],
            }
        )
        metrics_path = results_dir / "xgb_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved XGB metrics to: {metrics_path}")

        # Save predictions
        preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba})
        preds_path = results_dir / "xgb_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"Saved XGB predictions to: {preds_path}")

    plt.show()
    return accuracy, roc_auc, cm

def xgb_risk_drivers_report(
    model,
    X: pd.DataFrame,
    results_dir: Path,
    variable_labels: Optional[Dict[str, str]] = None,
    top_k: int = 12,
    prefix: str = "xgb",
    make_beeswarm: bool = True,
) -> pd.DataFrame:
    """
    Generate a report of top_k risk drivers based on SHAP values from the XGBoost model.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Compute SHAP values (robust across shap versions) ---
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_vals = explainer.shap_values(X)

    # Binary classification may return list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = np.array(shap_vals)

    # Create clean table of top_k risk drivers based on mean shap in absolute value
    importance = np.abs(shap_vals).mean(axis=0)         
    # sign of mean SHAP to indicate direction
    mean_shap = shap_vals.mean(axis=0)      

    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean_abs_shap": importance,
        "mean_shap": mean_shap,
    }).sort_values("importance_mean_abs_shap", ascending=False)

    out = out.head(top_k).copy()
    out["direction_hint"] = np.where(out["mean_shap"] > 0, "↑ risk", "↓ risk")

    if variable_labels is not None:
        out["label"] = out["feature"].map(variable_labels).fillna(out["feature"])
    else:
        out["label"] = out["feature"]

    tidy = out[["label", "feature", "importance_mean_abs_shap", "direction_hint"]].copy()

    # Save table
    tidy.to_csv(results_dir / f"{prefix}_risk_drivers.csv", index=False)

    # Plot 1: bar plot
    plot_df = tidy.sort_values("importance_mean_abs_shap", ascending=True) # for horizontal bar plot

    plt.figure()
    plt.barh(plot_df["label"], plot_df["importance_mean_abs_shap"])
    plt.xlabel("Mean(|SHAP|) importance")
    plt.title("XGBoost: top macro risk drivers")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_risk_drivers_bar.png", dpi=200)
    plt.close()

    # Plot 2 : SHAP beeswarm
    if make_beeswarm:
        # This creates a matplotlib figure internally
        shap.summary_plot(shap_vals, X, show=False, max_display=top_k)
        plt.tight_layout()
        plt.savefig(results_dir / f"{prefix}_risk_drivers_beeswarm.png", dpi=200)
        plt.close()

    return tidy