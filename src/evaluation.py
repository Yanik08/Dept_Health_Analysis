"""Model evaluation and visualization."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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

    print(f"Accuracy: {accuracy:.3f}") # accuracy printout
    print(f"ROC AUC: {roc_auc:.3f}") # ROC AUC printout
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
                "metric": ["accuracy", "roc_auc", "confusion_matrix"], # metric names
                "value": [accuracy, roc_auc, cm], # corresponding metric values
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
                "metric": ["accuracy", "roc_auc"],
                "value": [accuracy, roc_auc],
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
