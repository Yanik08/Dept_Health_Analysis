"""Model evaluation and visualization."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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