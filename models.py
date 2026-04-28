"""
models.py
Train and evaluate ML models with MLflow tracking.
Usage:
  py models.py --dataset adult --model lr --C 1.0
  py models.py --dataset adult --model dt --max_depth 5
  py models.py --dataset adult --model svm --C 0.5 --kernel rbf
  py models.py --dataset adult --model knn --n_neighbors 7
"""

import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils import load_dataset, preprocess


# ─────────────────────────────────────────────
# Fairness metrics
# ─────────────────────────────────────────────

def fairness_metrics(y_true, y_pred, protected):
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    protected = np.array(protected)

    priv_rate   = y_pred[protected == 1].mean()
    unpriv_rate = y_pred[protected == 0].mean()

    # TPR per group
    def tpr(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # FPR per group
    def fpr(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Precision per group
    def precision(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    dpd = float(priv_rate - unpriv_rate)
    dpr = float(unpriv_rate / priv_rate) if priv_rate > 0 else float("nan")
    eod = float(max(abs(tpr(1) - tpr(0)), abs(fpr(1) - fpr(0))))
    eqop = float(tpr(1) - tpr(0))
    ppd = float(precision(1) - precision(0))

    acc_priv   = accuracy_score(y_true[protected == 1], y_pred[protected == 1])
    acc_unpriv = accuracy_score(y_true[protected == 0], y_pred[protected == 0])

    return {
        "accuracy_overall":              float(accuracy_score(y_true, y_pred)),
        "accuracy_privileged":           float(acc_priv),
        "accuracy_unprivileged":         float(acc_unpriv),
        "privileged_positive_rate":      float(priv_rate),
        "unprivileged_positive_rate":    float(unpriv_rate),
        "privileged_predicted_positive": int(y_pred[protected == 1].sum()),
        "unprivileged_predicted_positive": int(y_pred[protected == 0].sum()),
        "privileged_total":              int((protected == 1).sum()),
        "unprivileged_total":            int((protected == 0).sum()),
        "demographic_parity_difference": dpd,
        "demographic_parity_ratio":      dpr,
        "equalized_odds_difference":     eod,
        "equal_opportunity_difference":  eqop,
        "predictive_parity_difference":  ppd,
    }


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(model_name, args):
    if model_name == "lr":
        return LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=42
        )
    elif model_name == "dt":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
    elif model_name == "svm":
        return SVC(
            C=args.C,
            kernel=args.kernel,
            random_state=42
        )
    elif model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["adult", "compas", "german"])
    parser.add_argument("--model",   required=True, choices=["lr", "dt", "svm", "knn"])

    # LR / SVM
    parser.add_argument("--C",               type=float, default=1.0)
    parser.add_argument("--max_iter",        type=int,   default=1000)
    parser.add_argument("--kernel",          type=str,   default="rbf")

    # DT
    parser.add_argument("--max_depth",       type=int,   default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)

    # kNN
    parser.add_argument("--n_neighbors",     type=int,   default=5)
    parser.add_argument("--weights",         type=str,   default="uniform")

    args = parser.parse_args()

    # Load data
    X, y, protected, attr = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test, p_train, p_test = preprocess(X, y, protected)

    # Build model
    model = build_model(args.model, args)

    # MLflow run
    mlflow.set_experiment(f"bias_{args.dataset}")

    with mlflow.start_run(run_name=f"{args.model}_{args.dataset}"):

        # Log params
        mlflow.log_param("dataset",   args.dataset)
        mlflow.log_param("model",     args.model)
        mlflow.log_param("protected", attr)

        if args.model in ["lr", "svm"]:
            mlflow.log_param("C",      args.C)
        if args.model == "svm":
            mlflow.log_param("kernel", args.kernel)
        if args.model == "dt":
            mlflow.log_param("max_depth",         args.max_depth)
            mlflow.log_param("min_samples_split", args.min_samples_split)
        if args.model == "knn":
            mlflow.log_param("n_neighbors", args.n_neighbors)
            mlflow.log_param("weights",     args.weights)

        # Train
        print(f"\nTraining {args.model.upper()} on {args.dataset.upper()}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = fairness_metrics(y_test, y_pred, p_test)

        # Log metrics to MLflow
        for k, v in metrics.items():
            if isinstance(v, float) and not np.isnan(v):
                mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, f"{args.model}_model")

        # Print results
        print(f"\n{'='*55}")
        print(f"  {args.dataset.upper()} | {args.model.upper()}")
        print(f"{'='*55}")
        print(f"  Accuracy (overall):       {metrics['accuracy_overall']:.4f}")
        print(f"  Accuracy (privileged):    {metrics['accuracy_privileged']:.4f}")
        print(f"  Accuracy (unprivileged):  {metrics['accuracy_unprivileged']:.4f}")
        print(f"  ---")
        print(f"  Predictions breakdown:")
        print(f"    Privileged:   {metrics['privileged_predicted_positive']:4d} / {metrics['privileged_total']:4d}  ({metrics['privileged_positive_rate']:.2%})")
        print(f"    Unprivileged: {metrics['unprivileged_predicted_positive']:4d} / {metrics['unprivileged_total']:4d}  ({metrics['unprivileged_positive_rate']:.2%})")
        print(f"  ---")
        print(f"  Demographic Parity Diff:  {metrics['demographic_parity_difference']:+.4f}  (ideal: 0)")
        print(f"  Demographic Parity Ratio: {metrics['demographic_parity_ratio']:.4f}   (ideal: 1, >0.8 ok)")
        print(f"  Equalized Odds Diff:      {metrics['equalized_odds_difference']:+.4f}  (ideal: 0)")
        print(f"  Equal Opportunity Diff:   {metrics['equal_opportunity_difference']:+.4f}  (ideal: 0)")
        print(f"  Predictive Parity Diff:   {metrics['predictive_parity_difference']:+.4f}  (ideal: 0)")
        print(f"\n  MLflow run logged.")

if __name__ == "__main__":
    main()