"""
mitigation.py
Bias mitigation using:
  - AIF360: Reweighing (pre-processing)
  - Fairlearn: ThresholdOptimizer (post-processing)

Usage:
  py mitigation.py --dataset adult --model lr --C 1.0
  py mitigation.py --dataset compas --model dt --max_depth 5
  py mitigation.py --dataset german --model knn --n_neighbors 7
"""

import argparse
import json
import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# AIF360
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# Fairlearn
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference as fl_dpd

from utils import load_dataset, preprocess


# ─────────────────────────────────────────────
# Fairness metrics (same as models.py)
# ─────────────────────────────────────────────

def fairness_metrics(y_true, y_pred, protected):
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    protected = np.array(protected)

    priv_rate   = y_pred[protected == 1].mean()
    unpriv_rate = y_pred[protected == 0].mean()

    def tpr(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def fpr(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    def precision(group):
        mask = protected == group
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    dpd  = float(priv_rate - unpriv_rate)
    dpr  = float(unpriv_rate / priv_rate) if priv_rate > 0 else float("nan")
    eod  = float(max(abs(tpr(1) - tpr(0)), abs(fpr(1) - fpr(0))))
    eqop = float(tpr(1) - tpr(0))
    ppd  = float(precision(1) - precision(0))

    acc_priv   = accuracy_score(y_true[protected == 1], y_pred[protected == 1])
    acc_unpriv = accuracy_score(y_true[protected == 0], y_pred[protected == 0])

    return {
        "accuracy_overall":               float(accuracy_score(y_true, y_pred)),
        "accuracy_privileged":            float(acc_priv),
        "accuracy_unprivileged":          float(acc_unpriv),
        "privileged_positive_rate":       float(priv_rate),
        "unprivileged_positive_rate":     float(unpriv_rate),
        "privileged_predicted_positive":  int(y_pred[protected == 1].sum()),
        "unprivileged_predicted_positive":int(y_pred[protected == 0].sum()),
        "privileged_total":               int((protected == 1).sum()),
        "unprivileged_total":             int((protected == 0).sum()),
        "demographic_parity_difference":  dpd,
        "demographic_parity_ratio":       dpr,
        "equalized_odds_difference":      eod,
        "equal_opportunity_difference":   eqop,
        "predictive_parity_difference":   ppd,
    }


def print_metrics(metrics, label):
    print(f"\n  [{label}]")
    print(f"  Accuracy (overall):       {metrics['accuracy_overall']:.4f}")
    print(f"  Accuracy (privileged):    {metrics['accuracy_privileged']:.4f}")
    print(f"  Accuracy (unprivileged):  {metrics['accuracy_unprivileged']:.4f}")
    print(f"  Predictions breakdown:")
    print(f"    Privileged:   {metrics['privileged_predicted_positive']:4d} / {metrics['privileged_total']:4d}  ({metrics['privileged_positive_rate']:.2%})")
    print(f"    Unprivileged: {metrics['unprivileged_predicted_positive']:4d} / {metrics['unprivileged_total']:4d}  ({metrics['unprivileged_positive_rate']:.2%})")
    print(f"  Demographic Parity Diff:  {metrics['demographic_parity_difference']:+.4f}  (ideal: 0)")
    print(f"  Demographic Parity Ratio: {metrics['demographic_parity_ratio']:.4f}   (ideal: 1, >0.8 ok)")
    print(f"  Equalized Odds Diff:      {metrics['equalized_odds_difference']:+.4f}  (ideal: 0)")
    print(f"  Equal Opportunity Diff:   {metrics['equal_opportunity_difference']:+.4f}  (ideal: 0)")
    print(f"  Predictive Parity Diff:   {metrics['predictive_parity_difference']:+.4f}  (ideal: 0)")


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(args):
    if args.model == "lr":
        return LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=42)
    elif args.model == "dt":
        return DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=args.min_samples_split, random_state=42)
    elif args.model == "svm":
        return SVC(C=args.C, kernel=args.kernel, probability=True, random_state=42)
    elif args.model == "knn":
        return KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, choices=["adult", "compas", "german"])
    parser.add_argument("--model",    required=True, choices=["lr", "dt", "svm", "knn"])
    parser.add_argument("--C",                type=float, default=1.0)
    parser.add_argument("--max_iter",         type=int,   default=1000)
    parser.add_argument("--kernel",           type=str,   default="rbf")
    parser.add_argument("--max_depth",        type=int,   default=5)
    parser.add_argument("--min_samples_split",type=int,   default=2)
    parser.add_argument("--n_neighbors",      type=int,   default=5)
    parser.add_argument("--weights",          type=str,   default="uniform")
    args = parser.parse_args()

    # Load data
    X, y, protected, attr = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test, p_train, p_test = preprocess(X, y, protected)

    print(f"\n{'='*55}")
    print(f"  MITIGATION: {args.dataset.upper()} | {args.model.upper()}")
    print(f"{'='*55}")

    # ─────────────────────────────────────────
    # BASELINE
    # ─────────────────────────────────────────
    model_base = build_model(args)
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    metrics_base = fairness_metrics(y_test, y_pred_base, p_test)
    print_metrics(metrics_base, "BASELINE")

    # ─────────────────────────────────────────
    # AIF360 — Reweighing (pre-processing)
    # ─────────────────────────────────────────
    import pandas as pd

    # Build AIF360 dataset
    train_df = X_train.copy()
    train_df["target"]    = y_train.values
    train_df[attr]        = p_train.values

    aif_train = BinaryLabelDataset(
        df=train_df,
        label_names=["target"],
        protected_attribute_names=[attr],
        favorable_label=1,
        unfavorable_label=0
    )

    RW = Reweighing(
        unprivileged_groups=[{attr: 0}],
        privileged_groups=[{attr: 1}]
    )
    aif_train_rw = RW.fit_transform(aif_train)

    sample_weights = aif_train_rw.instance_weights

    model_rw = build_model(args)
    model_rw.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_rw = model_rw.predict(X_test)
    metrics_rw = fairness_metrics(y_test, y_pred_rw, p_test)
    print_metrics(metrics_rw, "AIF360 Reweighing")

    # ─────────────────────────────────────────
    # Fairlearn — ThresholdOptimizer (post-processing)
    # ─────────────────────────────────────────
    model_fl = build_model(args)
    model_fl.fit(X_train, y_train)

    to = ThresholdOptimizer(
        estimator=model_fl,
        constraints="demographic_parity",
        predict_method="predict_proba" if args.model != "knn" else "predict_proba",
        objective="accuracy_score"
    )
    to.fit(X_train, y_train, sensitive_features=p_train)
    y_pred_fl = to.predict(X_test, sensitive_features=p_test)
    metrics_fl = fairness_metrics(y_test, y_pred_fl, p_test)
    print_metrics(metrics_fl, "Fairlearn ThresholdOptimizer")

    # ─────────────────────────────────────────
    # MLflow logging
    # ─────────────────────────────────────────
    mlflow.set_experiment(f"mitigation_{args.dataset}")

    with mlflow.start_run(run_name=f"mitig_{args.model}_{args.dataset}"):
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("model",   args.model)
        mlflow.log_param("protected", attr)
        if args.model in ["lr", "svm"]:
            mlflow.log_param("C", args.C)
        if args.model == "svm":
            mlflow.log_param("kernel", args.kernel)
        if args.model == "dt":
            mlflow.log_param("max_depth", args.max_depth)
        if args.model == "knn":
            mlflow.log_param("n_neighbors", args.n_neighbors)

        for k, v in metrics_base.items():
            if isinstance(v, float) and not np.isnan(v):
                mlflow.log_metric(f"baseline_{k}", v)
        for k, v in metrics_rw.items():
            if isinstance(v, float) and not np.isnan(v):
                mlflow.log_metric(f"aif360_{k}", v)
        for k, v in metrics_fl.items():
            if isinstance(v, float) and not np.isnan(v):
                mlflow.log_metric(f"fairlearn_{k}", v)

    # ─────────────────────────────────────────
    # Save JSON
    # ─────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    result = {
        "dataset":  args.dataset,
        "model":    args.model,
        "params":   vars(args),
        "baseline": metrics_base,
        "aif360":   metrics_rw,
        "fairlearn":metrics_fl,
    }
    filename = f"results/mitig_{args.dataset}_{args.model}_C{args.C}_depth{args.max_depth}_k{args.n_neighbors}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {filename}")


if __name__ == "__main__":
    main()