"""
app.py
Flask backend for the Bias in ML demo.
"""

import os
import json
import glob
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULTS_DIR = "results"


def find_baseline(dataset, model, C, max_depth, n_neighbors):
    pattern = os.path.join(RESULTS_DIR, f"{dataset}_{model}_*.json")
    for f in glob.glob(pattern):
        with open(f) as fp:
            r = json.load(fp)
        p = r.get("params", {})
        if model in ["lr", "svm"]:
            if abs(p.get("C", 1.0) - float(C)) < 1e-6:
                return r
        elif model == "dt":
            if p.get("max_depth", 5) == int(max_depth):
                return r
        elif model == "knn":
            if p.get("n_neighbors", 5) == int(n_neighbors):
                return r
    return None


def find_mitigation(dataset, model, C, max_depth, n_neighbors):
    pattern = os.path.join(RESULTS_DIR, f"mitig_{dataset}_{model}_*.json")
    for f in glob.glob(pattern):
        with open(f) as fp:
            r = json.load(fp)
        p = r.get("params", {})
        if model in ["lr", "svm"]:
            if abs(p.get("C", 1.0) - float(C)) < 1e-6:
                return r
        elif model == "dt":
            if p.get("max_depth", 5) == int(max_depth):
                return r
        elif model == "knn":
            if p.get("n_neighbors", 5) == int(n_neighbors):
                return r
    return None

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/baseline")
def api_baseline():
    dataset     = request.args.get("dataset", "adult")
    model       = request.args.get("model", "lr")
    C           = float(request.args.get("C", 1.0))
    max_depth   = int(request.args.get("max_depth", 5))
    n_neighbors = int(request.args.get("n_neighbors", 5))

    result = find_baseline(dataset, model, C, max_depth, n_neighbors)
    if result is None:
        return jsonify({"error": "No baseline result found"}), 404
    return jsonify(result)


@app.route("/api/mitigation")
def api_mitigation():
    dataset     = request.args.get("dataset", "adult")
    model       = request.args.get("model", "lr")
    C           = float(request.args.get("C", 1.0))
    max_depth   = int(request.args.get("max_depth", 5))
    n_neighbors = int(request.args.get("n_neighbors", 5))

    result = find_mitigation(dataset, model, C, max_depth, n_neighbors)
    if result is None:
        return jsonify({"error": "No mitigation result found"}), 404
    return jsonify(result)


@app.route("/api/datasets")
def api_datasets():
    info = {
        "adult": {
            "name": "UCI Adult",
            "description": "A snapshot of 1994 America: can an algorithm predict who earns over $50K a year? The catch — the data reflects a world where women were paid less, and the model learns exactly that bias.",
            "protected": "sex",
            "privileged": "Male",
            "unprivileged": "Female",
            "samples": 32561,
            "task": "Income prediction"
        },
        "compas": {
            "name": "COMPAS",
            "description": "US courts used this algorithm to predict whether a criminal defendant would reoffend. In 2016, journalists proved it flagged Black defendants as high-risk twice as often as white defendants with the same background.",
            "protected": "race",
            "privileged": "Caucasian",
            "unprivileged": "African-American",
            "samples": 5278,
            "task": "Recidivism prediction"
        },
        "german": {
            "name": "German Credit",
            "description": "A German bank decides who gets a loan. This dataset reveals how age quietly shapes that decision — younger applicants may get rejected not because of their finances, but simply because they are young.",
            "protected": "age",
            "privileged": "Age ≥ 25",
            "unprivileged": "Age < 25",
            "samples": 1000,
            "task": "Credit risk prediction"
        }
    }
    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=True, port=5050)