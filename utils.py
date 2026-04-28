"""
utils.py
Загрузка и препроцессинг датасетов: UCI Adult, COMPAS, German Credit.
Protected attributes: sex/race (Adult, COMPAS), age (German Credit).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# UCI Adult Dataset
# ─────────────────────────────────────────────

def load_adult():
    """
    Загружает UCI Adult dataset.
    Target: income (1 = >50K, 0 = <=50K)
    Protected: sex (1 = Male, 0 = Female)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)

    # Target
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)

    # Protected attribute
    df["sex"] = (df["sex"].str.strip() == "Male").astype(int)  # 1=Male, 0=Female

    # Drop unused
    df.drop(columns=["fnlwgt", "education", "native_country", "race"], inplace=True)

    # Encode categoricals
    cat_cols = ["workclass", "marital_status", "occupation", "relationship"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["income"])
    y = df["income"]
    protected = df["sex"]  # Series

    return X, y, protected, "sex"


# ─────────────────────────────────────────────
# COMPAS Dataset
# ─────────────────────────────────────────────

def load_compas():
    """
    Загружает ProPublica COMPAS dataset.
    Target: two_year_recid (1 = recidivated, 0 = not)
    Protected: race (1 = Caucasian, 0 = African-American)
    """
    url = (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv"
    )
    df = pd.read_csv(url)

    # Filter как в оригинальном анализе ProPublica
    df = df[df["days_b_screening_arrest"].between(-30, 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[df["race"].isin(["African-American", "Caucasian"])]

    # Protected attribute
    df["race_binary"] = (df["race"] == "Caucasian").astype(int)  # 1=Caucasian, 0=AA

    features = [
        "age", "priors_count", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "race_binary"
    ]
    df["sex"] = (df["sex"] == "Male").astype(int)
    features.append("sex")

    df["c_charge_degree"] = (df["c_charge_degree"] == "F").astype(int)
    features.append("c_charge_degree")

    X = df[features].copy()
    y = df["two_year_recid"]
    protected = df["race_binary"]

    return X, y, protected, "race_binary"


# ─────────────────────────────────────────────
# German Credit Dataset
# ─────────────────────────────────────────────

def load_german():
    """
    Загружает German Credit dataset.
    Target: credit risk (1 = good, 0 = bad)
    Protected: age (1 = >=25, 0 = <25)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status", "other_debtors",
        "residence_since", "property", "age", "other_installment", "housing",
        "existing_credits", "job", "liable_people", "telephone", "foreign_worker", "target"
    ]
    df = pd.read_csv(url, sep=" ", names=cols)

    # Target: 1=good credit, 0=bad credit
    df["target"] = (df["target"] == 1).astype(int)

    # Protected attribute: age >= 25
    df["age_binary"] = (df["age"] >= 25).astype(int)

    # Encode categoricals
    cat_cols = [
        "checking_status", "credit_history", "purpose", "savings",
        "employment", "personal_status", "other_debtors", "property",
        "other_installment", "housing", "job", "telephone", "foreign_worker"
    ]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["target"])
    y = df["target"]
    protected = df["age_binary"]

    return X, y, protected, "age_binary"


# ─────────────────────────────────────────────
# Unified loader
# ─────────────────────────────────────────────

DATASETS = {
    "adult": load_adult,
    "compas": load_compas,
    "german": load_german,
}

def load_dataset(name: str):
    """
    Возвращает X, y, protected, protected_attr_name для заданного датасета.
    name: 'adult' | 'compas' | 'german'
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    return DATASETS[name]()


def preprocess(X, y, protected, test_size=0.2, random_state=42, scale=True):
    """
    Train/test split + опциональная стандартизация.
    Возвращает: X_train, X_test, y_train, y_test, prot_train, prot_test
    """
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, protected,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

    return X_train, X_test, y_train, y_test, p_train, p_test


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for name in DATASETS:
        print(f"\n{'='*40}")
        print(f"Dataset: {name.upper()}")
        X, y, protected, attr = load_dataset(name)
        X_train, X_test, y_train, y_test, p_train, p_test = preprocess(X, y, protected)
        print(f"  Features:  {X.shape[1]}")
        print(f"  Samples:   {len(X)}")
        print(f"  Train/Test: {len(X_train)} / {len(X_test)}")
        print(f"  Target balance: {y.mean():.2%} positive")
        print(f"  Protected ({attr}): {protected.mean():.2%} privileged")