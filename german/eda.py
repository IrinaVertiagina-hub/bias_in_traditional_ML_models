"""
german/eda.py
Exploratory Data Analysis — German Credit Dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from utils import load_dataset

X, y, protected, attr = load_dataset("german")

print("=" * 50)
print("GERMAN CREDIT DATASET — EDA")
print("=" * 50)

print(f"\nShape: {X.shape}")
print(f"Features: {list(X.columns)}")

print(f"\nTarget distribution:")
print(f"  Bad credit  (0): {(y==0).sum()} ({(y==0).mean():.2%})")
print(f"  Good credit (1): {(y==1).sum()} ({(y==1).mean():.2%})")

print(f"\nProtected attribute (age):")
print(f"  Age >= 25 (1): {(protected==1).sum()} ({(protected==1).mean():.2%})")
print(f"  Age <  25 (0): {(protected==0).sum()} ({(protected==0).mean():.2%})")

print(f"\nMissing values: {X.isnull().sum().sum()}")

print(f"\nBasic stats:")
print(X.describe().round(2))

print("\n" + "=" * 50)
print("BIAS INDICATORS")
print("=" * 50)

old_pos   = y[protected == 1].mean()
young_pos = y[protected == 0].mean()
print(f"\nReal good credit rate in data:")
print(f"  Age >= 25: {old_pos:.2%}  ({(y[protected==1]==1).sum()} out of {(protected==1).sum()})")
print(f"  Age <  25: {young_pos:.2%}  ({(y[protected==0]==1).sum()} out of {(protected==0).sum()})")
print(f"  Ratio:     {young_pos/old_pos:.4f}  (ideal: 1.0, >0.8 ok)")
print(f"\n  Data is biased before any model is trained.")