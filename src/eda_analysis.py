# Optimized Exploratory Data Analysis (EDA) for Iris Dataset
# Author: Imtiaz Nabi


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =========================
# GLOBAL SETTINGS
# =========================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
"figure.figsize": (12, 8),
"axes.titlesize": 14,
"axes.labelsize": 12,
"xtick.labelsize": 10,
"ytick.labelsize": 10,
"legend.fontsize": 10
})


# =========================
# LOAD DATA
# =========================
features_df = pd.read_csv("data/features.csv")
targets_df = pd.read_csv("data/targets.csv")
df = pd.concat([features_df, targets_df], axis=1)


print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - IRIS DATASET")
print("=" * 80)


# =========================
# 1. DATASET OVERVIEW
# =========================
print("\n1. DATASET OVERVIEW\n" + "-" * 80)
print(f"Shape: {df.shape} | Samples: {len(df)} | Features: {len(features_df.columns)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")


# =========================
# 2. MISSING VALUES
# =========================
print("\n2. MISSING VALUES\n" + "-" * 80)
missing = df.isnull().sum()
print("Missing values:", missing.sum())
if missing.sum():
print(missing[missing > 0])
else:
print("No missing values!")


# =========================
# 3. TARGET DISTRIBUTION
# =========================
print("\n3. TARGET VARIABLE DISTRIBUTION\n" + "-" * 80)
class_counts = df["class"].value_counts()
print(class_counts)
print("Proportions:", df["class"].value_counts(normalize=True))
print("Balanced" if class_counts.std() < 5 else "Imbalanced")


# =========================
# 4. STATISTICAL SUMMARY
# =========================
print("\n4. STATISTICAL SUMMARY\n" + "-" * 80)
print(df.describe())


# =========================
# 5. FEATURE STATS BY CLASS
# =========================
print("\n5. FEATURE STATS BY CLASS\n" + "-" * 80)
for col in features_df.columns:
print(f"\n{col.upper()}:")
print(df.groupby("class")[col].agg(["mean", "std", "min", "max"]))


# =========================
# 6. CORRELATION ANALYSIS
# =========================
print("\n6. CORRELATION ANALYSIS\n" + "-" * 80)
corr = features_df.corr()
print(corr)
high_corr = [(i, j, corr.iloc[i, j])
for i in range(len(corr.columns))
for j in range(i+1, len(corr.columns))
if abs(corr.iloc[i, j]) > 0.8]
if high_corr:
print("\nHighly correlated features:")
for i, j, v in high_corr:
print(f" {corr.columns[i]} <-> {corr.columns[j]}: {v:.3f}")
else:
print("No high correlations (|r| > 0.8)")


# =========================
# 7. OUTLIER DETECTION (IQR)
print("=" * 80)