# ML Classification Pipeline for Iris Dataset
# Author: Imtiaz Nabi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import warnings
import pickle

# =========================
# GLOBAL SETTINGS
# =========================
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="coolwarm")

print("=" * 100)
print(" " * 30 + "IRIS CLASSIFICATION PIPELINE")
print(" " * 25 + "A Data Story by Yan Holtz Style")
print("=" * 100)

# =========================
# CHAPTER 1: Data Preparation
# =========================
features_df = pd.read_csv("data/features.csv")
targets_df = pd.read_csv("data/targets.csv")

X = features_df.values
y = targets_df.values.ravel()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n" + "-" * 100)
print("CHAPTER 1: Data Preparation")
print("-" * 100)
print(f"\n[DATA] Dataset: {X.shape[0]} samples x {X.shape[1]} features")
print(
    f"[TARGET] Classes: {list(label_encoder.classes_)} => Encoded as: {np.unique(y_encoded)}"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\n[SPLIT] Train-Test (70-30 stratified)")
print(f"   Training: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[SCALE] StandardScaler applied")

# =========================
# CHAPTER 2: Baseline Models
# =========================
print("\n" + "-" * 100)
print("CHAPTER 2: Baseline Model Training")
print("-" * 100)

classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
}

baseline_results = {}
print("\n[TRAIN] Fitting baseline models...")

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)

    baseline_results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(
        f"[OK] {name}: Acc={baseline_results[name]['accuracy']:.4f}, F1={baseline_results[name]['f1']:.4f}, CV={baseline_results[name]['cv_mean']:.4f} (+/-{baseline_results[name]['cv_std']:.4f})"
    )

# =========================
# CHAPTER 3: Hyperparameter Tuning
# =========================
print("\n" + "-" * 100)
print("CHAPTER 3: Hyperparameter Tuning")
print("-" * 100)

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
}

tuned_results = {}
print("\n[TUNE] GridSearchCV (5-fold)...\n")

for name, clf in classifiers.items():
    print(f"[TUNING] {name}...")
    grid_search = GridSearchCV(
        clf, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

    tuned_results[name] = {
        "model": best_model,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
    }

    print(f"  Best CV Score: {tuned_results[name]['best_cv_score']:.4f}")
    print(f"  Test Accuracy: {tuned_results[name]['accuracy']:.4f}")
    print(f"  Best Params : {tuned_results[name]['best_params']}")

# =========================
# CHAPTER 4: Performance Comparison
# =========================
print("\n" + "-" * 100)
print("CHAPTER 4: Performance Comparison")
print("-" * 100)

comparison_data = []
for name in classifiers.keys():
    for model_type, results in zip(
        ["Baseline", "Tuned"], [baseline_results[name], tuned_results[name]]
    ):
        comparison_data.append(
            {
                "Model": name,
                "Type": model_type,
                "Accuracy": results["accuracy"],
                "Precision": results["precision"],
                "Recall": results["recall"],
                "F1-Score": results["f1"],
                "CV Mean": results["cv_mean"],
                "CV Std": results["cv_std"],
            }
        )

comparison_df = pd.DataFrame(comparison_data)
print("\n[RESULTS] Baseline vs Tuned:")
print(comparison_df.to_string(index=False))

improvement_df = pd.DataFrame(
    [
        {
            "Model": name,
            "Baseline Accuracy": baseline_results[name]["accuracy"],
            "Tuned Accuracy": tuned_results[name]["accuracy"],
            "Improvement (%)": (
                (tuned_results[name]["accuracy"] - baseline_results[name]["accuracy"])
                / baseline_results[name]["accuracy"]
            )
            * 100,
        }
        for name in classifiers.keys()
    ]
)

print("\n[IMPROVE] Accuracy Gains:")
print(improvement_df.to_string(index=False))

# =========================
# CHAPTER 5: Classification Reports
# =========================
print("\n" + "-" * 100)
print("CHAPTER 5: Detailed Classification Reports")
print("-" * 100)

for name in classifiers:
    print(f"\n{'=' * 50}\n{name.upper()} - TUNED MODEL\n{'=' * 50}")
    print(
        classification_report(
            y_test, tuned_results[name]["y_pred"], target_names=label_encoder.classes_
        )
    )

# =========================
# SAVE RESULTS
# =========================
print("\n" + "=" * 100)
print("Saving results and generating visualizations...")
print("=" * 100)

results_summary = {
    "baseline_results": baseline_results,
    "tuned_results": tuned_results,
    "comparison_df": comparison_df,
    "improvement_df": improvement_df,
    "label_encoder": label_encoder,
    "scaler": scaler,
    "X_test": X_test_scaled,
    "y_test": y_test,
}

with open("ml_pipeline_results.pkl", "wb") as f:
    pickle.dump(results_summary, f)

print("\n[SAVE] Results saved to: ml_pipeline_results.pkl")
print("[READY] Ready for visualization generation...")
