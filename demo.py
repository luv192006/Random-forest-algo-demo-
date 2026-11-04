"""
Random Forest Demo Project: Breast Cancer Diagnosis
---------------------------------------------------
Covers: data exploration, baseline vs. forest, OOB score, cross-validation,
hyperparameter tuning, feature importance, permutation importance,
evaluation, and saving the model.

Install:  pip install scikit-learn pandas matplotlib joblib
Run:      python random_forest_demo.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and explore the data
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign

print("=== 1. Data overview ===")
print("Shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Class balance:\n", y.value_counts(normalize=True).rename({0: "malignant", 1: "benign"}).round(3))

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified keeps class ratios equal)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Baseline: a single decision tree vs. a random forest
# ---------------------------------------------------------------------------
print("\n=== 2. Single tree vs. random forest ===")
tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
forest = RandomForestClassifier(
    n_estimators=200, oob_score=True, n_jobs=-1, random_state=RANDOM_STATE
).fit(X_train, y_train)

print(f"Decision tree  train={tree.score(X_train, y_train):.3f}  test={tree.score(X_test, y_test):.3f}")
print(f"Random forest  train={forest.score(X_train, y_train):.3f}  test={forest.score(X_test, y_test):.3f}")
print(f"Random forest OOB score: {forest.oob_score_:.3f}  (free validation estimate from bagging)")

# ---------------------------------------------------------------------------
# 4. Effect of the number of trees
# ---------------------------------------------------------------------------
print("\n=== 3. Accuracy vs. number of trees ===")
tree_counts = [1, 5, 10, 25, 50, 100, 200, 400]
test_scores = []
for n in tree_counts:
    rf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=RANDOM_STATE).fit(X_train, y_train)
    test_scores.append(rf.score(X_test, y_test))
    print(f"n_estimators={n:<4} test accuracy={test_scores[-1]:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(tree_counts, test_scores, marker="o")
plt.xscale("log")
plt.xlabel("Number of trees")
plt.ylabel("Test accuracy")
plt.title("More trees stabilise the forest")
plt.tight_layout()
plt.savefig("trees_vs_accuracy.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Cross-validation for a more reliable estimate
# ---------------------------------------------------------------------------
print("\n=== 4. 5-fold cross-validation ===")
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    X_train, y_train, cv=5, scoring="roc_auc",
)
print(f"ROC-AUC per fold: {np.round(cv_scores, 3)}")
print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning
# ---------------------------------------------------------------------------
print("\n=== 5. Hyperparameter tuning (RandomizedSearchCV) ===")
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}
search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print(f"Best CV ROC-AUC: {search.best_score_:.3f}")
best_model = search.best_estimator_

# ---------------------------------------------------------------------------
# 7. Final evaluation on the held-out test set
# ---------------------------------------------------------------------------
print("\n=== 6. Test set evaluation ===")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. Feature importance: impurity-based vs. permutation
# ---------------------------------------------------------------------------
print("\n=== 7. Feature importance ===")
impurity_imp = pd.Series(best_model.feature_importances_, index=X.columns)
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)
perm_imp = pd.Series(perm.importances_mean, index=X.columns)

comparison = pd.DataFrame({"impurity": impurity_imp, "permutation": perm_imp})
print(comparison.sort_values("impurity", ascending=False).head(10).round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
impurity_imp.nlargest(10).sort_values().plot.barh(ax=axes[0], title="Impurity-based (top 10)")
perm_imp.nlargest(10).sort_values().plot.barh(ax=axes[1], title="Permutation (top 10)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. Save the model and use it for a new prediction
# ---------------------------------------------------------------------------
joblib.dump(best_model, "rf_breast_cancer.joblib")
loaded = joblib.load("rf_breast_cancer.joblib")

sample = X_test.iloc[[0]]
label = data.target_names[loaded.predict(sample)[0]]
prob = loaded.predict_proba(sample)[0].max()
print(f"\n=== 8. Sample prediction ===\nPredicted: {label} (confidence {prob:.2f}), actual: {data.target_names[y_test.iloc[0]]}")
print("\nSaved: rf_breast_cancer.joblib, trees_vs_accuracy.png, confusion_matrix.png, feature_importance.png")
