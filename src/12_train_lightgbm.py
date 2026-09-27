import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score
)

FEATURE_FILE = "experiments/ml_features.tsv"
MODEL_FILE = "models/lightgbm_entity_matcher.txt"
THRESHOLD_FILE = "models/best_threshold.json"

RANDOM_STATE = 42

print("=" * 70)
print("STEP 12 - LIGHTGBM TRAINING")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load feature dataset
# ---------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ---------------------------------------------------------
# 2. Define features
# ---------------------------------------------------------

ID_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "label"
]

FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in ID_COLUMNS
]

X = df[FEATURE_COLUMNS]
y = df["label"].astype(int)

groups = df["source1_entity_id"]

print(f"\nNumber of features: {len(FEATURE_COLUMNS)}")
print(f"Positive samples: {(y == 1).sum():,}")
print(f"Negative samples: {(y == 0).sum():,}")

# ---------------------------------------------------------
# 3. Group-based train/validation split
# ---------------------------------------------------------

print("\nCreating group-based train/validation split...")

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, valid_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

print(f"Training rows  : {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print(
    f"Training positives: {(y_train == 1).sum():,}"
)

print(
    f"Validation positives: {(y_valid == 1).sum():,}"
)

# ---------------------------------------------------------
# 4. LightGBM model
# ---------------------------------------------------------

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100,
            verbose=True
        )
    ]
)

# ---------------------------------------------------------
# 5. Validation prediction
# ---------------------------------------------------------

print("\nGenerating validation predictions...")

valid_prob = model.predict_proba(
    X_valid
)[:, 1]

auc = roc_auc_score(
    y_valid,
    valid_prob
)

print(f"\nValidation ROC-AUC: {auc:.6f}")

# ---------------------------------------------------------
# 6. Find best F0.5 threshold
# ---------------------------------------------------------

print("\nSearching for best F0.5 threshold...")

thresholds = np.arange(
    0.05,
    0.951,
    0.005
)

best_threshold = 0.5
best_f05 = -1.0
best_precision = 0.0
best_recall = 0.0

results = []

for threshold in thresholds:

    pred = (
        valid_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        pred,
        zero_division=0
    )

    f05 = fbeta_score(
        y_valid,
        pred,
        beta=0.5,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f05
        )
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

print("\n" + "=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(f"Threshold : {best_threshold:.3f}")
print(f"Precision : {best_precision:.6f}")
print(f"Recall    : {best_recall:.6f}")
print(f"F0.5      : {best_f05:.6f}")

# ---------------------------------------------------------
# 7. Show top thresholds
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "threshold",
        "precision",
        "recall",
        "f0_5"
    ]
)

print("\nTop 10 thresholds by F0.5:")

print(
    results_df
    .sort_values("f0_5", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.booster_.save_model(
    MODEL_FILE
)

print(f"\nModel saved to:")
print(MODEL_FILE)

# ---------------------------------------------------------
# 9. Save threshold
# ---------------------------------------------------------

threshold_info = {
    "threshold": float(best_threshold),
    "validation_f05": float(best_f05),
    "validation_precision": float(best_precision),
    "validation_recall": float(best_recall),
    "validation_auc": float(auc),
    "feature_columns": FEATURE_COLUMNS
}

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        threshold_info,
        f,
        indent=2
    )

print(f"Threshold saved to:")
print(THRESHOLD_FILE)

print("\n" + "=" * 70)
print("LIGHTGBM TRAINING COMPLETE")
print("=" * 70)
