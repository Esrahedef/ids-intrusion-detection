"""
Intrusion Detection System (IDS)
Author: Esra

This script trains a machine learning model on the CIC-IDS2017 dataset
using a realistic time-based split.

Key idea:
Train on past traffic → test on future traffic
to simulate real-world deployment.
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    balanced_accuracy_score
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# 1. LOAD DATA

# Expected structure:
# data/
# ├── Monday-WorkingHours.pcap_ISCX.csv
# ├── Tuesday-WorkingHours.pcap_ISCX.csv
# ├── Wednesday-workingHours.pcap_ISCX.csv
# ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

path = "data/"

print("Available files:")
for f in sorted(os.listdir(path)):
    print(f)


def load(file, day_id):
    """
    Load a CSV file and convert labels:
    BENIGN → 0
    ATTACK → 1
    """
    df = pd.read_csv(path + file)

    df[" Label"] = df[" Label"].apply(
        lambda x: 0 if x.strip() == "BENIGN" else 1
    )
    df.rename(columns={" Label": "Label"}, inplace=True)

    # Add day information (used later for splitting)
    df["day"] = day_id
    return df


# Load different days
monday = load("Monday-WorkingHours.pcap_ISCX.csv", 0)      # mostly benign
tuesday = load("Tuesday-WorkingHours.pcap_ISCX.csv", 1)    # brute force
wednesday = load("Wednesday-workingHours.pcap_ISCX.csv", 2)  # mixed attacks
friday = load("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 3)  # DDoS



# 2. TIME-BASED SPLIT
"""
Instead of random split, we simulate real life:

Train → past data (Mon, Tue, Wed)
Test  → future data (Friday)

This avoids unrealistic performance.
"""

train_df = pd.concat([monday, tuesday, wednesday]).reset_index(drop=True)
test_df = friday.reset_index(drop=True)

print("\n=== SPLIT DISTRIBUTION ===")
print("Train:", train_df["Label"].value_counts().to_dict())
print("Test: ", test_df["Label"].value_counts().to_dict())


# 3. REMOVE LEAKAGE FEATURES
"""
These features can leak attack information directly
and lead to unrealistically high performance.
"""

LEAKY = [
    " Destination Port",
    "Flow Bytes/s",
    "Flow Packets/s",
    " Flow Duration",
    " Fwd Packets/s",
    " Bwd Packets/s",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " min_seg_size_forward",
    " Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
]

for col in LEAKY:
    if col in train_df.columns:
        train_df.drop(columns=col, inplace=True)
        test_df.drop(columns=col, inplace=True)



# 4. FEATURE / TARGET SPLIT
X_train_raw = train_df.drop(columns=["Label", "day"])
y_train = train_df["Label"]

X_test_raw = test_df.drop(columns=["Label", "day"])
y_test = test_df["Label"]


# 5. CLEAN + IMPUTE
"""
Replace infinite values and fill missing values.
Important: fit only on training data.
"""

X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)


# 6. REMOVE DUPLICATES (TRAIN ONLY)
"""
Same network flows can appear multiple times.
If duplicates leak into test, model performance becomes inflated.
"""

train_arr = np.hstack([X_train_imp, y_train.values.reshape(-1, 1)])
train_arr = pd.DataFrame(train_arr).drop_duplicates().values

X_train_imp = train_arr[:, :-1]
y_train_dd = train_arr[:, -1].astype(int)

print(f"\nTrain size after deduplication: {len(X_train_imp):,}")
print(f"Attack samples: {(y_train_dd == 1).sum():,}")
print(f"Benign samples: {(y_train_dd == 0).sum():,}")


# 7. HANDLE IMBALANCE
"""
Dataset is highly imbalanced.

Strategy:
- Use SMOTE to increase attack samples
- Slightly reduce benign samples
"""

n_attack = (y_train_dd == 1).sum()
n_benign = (y_train_dd == 0).sum()

if n_attack > 80000:
    target_attack = n_attack
else:
    target_attack = min(int(n_benign * 0.25), 80000)

target_benign = int(target_attack * 2)

resampler = ImbPipeline([
    ("smote", SMOTE(sampling_strategy={1: target_attack}, random_state=42)),
    ("under", RandomUnderSampler(sampling_strategy={0: target_benign}, random_state=42))
])

X_bal, y_bal = resampler.fit_resample(X_train_imp, y_train_dd)

print("\nBalanced distribution:", pd.Series(y_bal).value_counts().to_dict())


# 8. SCALING
scaler = StandardScaler()
X_bal_sc = scaler.fit_transform(X_bal)
X_test_sc = scaler.transform(X_test_imp)


# 9. THRESHOLD OPTIMIZATION
"""
Instead of default threshold (0.5),
we find a better threshold using F1 score.
"""

from sklearn.model_selection import train_test_split as tts

X_t, X_v, y_t, y_v = tts(
    X_bal_sc, y_bal,
    test_size=0.3,
    stratify=y_bal,
    random_state=42
)

rf_params = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

model_thr = RandomForestClassifier(**rf_params)
model_thr.fit(X_t, y_t)

val_probs = model_thr.predict_proba(X_v)[:, 1]

prec, rec, thresholds = precision_recall_curve(y_v, val_probs)
f1 = 2 * prec * rec / (prec + rec + 1e-9)

best_thr = float(thresholds[np.argmax(f1)])
print(f"\nOptimal threshold: {best_thr:.3f}")


# 10. FINAL MODEL
model_final = RandomForestClassifier(**rf_params)
model_final.fit(X_bal_sc, y_bal)

test_probs = model_final.predict_proba(X_test_sc)[:, 1]
y_pred = (test_probs >= 0.5).astype(int)


# 11. EVALUATION
print("\n_____ FINAL RESULTS _____")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:\n{cm}")
print(f"Attack Recall:    {tp/(tp+fn):.4f}")
print(f"False Alarm Rate: {fp/(fp+tn):.4f}")
print(f"Balanced Acc:     {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:          {roc_auc_score(y_test, test_probs):.4f}")


# 12. SANITY CHECK
"""
If AUC is too high (~1.0), something is wrong (likely leakage).
"""

print("\n=== SANITY CHECK ===")
print(f"Mean attack prob: {test_probs[y_test==1].mean():.3f}")
print(f"Mean benign prob: {test_probs[y_test==0].mean():.3f}")

if roc_auc_score(y_test, test_probs) > 0.98:
    print("\n⚠ Suspiciously high AUC → check feature leakage.")
else:
    print("\n✓ Results look realistic.")
