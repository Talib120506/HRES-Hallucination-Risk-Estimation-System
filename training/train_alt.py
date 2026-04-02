import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.feature_selection import VarianceThreshold

# --- OLD MODELS COMMENTED OUT ---
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# --- NOVEL MODELS ---
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

# Setup directories
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "training_data"), exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
df = pd.read_excel(os.path.join(DATA_DIR, "processed", "features_shuffled_final.xlsx"))

meta_cols    = ["question_id", "doc_id", "question", "answer", "answer_type"]
feature_cols = [c for c in df.columns if c.startswith("v_")] + ["seq_len", "target_index"]
X    = df[feature_cols].values.astype(np.float64)
y    = df["label"].values.astype(int)
meta = df[meta_cols]

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features  |  label 0: {(y==0).sum()}, label 1: {(y==1).sum()}")

# ============================================================================
# 2. PREPROCESSING: Variance Threshold + StandardScaler
# ============================================================================
vt = VarianceThreshold(threshold=1e-6)
X_vt = vt.fit_transform(X)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_vt)

print(f"After preprocessing: {X_vt.shape[1]} features (removed {X.shape[1] - X_vt.shape[1]} low-variance)")

# ============================================================================
# 3. PCA VARIANCE ANALYSIS
# ============================================================================
full_pca = PCA(random_state=RANDOM_STATE)
full_pca.fit(X_scaled)
cumvar = np.cumsum(full_pca.explained_variance_ratio_)
n95 = np.searchsorted(cumvar, 0.95) + 1
n99 = np.searchsorted(cumvar, 0.99) + 1
print(f"PCA: 95% variance -> {n95} components, 99% -> {n99} components")

# ============================================================================
# 4. GPU CHECK
# ============================================================================
# GPU check bypassed for sklearn native models
USE_GPU = False
print("GPU: not needed for sklearn native ensemble/distance algorithms")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ============================================================================
# 5. DEFINE REDUCTION METHODS
# ============================================================================
def make_pca(n):
    return PCA(n_components=n, random_state=RANDOM_STATE)

def make_pca_lda(n):
    return Pipeline([
        ("pca", PCA(n_components=n, random_state=RANDOM_STATE)),
        ("lda", LDA(n_components=1)),
    ])

reduction_methods = {
    "PCA": {
        "factory":    make_pca,
        "n_list":     [150, 200, 250, 300],
        "supervised": False,
    },
    "PCA_LDA": {
        "factory":    make_pca_lda,
        "n_list":     [150, 200, 250, 300],
        "supervised": True,
    },
}

print(f"\nTesting reduction methods: {list(reduction_methods.keys())}")

# ============================================================================
# 6. CLASSIFIER CONFIGS
# ============================================================================

# AdaBoost: robust boosting algorithm sensitive to noisy data and outliers
ada_configs = [
    {"n_estimators": 50,  "learning_rate": 0.1},
    {"n_estimators": 100, "learning_rate": 0.5},
    {"n_estimators": 200, "learning_rate": 1.0},
    {"n_estimators": 300, "learning_rate": 1.0},
]

# ExtraTrees: extreme randomized trees (builds on Random Forest but creates splits at random, lowering variance further)
et_configs = [
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1,  "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2,  "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 10,   "min_samples_leaf": 1,  "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1,  "max_features": "log2"},
]

# KNeighbors: surprisingly performant distance-based neighbor metric on PCA spaces
knn_configs = [
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "distance"},
    {"n_neighbors": 10, "weights": "distance"},
]

'''
# ====== OLD CONFIGS COMMENTED OUT ======
xgb_configs = [...]
svm_configs = [...]
lr_configs = [...]
mlp_configs = [...]
rf_configs = [...]
'''

# ============================================================================
# 7. GRID SEARCH
# ============================================================================
best_overall_score     = -1
best_overall_reduction = None
best_overall_n         = None
best_overall_clf       = None
best_overall_params    = None
results_summary        = []

for red_name, red_cfg in reduction_methods.items():
    factory    = red_cfg["factory"]
    n_list     = red_cfg["n_list"]
    supervised = red_cfg["supervised"]

    print(f"\n{'='*70}\nTESTING {red_name}\n{'='*70}")

    # ── AdaBoost ──────────────────────────────────────────────────────────────
    print(f"{red_name} + AdaBoost grid search...", end=" ", flush=True)
    best_ada_score, best_ada_n, best_ada_params = -1, None, None

    for n in n_list:
        for ada_p in ada_configs:
            fold_accs = []
            for tr_idx, vl_idx in skf.split(X_scaled, y):
                reducer = factory(n)
                X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
                       else reducer.fit_transform(X_scaled[tr_idx])
                X_vl = reducer.transform(X_scaled[vl_idx])

                mdl = AdaBoostClassifier(random_state=RANDOM_STATE, **ada_p)
                mdl.fit(X_tr, y[tr_idx])
                fold_accs.append(accuracy_score(y[vl_idx], mdl.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_ada_score:
                best_ada_score, best_ada_n, best_ada_params = mean_acc, n, ada_p

    print(f"done  |  n={best_ada_n}, estimators={best_ada_params['n_estimators']}, lr={best_ada_params['learning_rate']}, Acc={best_ada_score:.4f}")
    results_summary.append({"Reduction": red_name, "Classifier": "AdaBoost",
                             "n_components": best_ada_n, "Accuracy": best_ada_score})
    if best_ada_score > best_overall_score:
        best_overall_score, best_overall_reduction = best_ada_score, red_name
        best_overall_n, best_overall_clf, best_overall_params = best_ada_n, "AdaBoost", best_ada_params

    # ── ExtraTrees ────────────────────────────────────────────────────────────
    print(f"{red_name} + ExtraTrees grid search...", end=" ", flush=True)
    best_et_score, best_et_n, best_et_params = -1, None, None

    for n in n_list:
        for et_p in et_configs:
            fold_accs = []
            for tr_idx, vl_idx in skf.split(X_scaled, y):
                reducer = factory(n)
                X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
                       else reducer.fit_transform(X_scaled[tr_idx])
                X_vl = reducer.transform(X_scaled[vl_idx])

                mdl = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, **et_p)
                mdl.fit(X_tr, y[tr_idx])
                fold_accs.append(accuracy_score(y[vl_idx], mdl.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_et_score:
                best_et_score, best_et_n, best_et_params = mean_acc, n, et_p

    print(f"done  |  n={best_et_n}, leaf={best_et_params['min_samples_leaf']}, Acc={best_et_score:.4f}")
    results_summary.append({"Reduction": red_name, "Classifier": "ExtraTrees",
                             "n_components": best_et_n, "Accuracy": best_et_score})
    if best_et_score > best_overall_score:
        best_overall_score, best_overall_reduction = best_et_score, red_name
        best_overall_n, best_overall_clf, best_overall_params = best_et_n, "ExtraTrees", best_et_params

    # ── KNeighbors ────────────────────────────────────────────────────────────
    print(f"{red_name} + KNeighbors grid search...", end=" ", flush=True)
    best_knn_score, best_knn_n, best_knn_params = -1, None, None

    for n in n_list:
        for knn_p in knn_configs:
            fold_accs = []
            for tr_idx, vl_idx in skf.split(X_scaled, y):
                reducer = factory(n)
                X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
                       else reducer.fit_transform(X_scaled[tr_idx])
                X_vl = reducer.transform(X_scaled[vl_idx])

                mdl = KNeighborsClassifier(**knn_p)
                mdl.fit(X_tr, y[tr_idx])
                fold_accs.append(accuracy_score(y[vl_idx], mdl.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_knn_score:
                best_knn_score, best_knn_n, best_knn_params = mean_acc, n, knn_p

    print(f"done  |  n={best_knn_n}, n={best_knn_params['n_neighbors']}, w={best_knn_params['weights']}, Acc={best_knn_score:.4f}")
    results_summary.append({"Reduction": red_name, "Classifier": "KNeighbors",
                             "n_components": best_knn_n, "Accuracy": best_knn_score})
    if best_knn_score > best_overall_score:
        best_overall_score, best_overall_reduction = best_knn_score, red_name
        best_overall_n, best_overall_clf, best_overall_params = best_knn_n, "KNeighbors", best_knn_params


# ============================================================================
# 8. RESULTS SUMMARY
# ============================================================================
print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
results_df = pd.DataFrame(results_summary).sort_values("Accuracy", ascending=False)
print(results_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"WINNER: {best_overall_reduction} + {best_overall_clf}")
print(f"n_components: {best_overall_n}  |  Accuracy: {best_overall_score:.4f}")
print(f"{'='*70}")

# ============================================================================
# 9. DETAILED EVALUATION OF WINNER
# ============================================================================
print(f"\nDetailed evaluation: {best_overall_reduction} + {best_overall_clf}...")

red_cfg    = reduction_methods[best_overall_reduction]
factory    = red_cfg["factory"]
supervised = red_cfg["supervised"]

# Fit final reducer on ALL data
final_reducer = factory(best_overall_n)
if supervised:
    X_reduced_full = final_reducer.fit_transform(X_scaled, y)
else:
    X_reduced_full = final_reducer.fit_transform(X_scaled)

# Export reduced training data
n_out_cols = X_reduced_full.shape[1]
col_names  = [f"{best_overall_reduction}_{i+1}" for i in range(n_out_cols)]
df_reduced = pd.concat([
    meta.reset_index(drop=True),
    pd.DataFrame({"label": y}),
    pd.DataFrame(X_reduced_full, columns=col_names),
], axis=1)
df_reduced.to_excel(
    os.path.join(DATA_DIR, "training_data", "training_data_final.xlsx"), index=False
)

# Cross-validated metrics for the winner
fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
all_y_true, all_y_pred, all_y_proba = [], [], []

def build_winner(params):
    """Return a fresh instance of the winning classifier."""
    if best_overall_clf == "AdaBoost":
        return AdaBoostClassifier(random_state=RANDOM_STATE, **params)
    elif best_overall_clf == "ExtraTrees":
        return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    elif best_overall_clf == "KNeighbors":
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {best_overall_clf}")

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"    Training fold {fold}/5 on {best_overall_reduction}({best_overall_n}):")
    reducer = factory(best_overall_n)
    X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
           else reducer.fit_transform(X_scaled[tr_idx])
    X_vl = reducer.transform(X_scaled[vl_idx])

    mdl = build_winner(best_overall_params)
    mdl.fit(X_tr, y[tr_idx])

    y_pred  = mdl.predict(X_vl)
    y_proba = mdl.predict_proba(X_vl)[:, 1]

    fold_metrics["accuracy"].append(accuracy_score(y[vl_idx], y_pred))
    fold_metrics["precision"].append(precision_score(y[vl_idx], y_pred, zero_division=0))
    fold_metrics["recall"].append(recall_score(y[vl_idx], y_pred, zero_division=0))
    fold_metrics["f1"].append(f1_score(y[vl_idx], y_pred, zero_division=0))
    fold_metrics["roc_auc"].append(roc_auc_score(y[vl_idx], y_proba))

    all_y_true.extend(y[vl_idx])
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

cm = confusion_matrix(all_y_true, all_y_pred)
print(f"\n{best_overall_clf} ({best_overall_reduction}={best_overall_n}):"
      f"  Acc={np.mean(fold_metrics['accuracy']):.4f}"
      f"  Prec={np.mean(fold_metrics['precision']):.4f}"
      f"  Rec={np.mean(fold_metrics['recall']):.4f}"
      f"  F1={np.mean(fold_metrics['f1']):.4f}"
      f"  AUC={np.mean(fold_metrics['roc_auc']):.4f}")
print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

# ============================================================================
# 10. TRAIN FINAL MODEL ON ALL DATA & SAVE
# ============================================================================
print(f"\nTraining final {best_overall_clf} on all data...", end=" ", flush=True)

MODEL_FILENAMES = {
    "AdaBoost":           "ada_model_final.pkl",
    "ExtraTrees":         "et_model_final.pkl",
    "KNeighbors":         "knn_model_final.pkl",
    # "XGBoost":            "xgb_model_final.pkl",
    # "SVM":                "svm_model_final.pkl",
    # "LogisticRegression": "lr_model_final.pkl",
    # "MLP":                "mlp_model_final.pkl",
    # "RandomForest":       "rf_model_final.pkl",
}

final_model = build_winner(best_overall_params)
final_model.fit(X_reduced_full, y)

model_file = MODEL_FILENAMES[best_overall_clf]
joblib.dump(final_model, os.path.join(MODELS_DIR, model_file))

# Save reducer + preprocessors
joblib.dump(final_reducer, os.path.join(MODELS_DIR, "reduction_final.pkl"))
joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler_final.pkl"))
joblib.dump(vt,            os.path.join(MODELS_DIR, "variance_threshold_final.pkl"))

# Save metadata so app.py knows which reduction method was used
pd.DataFrame([{
    "method":       best_overall_reduction,
    "n_components": best_overall_n,
    "n_output_dims": n_out_cols,
    "classifier":   best_overall_clf,
    "accuracy":     best_overall_score,
}]).to_csv(os.path.join(MODELS_DIR, "reduction_metadata.csv"), index=False)

print(f"done\nSaved: {model_file}, reduction_final.pkl, scaler_final.pkl, variance_threshold_final.pkl, reduction_metadata.csv")
