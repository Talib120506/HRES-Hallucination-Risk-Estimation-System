import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
df = pd.read_excel("features_shuffled_final.xlsx")

meta_cols = ["question_id", "doc_id", "question", "answer", "answer_type"]
feature_cols = [c for c in df.columns if c.startswith("v_")] + ["seq_len", "target_index"]
X = df[feature_cols].values.astype(np.float64)
y = df["label"].values.astype(int)
meta = df[meta_cols]

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features  |  label 0: {(y==0).sum()}, label 1: {(y==1).sum()}")

# ============================================================================
# 2. PREPROCESSING: Variance Threshold + StandardScaler
# ============================================================================
vt = VarianceThreshold(threshold=1e-6)
X_vt = vt.fit_transform(X)

scaler = StandardScaler()
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
try:
    test_m = XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
    test_m.fit(X_scaled[:10, :10], y[:10])
    USE_GPU = True
    print("GPU: CUDA detected")
except Exception:
    USE_GPU = False
    print("GPU: not available, using CPU")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ============================================================================
# 5. XGBOOST GRID SEARCH
# ============================================================================
print("\nXGBoost grid search (5-fold CV)...", end=" ", flush=True)

pca_candidates = [150, 200, 250, 300]
param_configs = [
    {"max_depth": 1, "learning_rate": 0.005, "n_estimators": 1000,
     "min_child_weight": 10, "subsample": 0.5, "colsample_bytree": 0.5,
     "reg_alpha": 2.0, "reg_lambda": 5.0, "gamma": 0.5},
    {"max_depth": 2, "learning_rate": 0.005, "n_estimators": 1000,
     "min_child_weight": 8, "subsample": 0.5, "colsample_bytree": 0.6,
     "reg_alpha": 2.0, "reg_lambda": 4.0, "gamma": 0.4},
]

best_xgb_score = -1
best_xgb_pca_n = None
best_xgb_params = None
best_xgb_folds = None

total_combos = len(pca_candidates) * len(param_configs)
combo = 0

for pca_n in pca_candidates:
    pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    for params in param_configs:
        combo += 1
        fold_accs = []

        for train_idx, val_idx in skf.split(X_pca, y):
            X_tr, X_vl = X_pca[train_idx], X_pca[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            mdl = XGBClassifier(
                device="cuda" if USE_GPU else "cpu",
                eval_metric="logloss",
                early_stopping_rounds=80,
                random_state=RANDOM_STATE,
                verbosity=0,
                **params,
            )
            mdl.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            fold_accs.append(accuracy_score(y_vl, mdl.predict(X_vl)))

        mean_acc = np.mean(fold_accs)
        if mean_acc > best_xgb_score:
            best_xgb_score = mean_acc
            best_xgb_pca_n = pca_n
            best_xgb_params = params
            best_xgb_folds = fold_accs

print(f"done  |  Best: PCA={best_xgb_pca_n}, depth={best_xgb_params['max_depth']}, Acc={best_xgb_score:.4f}")

# ============================================================================
# 6. SVM GRID SEARCH
# ============================================================================
print("SVM grid search (5-fold CV)...", end=" ", flush=True)

svm_configs = [
    {"C": 1,   "gamma": "scale"},
    {"C": 10,  "gamma": "scale"},
    {"C": 50,  "gamma": "scale"},
    {"C": 100, "gamma": "scale"},
    {"C": 1,   "gamma": "auto"},
    {"C": 10,  "gamma": "auto"},
    {"C": 50,  "gamma": "auto"},
    {"C": 100, "gamma": "auto"},
]

best_svm_score = -1
best_svm_pca_n = None
best_svm_params = None
best_svm_folds = None

total_svm = len(pca_candidates) * len(svm_configs)
combo = 0

for pca_n in pca_candidates:
    pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    for svm_p in svm_configs:
        combo += 1
        fold_accs = []

        for train_idx, val_idx in skf.split(X_pca, y):
            X_tr, X_vl = X_pca[train_idx], X_pca[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            svm = SVC(kernel="rbf", C=svm_p["C"], gamma=svm_p["gamma"],
                      probability=True, random_state=RANDOM_STATE)
            svm.fit(X_tr, y_tr)
            fold_accs.append(accuracy_score(y_vl, svm.predict(X_vl)))

        mean_acc = np.mean(fold_accs)
        if mean_acc > best_svm_score:
            best_svm_score = mean_acc
            best_svm_pca_n = pca_n
            best_svm_params = svm_p
            best_svm_folds = fold_accs

print(f"done  |  Best: PCA={best_svm_pca_n}, C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, Acc={best_svm_score:.4f}")

# ============================================================================
# 7. HEAD-TO-HEAD COMPARISON
# ============================================================================

# Pick the winner
if best_svm_score > best_xgb_score:
    WINNER = "SVM"
    winner_pca_n = best_svm_pca_n
elif best_xgb_score > best_svm_score:
    WINNER = "XGBoost"
    winner_pca_n = best_xgb_pca_n
else:
    WINNER = "XGBoost"
    winner_pca_n = best_xgb_pca_n

print(f"\nWinner: {WINNER}  |  XGBoost={best_xgb_score:.4f} vs SVM={best_svm_score:.4f}")

# ============================================================================
# 8. DETAILED EVALUATION OF BOTH MODELS
# ============================================================================
pca_final = PCA(n_components=winner_pca_n, random_state=RANDOM_STATE)
X_pca_final = pca_final.fit_transform(X_scaled)

# Save PCA-transformed training data
pca_col_names = [f"PC_{i+1}" for i in range(winner_pca_n)]
df_pca = pd.concat([
    meta.reset_index(drop=True),
    pd.DataFrame({"label": y}),
    pd.DataFrame(X_pca_final, columns=pca_col_names)
], axis=1)
df_pca.to_excel("training_data_final.xlsx", index=False)

for model_name in ["XGBoost", "SVM"]:
    if model_name == "XGBoost":
        use_pca_n = best_xgb_pca_n
    else:
        use_pca_n = best_svm_pca_n

    pca_eval = PCA(n_components=use_pca_n, random_state=RANDOM_STATE)
    X_pca_eval = pca_eval.fit_transform(X_scaled)

    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca_eval, y), 1):
        X_tr, X_vl = X_pca_eval[train_idx], X_pca_eval[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        if model_name == "XGBoost":
            mdl = XGBClassifier(
                device="cuda" if USE_GPU else "cpu",
                eval_metric="logloss",
                early_stopping_rounds=80,
                random_state=RANDOM_STATE,
                verbosity=0,
                **best_xgb_params,
            )
            mdl.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        else:
            mdl = SVC(kernel="rbf", C=best_svm_params["C"],
                      gamma=best_svm_params["gamma"],
                      probability=True, random_state=RANDOM_STATE)
            mdl.fit(X_tr, y_tr)

        y_pred = mdl.predict(X_vl)
        y_proba = mdl.predict_proba(X_vl)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_vl, y_pred))
        fold_metrics["precision"].append(precision_score(y_vl, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_vl, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_vl, y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_vl, y_proba))

        all_y_true.extend(y_vl)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n{model_name} (PCA={use_pca_n}):"
          f"  Acc={np.mean(fold_metrics['accuracy']):.4f}"
          f"  Prec={np.mean(fold_metrics['precision']):.4f}"
          f"  Rec={np.mean(fold_metrics['recall']):.4f}"
          f"  F1={np.mean(fold_metrics['f1']):.4f}"
          f"  AUC={np.mean(fold_metrics['roc_auc']):.4f}")
    print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

# ============================================================================
# 9. TRAIN FINAL WINNER MODEL ON ALL DATA & SAVE
# ============================================================================
print(f"\nTraining final {WINNER} on all data...", end=" ", flush=True)

if WINNER == "XGBoost":
    best_iters = []
    pca_save = PCA(n_components=best_xgb_pca_n, random_state=RANDOM_STATE)
    X_save = pca_save.fit_transform(X_scaled)

    for train_idx, val_idx in skf.split(X_save, y):
        mdl = XGBClassifier(
            device="cuda" if USE_GPU else "cpu",
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=RANDOM_STATE,
            verbosity=0,
            **best_xgb_params,
        )
        mdl.fit(X_save[train_idx], y[train_idx],
                eval_set=[(X_save[val_idx], y[val_idx])], verbose=False)
        best_iters.append(mdl.best_iteration + 1)

    final_n_trees = int(np.mean(best_iters))

    final_model = XGBClassifier(
        device="cuda" if USE_GPU else "cpu",
        n_estimators=final_n_trees,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
    )
    final_model.fit(X_save, y)
    joblib.dump(final_model, "xgb_model_final.pkl")
    joblib.dump(pca_save, "pca_final.pkl")
else:
    pca_save = PCA(n_components=best_svm_pca_n, random_state=RANDOM_STATE)
    X_save = pca_save.fit_transform(X_scaled)

    final_model = SVC(kernel="rbf", C=best_svm_params["C"],
                      gamma=best_svm_params["gamma"],
                      probability=True, random_state=RANDOM_STATE)
    final_model.fit(X_save, y)
    joblib.dump(final_model, "svm_model_final.pkl")
    joblib.dump(pca_save, "pca_final.pkl")

joblib.dump(scaler, "scaler_final.pkl")
joblib.dump(vt, "variance_threshold_final.pkl")

model_file = "xgb_model_final.pkl" if WINNER == "XGBoost" else "svm_model_final.pkl"
print(f"done\nSaved: {model_file}, pca_final.pkl, scaler_final.pkl, variance_threshold_final.pkl")
