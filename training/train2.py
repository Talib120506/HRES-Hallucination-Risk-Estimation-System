# import sys
# import io
# import os
# # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_auc_score, confusion_matrix,
# )
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# import joblib

# # Setup directories
# BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# DATA_DIR   = os.path.join(BASE_DIR, "data")
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(os.path.join(DATA_DIR, "training_data"), exist_ok=True)

# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)

# # ============================================================================
# # 1. LOAD DATA
# # ============================================================================
# df = pd.read_excel(os.path.join(DATA_DIR, "processed", "features_shuffled_final.xlsx"))

# meta_cols    = ["question_id", "doc_id", "question", "answer", "answer_type"]
# feature_cols = [c for c in df.columns if c.startswith("v_")] + ["seq_len", "target_index"]
# X    = df[feature_cols].values.astype(np.float64)
# y    = df["label"].values.astype(int)
# meta = df[meta_cols]

# print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features  |  label 0: {(y==0).sum()}, label 1: {(y==1).sum()}")

# # ============================================================================
# # 2. PREPROCESSING: Variance Threshold + StandardScaler
# # ============================================================================
# vt = VarianceThreshold(threshold=1e-6)
# X_vt = vt.fit_transform(X)

# scaler   = StandardScaler()
# X_scaled = scaler.fit_transform(X_vt)

# print(f"After preprocessing: {X_vt.shape[1]} features (removed {X.shape[1] - X_vt.shape[1]} low-variance)")

# # ============================================================================
# # 3. PCA VARIANCE ANALYSIS
# # ============================================================================
# full_pca = PCA(random_state=RANDOM_STATE)
# full_pca.fit(X_scaled)
# cumvar = np.cumsum(full_pca.explained_variance_ratio_)
# n95 = np.searchsorted(cumvar, 0.95) + 1
# n99 = np.searchsorted(cumvar, 0.99) + 1
# print(f"PCA: 95% variance -> {n95} components, 99% -> {n99} components")

# # ============================================================================
# # 4. GPU CHECK
# # ============================================================================
# try:
#     test_m = XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
#     test_m.fit(X_scaled[:10, :10], y[:10])
#     USE_GPU = True
#     print("GPU: CUDA detected")
# except Exception:
#     USE_GPU = False
#     print("GPU: not available, using CPU")

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# # ============================================================================
# # 5. DEFINE REDUCTION METHODS
# # ============================================================================
# # Each method is a plain factory function that returns a fresh fitted/unfitted
# # reducer.  No lambda tricks — avoids the n_components keyword bug.
# #
# # PCA:      unsupervised, outputs n components  → XGBoost sees 150-300 features
# # PCA_LDA:  supervised,  outputs 1 component   → useful but limited ceiling
# #
# # We run both so you can compare directly.  PCA will almost always give XGBoost
# # more signal; PCA_LDA gives the most class-discriminative single axis.
# # ============================================================================

# def make_pca(n):
#     return PCA(n_components=n, random_state=RANDOM_STATE)

# def make_pca_lda(n):
#     return Pipeline([
#         ("pca", PCA(n_components=n, random_state=RANDOM_STATE)),
#         ("lda", LDA(n_components=1)),          # binary → always 1 output dim
#     ])

# reduction_methods = {
#     # ── Pure PCA: gives XGBoost 150-300 features to work with ────────────────
#     # This is the critical one to include. XGBoost can build real decision trees
#     # across hundreds of PCA components. Without this, XGBoost with PCA_LDA is
#     # just a threshold on a single number.
#     "PCA": {
#         "factory":    make_pca,
#         "n_list":     [150, 200, 250, 300],
#         "supervised": False,
#     },

#     # ── PCA → LDA: collapses to 1 dimension, but maximally discriminative ────
#     # Useful as a strong baseline and often wins for SVM (which works well in
#     # low-dimensional spaces). XGBoost usually does better with plain PCA.
#     "PCA_LDA": {
#         "factory":    make_pca_lda,
#         "n_list":     [150, 200, 250, 300],
#         "supervised": True,
#     },
# }

# print(f"\nTesting reduction methods: {list(reduction_methods.keys())}")

# # ============================================================================
# # 6. CLASSIFIER CONFIGS
# # ============================================================================
# xgb_configs = [
#     {"max_depth": 1, "learning_rate": 0.005, "n_estimators": 1000,
#      "min_child_weight": 10, "subsample": 0.5, "colsample_bytree": 0.5,
#      "reg_alpha": 2.0, "reg_lambda": 5.0, "gamma": 0.5},
#     {"max_depth": 2, "learning_rate": 0.005, "n_estimators": 1000,
#      "min_child_weight": 8,  "subsample": 0.5, "colsample_bytree": 0.6,
#      "reg_alpha": 2.0, "reg_lambda": 4.0, "gamma": 0.4},
# ]

# svm_configs = [
#     {"C": 1,   "gamma": "scale"},
#     {"C": 10,  "gamma": "scale"},
#     {"C": 50,  "gamma": "scale"},
#     {"C": 100, "gamma": "scale"},
#     {"C": 1,   "gamma": "auto"},
#     {"C": 10,  "gamma": "auto"},
#     {"C": 50,  "gamma": "auto"},
#     {"C": 100, "gamma": "auto"},
# ]

# # ============================================================================
# # 7. GRID SEARCH
# # ============================================================================
# best_overall_score     = -1
# best_overall_reduction = None
# best_overall_n         = None
# best_overall_clf       = None
# best_overall_params    = None
# results_summary        = []

# for red_name, red_cfg in reduction_methods.items():
#     factory    = red_cfg["factory"]
#     n_list     = red_cfg["n_list"]
#     supervised = red_cfg["supervised"]

#     print(f"\n{'='*70}\nTESTING {red_name}\n{'='*70}")

#     # ── XGBoost ───────────────────────────────────────────────────────────────
#     print(f"{red_name} + XGBoost grid search...", end=" ", flush=True)
#     best_xgb_score, best_xgb_n, best_xgb_params = -1, None, None

#     for n in n_list:
#         for xgb_p in xgb_configs:
#             fold_accs = []
#             for tr_idx, vl_idx in skf.split(X_scaled, y):
#                 reducer = factory(n)
#                 X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
#                        else reducer.fit_transform(X_scaled[tr_idx])
#                 X_vl = reducer.transform(X_scaled[vl_idx])

#                 mdl = XGBClassifier(
#                     device="cuda" if USE_GPU else "cpu",
#                     eval_metric="logloss", early_stopping_rounds=80,
#                     random_state=RANDOM_STATE, verbosity=0, **xgb_p,
#                 )
#                 mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)
#                 fold_accs.append(accuracy_score(y[vl_idx], mdl.predict(X_vl)))

#             mean_acc = np.mean(fold_accs)
#             if mean_acc > best_xgb_score:
#                 best_xgb_score, best_xgb_n, best_xgb_params = mean_acc, n, xgb_p

#     print(f"done  |  n={best_xgb_n}, depth={best_xgb_params['max_depth']}, Acc={best_xgb_score:.4f}")
#     results_summary.append({"Reduction": red_name, "Classifier": "XGBoost",
#                              "n_components": best_xgb_n, "Accuracy": best_xgb_score})
#     if best_xgb_score > best_overall_score:
#         best_overall_score, best_overall_reduction = best_xgb_score, red_name
#         best_overall_n, best_overall_clf, best_overall_params = best_xgb_n, "XGBoost", best_xgb_params

#     # ── SVM ───────────────────────────────────────────────────────────────────
#     print(f"{red_name} + SVM grid search...", end=" ", flush=True)
#     best_svm_score, best_svm_n, best_svm_params = -1, None, None

#     for n in n_list:
#         for svm_p in svm_configs:
#             fold_accs = []
#             for tr_idx, vl_idx in skf.split(X_scaled, y):
#                 reducer = factory(n)
#                 X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
#                        else reducer.fit_transform(X_scaled[tr_idx])
#                 X_vl = reducer.transform(X_scaled[vl_idx])

#                 svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **svm_p)
#                 svm.fit(X_tr, y[tr_idx])
#                 fold_accs.append(accuracy_score(y[vl_idx], svm.predict(X_vl)))

#             mean_acc = np.mean(fold_accs)
#             if mean_acc > best_svm_score:
#                 best_svm_score, best_svm_n, best_svm_params = mean_acc, n, svm_p

#     print(f"done  |  n={best_svm_n}, C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, Acc={best_svm_score:.4f}")
#     results_summary.append({"Reduction": red_name, "Classifier": "SVM",
#                              "n_components": best_svm_n, "Accuracy": best_svm_score})
#     if best_svm_score > best_overall_score:
#         best_overall_score, best_overall_reduction = best_svm_score, red_name
#         best_overall_n, best_overall_clf, best_overall_params = best_svm_n, "SVM", best_svm_params

# # ============================================================================
# # 8. RESULTS SUMMARY
# # ============================================================================
# print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
# results_df = pd.DataFrame(results_summary).sort_values("Accuracy", ascending=False)
# print(results_df.to_string(index=False))

# print(f"\n{'='*70}")
# print(f"WINNER: {best_overall_reduction} + {best_overall_clf}")
# print(f"n_components: {best_overall_n}  |  Accuracy: {best_overall_score:.4f}")
# print(f"{'='*70}")

# # ============================================================================
# # 9. DETAILED EVALUATION OF WINNER
# # ============================================================================
# print(f"\nDetailed evaluation: {best_overall_reduction} + {best_overall_clf}...")

# red_cfg    = reduction_methods[best_overall_reduction]
# factory    = red_cfg["factory"]
# supervised = red_cfg["supervised"]

# # Fit final reducer on ALL data (for saving + reduced data export)
# final_reducer = factory(best_overall_n)
# if supervised:
#     X_reduced_full = final_reducer.fit_transform(X_scaled, y)
# else:
#     X_reduced_full = final_reducer.fit_transform(X_scaled)

# # Export reduced training data
# n_out_cols = X_reduced_full.shape[1]
# col_names  = [f"{best_overall_reduction}_{i+1}" for i in range(n_out_cols)]
# df_reduced = pd.concat([
#     meta.reset_index(drop=True),
#     pd.DataFrame({"label": y}),
#     pd.DataFrame(X_reduced_full, columns=col_names),
# ], axis=1)
# df_reduced.to_excel(
#     os.path.join(DATA_DIR, "training_data", "training_data_final.xlsx"), index=False
# )

# # Cross-validated metrics for the winner
# fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
# all_y_true, all_y_pred, all_y_proba = [], [], []

# for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_scaled, y), 1):
#     print(f"    Training fold {fold}/5 on {best_overall_reduction}({best_overall_n}):")
#     reducer = factory(best_overall_n)
#     X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
#            else reducer.fit_transform(X_scaled[tr_idx])
#     X_vl = reducer.transform(X_scaled[vl_idx])

#     if best_overall_clf == "XGBoost":
#         mdl = XGBClassifier(
#             device="cuda" if USE_GPU else "cpu",
#             eval_metric="logloss", early_stopping_rounds=80,
#             random_state=RANDOM_STATE, verbosity=1, **best_overall_params,
#         )
#         mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=10)
#     else:
#         mdl = SVC(kernel="rbf", probability=True,
#                   random_state=RANDOM_STATE, **best_overall_params)
#         mdl.fit(X_tr, y[tr_idx])

#     y_pred  = mdl.predict(X_vl)
#     y_proba = mdl.predict_proba(X_vl)[:, 1]

#     fold_metrics["accuracy"].append(accuracy_score(y[vl_idx], y_pred))
#     fold_metrics["precision"].append(precision_score(y[vl_idx], y_pred, zero_division=0))
#     fold_metrics["recall"].append(recall_score(y[vl_idx], y_pred, zero_division=0))
#     fold_metrics["f1"].append(f1_score(y[vl_idx], y_pred, zero_division=0))
#     fold_metrics["roc_auc"].append(roc_auc_score(y[vl_idx], y_proba))

#     all_y_true.extend(y[vl_idx])
#     all_y_pred.extend(y_pred)
#     all_y_proba.extend(y_proba)

# cm = confusion_matrix(all_y_true, all_y_pred)
# print(f"\n{best_overall_clf} ({best_overall_reduction}={best_overall_n}):"
#       f"  Acc={np.mean(fold_metrics['accuracy']):.4f}"
#       f"  Prec={np.mean(fold_metrics['precision']):.4f}"
#       f"  Rec={np.mean(fold_metrics['recall']):.4f}"
#       f"  F1={np.mean(fold_metrics['f1']):.4f}"
#       f"  AUC={np.mean(fold_metrics['roc_auc']):.4f}")
# print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

# # ============================================================================
# # 10. TRAIN FINAL MODEL ON ALL DATA & SAVE
# # ============================================================================
# print(f"\nTraining final {best_overall_clf} on all data...", end=" ", flush=True)

# if best_overall_clf == "XGBoost":
#     # Estimate best n_estimators via CV then retrain on full data
#     best_iters = []
#     for tr_idx, vl_idx in skf.split(X_scaled, y):
#         reducer = factory(best_overall_n)
#         X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
#                else reducer.fit_transform(X_scaled[tr_idx])
#         X_vl = reducer.transform(X_scaled[vl_idx])
#         mdl = XGBClassifier(
#             device="cuda" if USE_GPU else "cpu",
#             eval_metric="logloss", early_stopping_rounds=80,
#             random_state=RANDOM_STATE, verbosity=0, **best_overall_params,
#         )
#         mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)
#         best_iters.append(mdl.best_iteration + 1)

#     final_n_trees = int(np.mean(best_iters))
#     final_model = XGBClassifier(
#         device="cuda" if USE_GPU else "cpu",
#         n_estimators=final_n_trees, eval_metric="logloss",
#         random_state=RANDOM_STATE, verbosity=0,
#         **{k: v for k, v in best_overall_params.items() if k != "n_estimators"},
#     )
#     final_model.fit(X_reduced_full, y)
#     joblib.dump(final_model, os.path.join(MODELS_DIR, "xgb_model_final.pkl"))
# else:
#     final_model = SVC(kernel="rbf", probability=True,
#                       random_state=RANDOM_STATE, **best_overall_params)
#     final_model.fit(X_reduced_full, y)
#     joblib.dump(final_model, os.path.join(MODELS_DIR, "svm_model_final.pkl"))

# # Save reducer + preprocessors
# joblib.dump(final_reducer, os.path.join(MODELS_DIR, "reduction_final.pkl"))
# joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler_final.pkl"))
# joblib.dump(vt,            os.path.join(MODELS_DIR, "variance_threshold_final.pkl"))

# # Save metadata so app.py knows which reduction method was used
# pd.DataFrame([{
#     "method":       best_overall_reduction,
#     "n_components": best_overall_n,
#     "n_output_dims": n_out_cols,
#     "classifier":   best_overall_clf,
#     "accuracy":     best_overall_score,
# }]).to_csv(os.path.join(MODELS_DIR, "reduction_metadata.csv"), index=False)

# model_file = "xgb_model_final.pkl" if best_overall_clf == "XGBoost" else "svm_model_final.pkl"
# print(f"done\nSaved: {model_file}, reduction_final.pkl, scaler_final.pkl, variance_threshold_final.pkl, reduction_metadata.csv")


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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
    # "PCA_LDA": {
    #     "factory":    make_pca_lda,
    #     "n_list":     [150, 200, 250, 300],
    #     "supervised": True,
    # },
}

print(f"\nTesting reduction methods: {list(reduction_methods.keys())}")

# ============================================================================
# 6. CLASSIFIER CONFIGS
# ============================================================================
# xgb_configs = [
#     {"max_depth": 1, "learning_rate": 0.005, "n_estimators": 1000,
#      "min_child_weight": 10, "subsample": 0.5, "colsample_bytree": 0.5,
#      "reg_alpha": 2.0, "reg_lambda": 5.0, "gamma": 0.5},
#     {"max_depth": 2, "learning_rate": 0.005, "n_estimators": 1000,
#      "min_child_weight": 8,  "subsample": 0.5, "colsample_bytree": 0.6,
#      "reg_alpha": 2.0, "reg_lambda": 4.0, "gamma": 0.4},
# ]

# svm_configs = [
#     {"C": 1,   "gamma": "scale"},
#     {"C": 10,  "gamma": "scale"},
#     {"C": 50,  "gamma": "scale"},
#     {"C": 100, "gamma": "scale"},
#     {"C": 1,   "gamma": "auto"},
#     {"C": 10,  "gamma": "auto"},
#     {"C": 50,  "gamma": "auto"},
#     {"C": 100, "gamma": "auto"},
# ]

# Logistic Regression: L2 regularisation over a strong penalty range
lr_configs = [
    {"C": 0.01,  "max_iter": 1000},
    {"C": 0.1,   "max_iter": 1000},
    {"C": 1.0,   "max_iter": 1000},
    {"C": 10.0,  "max_iter": 1000},
]

# MLP: small networks to avoid overfitting on ~950 samples
# early_stopping + high alpha act as regularisers in place of explicit dropout
# mlp_configs = [
#     {"hidden_layer_sizes": (64,),       "alpha": 0.01,  "max_iter": 500},
#     {"hidden_layer_sizes": (128,),      "alpha": 0.01,  "max_iter": 500},
#     {"hidden_layer_sizes": (64, 32),    "alpha": 0.01,  "max_iter": 500},
#     {"hidden_layer_sizes": (128, 64),   "alpha": 0.01,  "max_iter": 500},
#     {"hidden_layer_sizes": (64,),       "alpha": 0.1,   "max_iter": 500},
#     {"hidden_layer_sizes": (128, 64),   "alpha": 0.1,   "max_iter": 500},
# ]

# Random Forest: naturally handles high-dimensional PCA spaces well
# n_jobs=-1 to use all cores; vary depth and leaf size for regularisation
# rf_configs = [
#     {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1,  "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2,  "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 4,  "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": 10,   "min_samples_leaf": 1,  "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": 10,   "min_samples_leaf": 2,  "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1,  "max_features": "log2"},
#     {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2,  "max_features": "log2"},
# ]

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

    # # ── XGBoost ───────────────────────────────────────────────────────────────
    # print(f"{red_name} + XGBoost grid search...", end=" ", flush=True)
    # best_xgb_score, best_xgb_n, best_xgb_params = -1, None, None

    # for n in n_list:
    #     for xgb_p in xgb_configs:
    #         fold_accs = []
    #         for tr_idx, vl_idx in skf.split(X_scaled, y):
    #             reducer = factory(n)
    #             X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
    #                    else reducer.fit_transform(X_scaled[tr_idx])
    #             X_vl = reducer.transform(X_scaled[vl_idx])

    #             mdl = XGBClassifier(
    #                 device="cuda" if USE_GPU else "cpu",
    #                 eval_metric="logloss", early_stopping_rounds=80,
    #                 random_state=RANDOM_STATE, verbosity=0, **xgb_p,
    #             )
    #             mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)
    #             fold_accs.append(accuracy_score(y[vl_idx], mdl.predict(X_vl)))

    #         mean_acc = np.mean(fold_accs)
    #         if mean_acc > best_xgb_score:
    #             best_xgb_score, best_xgb_n, best_xgb_params = mean_acc, n, xgb_p

    # print(f"done  |  n={best_xgb_n}, depth={best_xgb_params['max_depth']}, Acc={best_xgb_score:.4f}")
    # results_summary.append({"Reduction": red_name, "Classifier": "XGBoost",
    #                          "n_components": best_xgb_n, "Accuracy": best_xgb_score})
    # if best_xgb_score > best_overall_score:
    #     best_overall_score, best_overall_reduction = best_xgb_score, red_name
    #     best_overall_n, best_overall_clf, best_overall_params = best_xgb_n, "XGBoost", best_xgb_params

    # # ── SVM ───────────────────────────────────────────────────────────────────
    # print(f"{red_name} + SVM grid search...", end=" ", flush=True)
    # best_svm_score, best_svm_n, best_svm_params = -1, None, None

    # for n in n_list:
    #     for svm_p in svm_configs:
    #         fold_accs = []
    #         for tr_idx, vl_idx in skf.split(X_scaled, y):
    #             reducer = factory(n)
    #             X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
    #                    else reducer.fit_transform(X_scaled[tr_idx])
    #             X_vl = reducer.transform(X_scaled[vl_idx])

    #             svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **svm_p)
    #             svm.fit(X_tr, y[tr_idx])
    #             fold_accs.append(accuracy_score(y[vl_idx], svm.predict(X_vl)))

    #         mean_acc = np.mean(fold_accs)
    #         if mean_acc > best_svm_score:
    #             best_svm_score, best_svm_n, best_svm_params = mean_acc, n, svm_p

    # print(f"done  |  n={best_svm_n}, C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, Acc={best_svm_score:.4f}")
    # results_summary.append({"Reduction": red_name, "Classifier": "SVM",
    #                          "n_components": best_svm_n, "Accuracy": best_svm_score})
    # if best_svm_score > best_overall_score:
    #     best_overall_score, best_overall_reduction = best_svm_score, red_name
    #     best_overall_n, best_overall_clf, best_overall_params = best_svm_n, "SVM", best_svm_params

    # ── Logistic Regression ───────────────────────────────────────────────────
    print(f"{red_name} + LogisticRegression grid search...", end=" ", flush=True)
    best_lr_score, best_lr_n, best_lr_params = -1, None, None

    for n in n_list:
        for lr_p in lr_configs:
            fold_accs = []
            for tr_idx, vl_idx in skf.split(X_scaled, y):
                reducer = factory(n)
                X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
                       else reducer.fit_transform(X_scaled[tr_idx])
                X_vl = reducer.transform(X_scaled[vl_idx])

                lr = LogisticRegression(
                    solver="lbfgs", penalty="l2",
                    random_state=RANDOM_STATE, **lr_p
                )
                lr.fit(X_tr, y[tr_idx])
                fold_accs.append(accuracy_score(y[vl_idx], lr.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_lr_score:
                best_lr_score, best_lr_n, best_lr_params = mean_acc, n, lr_p

    print(f"done  |  n={best_lr_n}, C={best_lr_params['C']}, Acc={best_lr_score:.4f}")
    results_summary.append({"Reduction": red_name, "Classifier": "LogisticRegression",
                             "n_components": best_lr_n, "Accuracy": best_lr_score})
    if best_lr_score > best_overall_score:
        best_overall_score, best_overall_reduction = best_lr_score, red_name
        best_overall_n, best_overall_clf, best_overall_params = best_lr_n, "LogisticRegression", best_lr_params

    # # ── MLP ───────────────────────────────────────────────────────────────────
    # print(f"{red_name} + MLP grid search...", end=" ", flush=True)
    # best_mlp_score, best_mlp_n, best_mlp_params = -1, None, None

    # for n in n_list:
    #     for mlp_p in mlp_configs:
    #         fold_accs = []
    #         for tr_idx, vl_idx in skf.split(X_scaled, y):
    #             reducer = factory(n)
    #             X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
    #                    else reducer.fit_transform(X_scaled[tr_idx])
    #             X_vl = reducer.transform(X_scaled[vl_idx])

    #             mlp = MLPClassifier(
    #                 activation="relu",
    #                 solver="adam",
    #                 early_stopping=True,
    #                 validation_fraction=0.1,
    #                 n_iter_no_change=20,
    #                 random_state=RANDOM_STATE,
    #                 **mlp_p,
    #             )
    #             mlp.fit(X_tr, y[tr_idx])
    #             fold_accs.append(accuracy_score(y[vl_idx], mlp.predict(X_vl)))

    #         mean_acc = np.mean(fold_accs)
    #         if mean_acc > best_mlp_score:
    #             best_mlp_score, best_mlp_n, best_mlp_params = mean_acc, n, mlp_p

    # print(f"done  |  n={best_mlp_n}, layers={best_mlp_params['hidden_layer_sizes']}, alpha={best_mlp_params['alpha']}, Acc={best_mlp_score:.4f}")
    # results_summary.append({"Reduction": red_name, "Classifier": "MLP",
    #                          "n_components": best_mlp_n, "Accuracy": best_mlp_score})
    # if best_mlp_score > best_overall_score:
    #     best_overall_score, best_overall_reduction = best_mlp_score, red_name
    #     best_overall_n, best_overall_clf, best_overall_params = best_mlp_n, "MLP", best_mlp_params

    # # ── Random Forest ─────────────────────────────────────────────────────────
    # print(f"{red_name} + RandomForest grid search...", end=" ", flush=True)
    # best_rf_score, best_rf_n, best_rf_params = -1, None, None

    # for n in n_list:
    #     for rf_p in rf_configs:
    #         fold_accs = []
    #         for tr_idx, vl_idx in skf.split(X_scaled, y):
    #             reducer = factory(n)
    #             X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
    #                    else reducer.fit_transform(X_scaled[tr_idx])
    #             X_vl = reducer.transform(X_scaled[vl_idx])

    #             rf = RandomForestClassifier(
    #                 random_state=RANDOM_STATE, n_jobs=-1, **rf_p
    #             )
    #             rf.fit(X_tr, y[tr_idx])
    #             fold_accs.append(accuracy_score(y[vl_idx], rf.predict(X_vl)))

    #         mean_acc = np.mean(fold_accs)
    #         if mean_acc > best_rf_score:
    #             best_rf_score, best_rf_n, best_rf_params = mean_acc, n, rf_p

    # print(f"done  |  n={best_rf_n}, depth={best_rf_params['max_depth']}, min_leaf={best_rf_params['min_samples_leaf']}, Acc={best_rf_score:.4f}")
    # results_summary.append({"Reduction": red_name, "Classifier": "RandomForest",
    #                          "n_components": best_rf_n, "Accuracy": best_rf_score})
    # if best_rf_score > best_overall_score:
    #     best_overall_score, best_overall_reduction = best_rf_score, red_name
    #     best_overall_n, best_overall_clf, best_overall_params = best_rf_n, "RandomForest", best_rf_params

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
    # if best_overall_clf == "XGBoost":
    #     return XGBClassifier(
    #         device="cuda" if USE_GPU else "cpu",
    #         eval_metric="logloss", early_stopping_rounds=80,
    #         random_state=RANDOM_STATE, verbosity=1, **params,
    #     )
    # elif best_overall_clf == "SVM":
    #     return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **params)
    elif best_overall_clf == "LogisticRegression":
        return LogisticRegression(solver="lbfgs", penalty="l2",
                                  random_state=RANDOM_STATE, **params)
    # elif best_overall_clf == "MLP":
    #     return MLPClassifier(
    #         activation="relu", solver="adam",
    #         early_stopping=True, validation_fraction=0.1,
    #         n_iter_no_change=20, random_state=RANDOM_STATE, **params,
    #     )
    # elif best_overall_clf == "RandomForest":
    #     return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown classifier: {best_overall_clf}")

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"    Training fold {fold}/5 on {best_overall_reduction}({best_overall_n}):")
    reducer = factory(best_overall_n)
    X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
           else reducer.fit_transform(X_scaled[tr_idx])
    X_vl = reducer.transform(X_scaled[vl_idx])

    mdl = build_winner(best_overall_params)
    # if best_overall_clf == "XGBoost":
    #     mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=10)
    # else:
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
    # "XGBoost":            "xgb_model_final.pkl",
    # "SVM":                "svm_model_final.pkl",
    "LogisticRegression": "lr_model_final.pkl",
    # "MLP":                "mlp_model_final.pkl",
    # "RandomForest":       "rf_model_final.pkl",
}

# if best_overall_clf == "XGBoost":
#     # Estimate best n_estimators via CV then retrain on full data without eval set
#     best_iters = []
#     for tr_idx, vl_idx in skf.split(X_scaled, y):
#         reducer = factory(best_overall_n)
#         X_tr = reducer.fit_transform(X_scaled[tr_idx], y[tr_idx]) if supervised \
#                else reducer.fit_transform(X_scaled[tr_idx])
#         X_vl = reducer.transform(X_scaled[vl_idx])
#         mdl = XGBClassifier(
#             device="cuda" if USE_GPU else "cpu",
#             eval_metric="logloss", early_stopping_rounds=80,
#             random_state=RANDOM_STATE, verbosity=0, **best_overall_params,
#         )
#         mdl.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)
#         best_iters.append(mdl.best_iteration + 1)

#     final_n_trees = int(np.mean(best_iters))
#     final_model = XGBClassifier(
#         device="cuda" if USE_GPU else "cpu",
#         n_estimators=final_n_trees, eval_metric="logloss",
#         random_state=RANDOM_STATE, verbosity=0,
#         **{k: v for k, v in best_overall_params.items() if k != "n_estimators"},
#     )
#     final_model.fit(X_reduced_full, y)
# else:
final_model = build_winner(best_overall_params)
# MLP: disable early_stopping for final fit (no held-out set available)
# if best_overall_clf == "MLP":
#     final_model.set_params(early_stopping=False)
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