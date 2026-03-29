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
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "training_data"), exist_ok=True)

# Try to import UMAP (optional dependency)
# try:
#     from umap import UMAP
#     UMAP_AVAILABLE = True
# except ImportError:
#     UMAP_AVAILABLE = False
#     print("WARNING: UMAP not installed. Install with: pip install umap-learn")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
df = pd.read_excel(os.path.join(DATA_DIR, "processed", "features_shuffled_final.xlsx"))

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
# 3. PCA VARIANCE ANALYSIS (for reference)
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
# 5. DEFINE DIMENSIONALITY REDUCTION METHODS TO TEST
# ============================================================================
reduction_methods = {}

# # PCA (unsupervised)
# reduction_methods['PCA'] = {
#     'class': PCA,
#     'n_components': [150, 200, 250, 300],
#     'params': lambda n: {'n_components': n, 'random_state': RANDOM_STATE},
#     'supervised': False
# }

# # LDA (supervised - can use labels)
# # Note: LDA max components = min(n_features, n_classes - 1) = min(2054, 1) = 1
# # So we'll test just 1 component for LDA
# reduction_methods['LDA'] = {
#     'class': LDA,
#     'n_components': [1],  # Binary classification = max 1 component
#     'params': lambda n: {'n_components': n},
#     'supervised': True
# }

def create_pca_lda(n_components):
    return Pipeline([
        ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ('lda', LDA(n_components=1))
    ])

def apply_pca_tsne_reduction(X_train, X_val, y_train, n_components):
    """
    Apply PCA then TSNE. Since TSNE doesn't support transform on new data,
    we fit TSNE on combined data, then extract embeddings for train/val subsets.
    """
    # First: Apply PCA to both train and val
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    # Combine for TSNE (TSNE needs all data to compute consistent embeddings)
    X_combined_pca = np.vstack([X_train_pca, X_val_pca])
    
    # Apply TSNE to combined data
    tsne = TSNE(n_components=1, random_state=RANDOM_STATE, verbose=0)
    X_combined_tsne = tsne.fit_transform(X_combined_pca)
    
    # Split back into train and val
    X_train_reduced = X_combined_tsne[:X_train_pca.shape[0]]
    X_val_reduced = X_combined_tsne[X_train_pca.shape[0]:]
    
    return X_train_reduced, X_val_reduced

# PCA followed by LDA
reduction_methods['PCA_LDA'] = {
    'class': create_pca_lda,
    'n_components': [150, 200, 250, 300],
    'params': lambda n: n,
    'supervised': True,
    'is_pipeline': True
}

# PCA followed by TSNE
reduction_methods['PCA_TSNE'] = {
    'class': apply_pca_tsne_reduction,
    'n_components': [150, 200, 250, 300],
    'params': lambda n: n,
    'supervised': False,
    'is_pipeline': False
}

# # UMAP (unsupervised - manifold learning)
# if UMAP_AVAILABLE:
#     reduction_methods['UMAP'] = {
#         'class': UMAP,
#         'n_components': [150, 200, 250],  # Smaller range for UMAP (slower)
#         'params': lambda n: {'n_components': n, 'random_state': RANDOM_STATE, 'n_neighbors': 15, 'min_dist': 0.1},
#         'supervised': False
#     }

print(f"\nTesting {len(reduction_methods)} dimensionality reduction methods: {list(reduction_methods.keys())}")

# ============================================================================
# 6. CLASSIFIER HYPERPARAMETERS
# ============================================================================
xgb_param_configs = [
    {"max_depth": 1, "learning_rate": 0.005, "n_estimators": 1000,
     "min_child_weight": 10, "subsample": 0.5, "colsample_bytree": 0.5,
     "reg_alpha": 2.0, "reg_lambda": 5.0, "gamma": 0.5},
    {"max_depth": 2, "learning_rate": 0.005, "n_estimators": 1000,
     "min_child_weight": 8, "subsample": 0.5, "colsample_bytree": 0.6,
     "reg_alpha": 2.0, "reg_lambda": 4.0, "gamma": 0.4},
]

svm_param_configs = [
    {"C": 1,   "gamma": "scale"},
    {"C": 10,  "gamma": "scale"},
    {"C": 50,  "gamma": "scale"},
    {"C": 100, "gamma": "scale"},
    {"C": 1,   "gamma": "auto"},
    {"C": 10,  "gamma": "auto"},
    {"C": 50,  "gamma": "auto"},
    {"C": 100, "gamma": "auto"},
]

# ============================================================================
# 7. GRID SEARCH ACROSS ALL COMBINATIONS
# ============================================================================
best_overall_score = -1
best_overall_reduction = None
best_overall_reduction_n = None
best_overall_classifier = None
best_overall_params = None
best_overall_reduction_obj = None

results_summary = []

for reduction_name, reduction_config in reduction_methods.items():
    print(f"\n{'='*80}")
    print(f"TESTING {reduction_name}")
    print(f"{'='*80}")

    reduction_class = reduction_config['class']
    n_components_list = reduction_config['n_components']
    is_supervised = reduction_config['supervised']
    is_pipeline = reduction_config.get('is_pipeline', True)

    best_xgb_score = -1
    best_xgb_n = None
    best_xgb_params = None
    best_xgb_reduction = None

    best_svm_score = -1
    best_svm_n = None
    best_svm_params = None
    best_svm_reduction = None

    for n in n_components_list:
        n_comp = reduction_config['params'](n)
        print(f"  -> n_components = {n}")
        print(f"     Precomputing dimensionality reduction for 5 folds...", end=" ", flush=True)

        # Precompute folds for this n_components to avoid repeating expensive TSNE/PCA computations
        reduced_folds = []
        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr_orig, X_vl_orig = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]
            
            if is_pipeline:
                reducer = reduction_class(n_comp)
                if is_supervised:
                    X_tr = reducer.fit_transform(X_tr_orig, y_tr)
                else:
                    X_tr = reducer.fit_transform(X_tr_orig)
                X_vl = reducer.transform(X_vl_orig)
            else:
                X_tr, X_vl = reduction_class(X_tr_orig, X_vl_orig, y_tr, n_comp)
            
            reduced_folds.append((X_tr, X_vl, y_tr, y_vl))
        print("done.")

        # --- XGBoost ---
        for xgb_params in xgb_param_configs:
            fold_accs = []
            for X_tr, X_vl, y_tr, y_vl in reduced_folds:
                mdl = XGBClassifier(
                    device="cuda" if USE_GPU else "cpu",
                    eval_metric="logloss",
                    early_stopping_rounds=80,
                    random_state=RANDOM_STATE,
                    verbosity=0,
                    **xgb_params,
                )
                mdl.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                fold_accs.append(accuracy_score(y_vl, mdl.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_xgb_score:
                best_xgb_score = mean_acc
                best_xgb_n = n
                best_xgb_params = xgb_params
                best_xgb_reduction = reduction_class(n_comp) if is_pipeline else n_comp

        # --- SVM ---
        for svm_params in svm_param_configs:
            fold_accs = []
            for X_tr, X_vl, y_tr, y_vl in reduced_folds:
                svm = SVC(kernel="rbf", C=svm_params["C"], gamma=svm_params["gamma"],
                          probability=True, random_state=RANDOM_STATE)
                svm.fit(X_tr, y_tr)
                fold_accs.append(accuracy_score(y_vl, svm.predict(X_vl)))

            mean_acc = np.mean(fold_accs)
            if mean_acc > best_svm_score:
                best_svm_score = mean_acc
                best_svm_n = n
                best_svm_params = svm_params
                best_svm_reduction = reduction_class(n_comp) if is_pipeline else n_comp

    results_summary.append({
        'Reduction': reduction_name,
        'Classifier': 'XGBoost',
        'n_components': best_xgb_n,
        'Params': f"depth={best_xgb_params['max_depth']}",
        'Validation Accuracy': f"{best_xgb_score:.6f}"
    })
    
    if best_xgb_score > best_overall_score:
        best_overall_score = best_xgb_score
        best_overall_reduction = reduction_name
        best_overall_reduction_n = best_xgb_n
        best_overall_classifier = 'XGBoost'
        best_overall_params = best_xgb_params
        best_overall_reduction_obj = best_xgb_reduction

    results_summary.append({
        'Reduction': reduction_name,
        'Classifier': 'SVM',
        'n_components': best_svm_n,
        'Params': f"C={best_svm_params['C']}, gamma={best_svm_params['gamma']}",
        'Validation Accuracy': f"{best_svm_score:.6f}"
    })
    
    if best_svm_score > best_overall_score:
        best_overall_score = best_svm_score
        best_overall_reduction = reduction_name
        best_overall_reduction_n = best_svm_n
        best_overall_classifier = 'SVM'
        best_overall_params = best_svm_params
        best_overall_reduction_obj = best_svm_reduction

# ============================================================================
# 8. RESULTS SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("FINAL RESULTS - ALL COMBINATIONS")
print(f"{'='*80}\n")

results_df = pd.DataFrame(results_summary)
results_df['Validation Accuracy'] = results_df['Validation Accuracy'].astype(float)
results_df = results_df.sort_values('Validation Accuracy', ascending=False)
print(results_df.to_string(index=False))

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_overall_reduction} + {best_overall_classifier}")
print(f"Parameters: n_components={best_overall_reduction_n}")
print(f"Validation Accuracy: {best_overall_score:.6f}")
print(f"{'='*80}\n")

# ============================================================================
# 9. DETAILED EVALUATION OF WINNER
# ============================================================================

# Get the reduction config
reduction_config = reduction_methods[best_overall_reduction]
n_comp = reduction_config['params'](best_overall_reduction_n)
is_supervised = reduction_config['supervised']
is_pipeline = reduction_config.get('is_pipeline', True)

# Fit the reduction transformer on full data
if is_pipeline:
    best_overall_reduction_obj = reduction_config['class'](n_comp)
    if is_supervised:
        best_overall_reduction_obj.fit(X_scaled, y)
        X_reduced_full = best_overall_reduction_obj.transform(X_scaled)
    else:
        best_overall_reduction_obj.fit(X_scaled)
        X_reduced_full = best_overall_reduction_obj.transform(X_scaled)
else:
    # For PCA_TSNE, apply it to full data
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=1, random_state=RANDOM_STATE, verbose=0)
    X_reduced_full = tsne.fit_transform(X_pca)
    best_overall_reduction_obj = None  # TSNE doesn't need to be saved

# Save reduced training data
reduction_col_names = [f"{best_overall_reduction}_{i+1}" for i in range(X_reduced_full.shape[1])]
df_reduced = pd.concat([
    meta.reset_index(drop=True),
    pd.DataFrame({"label": y}),
    pd.DataFrame(X_reduced_full, columns=reduction_col_names)
], axis=1)
df_reduced.to_excel(os.path.join(DATA_DIR, "training_data", "training_data_final.xlsx"), index=False)

# Detailed cross-validation
fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
all_y_true, all_y_pred, all_y_proba = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_tr_orig, X_vl_orig = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_vl = y[train_idx], y[val_idx]
    
    # Apply dimensionality reduction
    if is_pipeline:
        reducer = reduction_config['class'](n_comp)
        if is_supervised:
            X_tr = reducer.fit_transform(X_tr_orig, y_tr)
        else:
            X_tr = reducer.fit_transform(X_tr_orig)
        X_vl = reducer.transform(X_vl_orig)
    else:
        # PCA_TSNE
        X_tr, X_vl = reduction_config['class'](X_tr_orig, X_vl_orig, y_tr, n_comp)

    if best_overall_classifier == "XGBoost":
        mdl = XGBClassifier(
            device="cuda" if USE_GPU else "cpu",
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=RANDOM_STATE,
            verbosity=0,
            **best_overall_params,
        )
        mdl.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    else:  # SVM
        mdl = SVC(kernel="rbf", C=best_overall_params["C"],
                  gamma=best_overall_params["gamma"],
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
print(f"\nFINAL CROSS-VALIDATION METRICS FOR {best_overall_reduction} + {best_overall_classifier}:")
print(f"  Accuracy:  {np.mean(fold_metrics['accuracy']):.6f}")
print(f"  Precision: {np.mean(fold_metrics['precision']):.6f}")
print(f"  Recall:    {np.mean(fold_metrics['recall']):.6f}")
print(f"  F1-Score:  {np.mean(fold_metrics['f1']):.6f}")
print(f"  ROC-AUC:   {np.mean(fold_metrics['roc_auc']):.6f}")
print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
print()

# ============================================================================
# 10. TRAIN FINAL WINNER MODEL ON ALL DATA & SAVE
# ============================================================================
print(f"Training final {best_overall_classifier} model on all data...", end=" ", flush=True)

if best_overall_classifier == "XGBoost":
    # Determine optimal number of trees via cross-validation
    best_iters = []
    for train_idx, val_idx in skf.split(X_scaled, y):
        X_tr_orig, X_vl_orig = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        
        # Apply dimensionality reduction
        if is_pipeline:
            reducer = reduction_config['class'](n_comp)
            if is_supervised:
                X_tr = reducer.fit_transform(X_tr_orig, y_tr)
            else:
                X_tr = reducer.fit_transform(X_tr_orig)
            X_vl = reducer.transform(X_vl_orig)
        else:
            X_tr, X_vl = reduction_config['class'](X_tr_orig, X_vl_orig, y_tr, n_comp)

        mdl = XGBClassifier(
            device="cuda" if USE_GPU else "cpu",
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=RANDOM_STATE,
            verbosity=0,
            **best_overall_params,
        )
        mdl.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        best_iters.append(mdl.best_iteration + 1)

    final_n_trees = int(np.mean(best_iters))

    final_model = XGBClassifier(
        device="cuda" if USE_GPU else "cpu",
        n_estimators=final_n_trees,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        **{k: v for k, v in best_overall_params.items() if k != "n_estimators"},
    )
    final_model.fit(X_reduced_full, y)
    joblib.dump(final_model, os.path.join(MODELS_DIR, "xgb_model_final.pkl"))
else:  # SVM
    final_model = SVC(kernel="rbf", C=best_overall_params["C"],
                      gamma=best_overall_params["gamma"],
                      probability=True, random_state=RANDOM_STATE)
    final_model.fit(X_reduced_full, y)
    joblib.dump(final_model, os.path.join(MODELS_DIR, "svm_model_final.pkl"))

# Save the dimensionality reduction transformer
if best_overall_reduction_obj is not None and is_pipeline:
    joblib.dump(best_overall_reduction_obj, os.path.join(MODELS_DIR, "reduction_final.pkl"))

# Save preprocessing transformers
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_final.pkl"))
joblib.dump(vt, os.path.join(MODELS_DIR, "variance_threshold_final.pkl"))

# Save metadata about the reduction method used
reduction_metadata = {
    'method': best_overall_reduction,
    'n_components': best_overall_reduction_n,
    'classifier': best_overall_classifier,
    'accuracy': best_overall_score
}
pd.DataFrame([reduction_metadata]).to_csv(os.path.join(MODELS_DIR, "reduction_metadata.csv"), index=False)

model_file = "xgb_model_final.pkl" if best_overall_classifier == "XGBoost" else "svm_model_final.pkl"
print(f"done\nSaved:")
print(f"  - {model_file}")
if is_pipeline:
    print(f"  - reduction_final.pkl ({best_overall_reduction})")
print(f"  - scaler_final.pkl")
print(f"  - variance_threshold_final.pkl")
print(f"  - reduction_metadata.csv")
