"""
Unified Training Script for Hallucination Detection

Combines the original train.py (SVM/XGBoost) and train_alt.py (AdaBoost/ExtraTrees/KNN)
into a single comprehensive training pipeline.

Tests multiple classifiers with both PCA and PCA+LDA dimensionality reduction,
then saves only the single best-performing model for production use.

Usage:
    python training/train_unified.py

Output:
    - models/best_model_final.pkl       # The winning classifier
    - models/scaler_final.pkl           # StandardScaler
    - models/reduction_final.pkl        # PCA or PCA+LDA pipeline
    - models/variance_threshold_final.pkl
    - models/reduction_metadata.csv     # Winner info (method, model_type, accuracy, n_components)
    - data/training_data/training_data_final.xlsx
"""

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
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available, skipping XGBoost models")

# Setup directories
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "training_data"), exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("  UNIFIED HALLUCINATION DETECTION MODEL TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/9] Loading data...")
df = pd.read_excel(os.path.join(DATA_DIR, "processed", "features_shuffled_final.xlsx"))

meta_cols    = ["question_id", "doc_id", "question", "answer", "answer_type"]
feature_cols = [c for c in df.columns if c.startswith("v_")] + ["seq_len", "target_index"]
X    = df[feature_cols].values.astype(np.float64)
y    = df["label"].values.astype(int)
meta = df[meta_cols]

print(f"✓ Loaded {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Class distribution: label 0 (correct): {(y==0).sum()}, label 1 (hallucinated): {(y==1).sum()}")

# ============================================================================
# 2. PREPROCESSING: Variance Threshold + StandardScaler
# ============================================================================
print("\n[2/9] Preprocessing features...")
vt = VarianceThreshold(threshold=1e-6)
X_vt = vt.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vt)

print(f"✓ After VarianceThreshold: {X_vt.shape[1]} features (removed {X.shape[1] - X_vt.shape[1]} low-variance)")

# ============================================================================
# 3. PCA VARIANCE ANALYSIS
# ============================================================================
print("\n[3/9] Analyzing PCA variance...")
full_pca = PCA(random_state=RANDOM_STATE)
full_pca.fit(X_scaled)
cumvar = np.cumsum(full_pca.explained_variance_ratio_)
n80 = np.searchsorted(cumvar, 0.80) + 1
n85 = np.searchsorted(cumvar, 0.85) + 1
n90 = np.searchsorted(cumvar, 0.90) + 1
n95 = np.searchsorted(cumvar, 0.95) + 1
n99 = np.searchsorted(cumvar, 0.99) + 1

print(f"✓ PCA variance thresholds:")
print(f"  80% → {n80} components")
print(f"  85% → {n85} components")
print(f"  90% → {n90} components")
print(f"  95% → {n95} components")
print(f"  99% → {n99} components")

# ============================================================================
# 4. DEFINE REDUCTION METHODS
# ============================================================================
def make_pca(n):
    return PCA(n_components=n, random_state=RANDOM_STATE)

def make_pca_lda(n):
    return Pipeline([
        ("pca", PCA(n_components=n, random_state=RANDOM_STATE)),
        ("lda", LDA(n_components=1)),
    ])

n_components_list = [150, 200]

reduction_methods = {
    "PCA": [(f"PCA_{n}", make_pca(n)) for n in n_components_list],
}

# ============================================================================
# 5. DEFINE CLASSIFIERS
# ============================================================================
print("\n[4/9] Defining classifier configurations...")

classifiers = {}

# SVM configurations
classifiers["SVM"] = [
    ("SVM_C1_scale",    SVC(C=1,   gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C1_auto",     SVC(C=1,   gamma="auto",  kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C10_scale",   SVC(C=10,  gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C10_auto",    SVC(C=10,  gamma="auto",  kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C50_scale",   SVC(C=50,  gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C50_auto",    SVC(C=50,  gamma="auto",  kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C100_scale",  SVC(C=100, gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ("SVM_C100_auto",   SVC(C=100, gamma="auto",  kernel="rbf", probability=True, random_state=RANDOM_STATE)),
]

# Logistic Regression
classifiers["LogisticRegression"] = [
    ("LR_C01",  LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)),
    ("LR_C1",   LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
    ("LR_C10",  LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_STATE)),
]

# AdaBoost configurations
classifiers["AdaBoost"] = [
    ("Ada_50_01",   AdaBoostClassifier(n_estimators=50,  learning_rate=0.1, random_state=RANDOM_STATE)),
    ("Ada_50_05",   AdaBoostClassifier(n_estimators=50,  learning_rate=0.5, random_state=RANDOM_STATE)),
    ("Ada_100_01",  AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)),
    ("Ada_100_05",  AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE)),
    ("Ada_200_01",  AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE)),
    ("Ada_300_10",  AdaBoostClassifier(n_estimators=300, learning_rate=1.0, random_state=RANDOM_STATE)),
]

# ExtraTrees configurations
classifiers["ExtraTrees"] = [
    ("ET_500_None_1",   ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
    ("ET_500_None_2",   ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, max_features="log2", random_state=RANDOM_STATE, n_jobs=-1)),
    ("ET_500_10_1",     ExtraTreesClassifier(n_estimators=500, max_depth=10,   min_samples_leaf=1, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
    ("ET_500_10_2",     ExtraTreesClassifier(n_estimators=500, max_depth=10,   min_samples_leaf=2, max_features="log2", random_state=RANDOM_STATE, n_jobs=-1)),
]

# KNeighbors configurations
classifiers["KNN"] = [
    ("KNN_3_uniform",   KNeighborsClassifier(n_neighbors=3,  weights="uniform",  n_jobs=-1)),
    ("KNN_3_distance",  KNeighborsClassifier(n_neighbors=3,  weights="distance", n_jobs=-1)),
    ("KNN_5_uniform",   KNeighborsClassifier(n_neighbors=5,  weights="uniform",  n_jobs=-1)),
    ("KNN_5_distance",  KNeighborsClassifier(n_neighbors=5,  weights="distance", n_jobs=-1)),
    ("KNN_10_uniform",  KNeighborsClassifier(n_neighbors=10, weights="uniform",  n_jobs=-1)),
]

# XGBoost configurations (if available)
if HAS_XGBOOST:
    # Check for GPU support
    try:
        test_m = XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
        test_m.fit(X_scaled[:10, :10], y[:10])
        device = "cuda"
        print("✓ XGBoost GPU detected")
    except Exception:
        device = "cpu"
        print("  XGBoost using CPU")
    
    classifiers["XGBoost"] = [
        ("XGB_d1_lr005", XGBClassifier(max_depth=1, learning_rate=0.005, n_estimators=1000, 
                                       reg_alpha=10, reg_lambda=10, gamma=5, min_child_weight=10,
                                       device=device, random_state=RANDOM_STATE, verbosity=0)),
        ("XGB_d2_lr005", XGBClassifier(max_depth=2, learning_rate=0.005, n_estimators=1000,
                                       reg_alpha=10, reg_lambda=10, gamma=5, min_child_weight=10,
                                       device=device, random_state=RANDOM_STATE, verbosity=0)),
    ]

total_classifiers = sum(len(v) for v in classifiers.values())
total_reductions = sum(len(v) for v in reduction_methods.values())
total_combinations = total_classifiers * total_reductions

print(f"✓ Configured {total_classifiers} classifiers across {len(classifiers)} model families")
print(f"✓ Testing {total_reductions} dimensionality reduction configurations")
print(f"  Total combinations to evaluate: {total_combinations}")

# ============================================================================
# 6. GRID SEARCH WITH CROSS-VALIDATION
# ============================================================================
print(f"\n[5/9] Running comprehensive grid search...")
print(f"  Using 5-fold stratified cross-validation")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
best_acc = -1
best_config = None

total_tested = 0

for red_family, red_configs in reduction_methods.items():
    for red_name, red_maker in red_configs:
        for clf_family, clf_configs in classifiers.items():
            for clf_name, clf_template in clf_configs:
                total_tested += 1
                combo_name = f"{red_name}+{clf_name}"
                
                # Cross-validation
                fold_accs = []
                for train_idx, val_idx in skf.split(X_scaled, y):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Clone and fit reducer
                    if red_family == "PCA":
                        reducer = make_pca(int(red_name.split("_")[-1]))
                    else:  # PCA_LDA
                        reducer = make_pca_lda(int(red_name.split("_")[-1]))
                    
                    X_train_red = reducer.fit_transform(X_train, y_train)
                    X_val_red = reducer.transform(X_val)
                    
                    # Clone and fit classifier
                    from sklearn.base import clone
                    clf = clone(clf_template)
                    clf.fit(X_train_red, y_train)
                    
                    # Predict and score
                    y_pred = clf.predict(X_val_red)
                    acc = accuracy_score(y_val, y_pred)
                    fold_accs.append(acc)
                
                mean_acc = np.mean(fold_accs)
                std_acc = np.std(fold_accs)
                
                results.append({
                    "reduction_family": red_family,
                    "reduction_config": red_name,
                    "classifier_family": clf_family,
                    "classifier_config": clf_name,
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                })
                
                print(f"  [{total_tested:3d}/{total_combinations}] {combo_name:50s} → {mean_acc:.4f} ± {std_acc:.4f}")
                
                # Track best
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_config = {
                        "reduction_family": red_family,
                        "reduction_name": red_name,
                        "n_components": int(red_name.split("_")[-1]),
                        "classifier_family": clf_family,
                        "classifier_name": clf_name,
                        "classifier_template": clf_template,
                        "reducer_maker": red_maker if red_family == "PCA" else make_pca_lda(int(red_name.split("_")[-1])),
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                    }

print(f"\n{'='*80}")
print(f"🏆 BEST CONFIGURATION")
print(f"{'='*80}")
print(f"  Reduction:   {best_config['reduction_name']}")
print(f"  Classifier:  {best_config['classifier_name']}")
print(f"  Accuracy:    {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
print(f"{'='*80}")

# ============================================================================
# 7. DETAILED EVALUATION OF WINNER
# ============================================================================
print(f"\n[6/9] Detailed evaluation of winning configuration...")

fold_metrics = {
    "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []
}

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Fit reducer
    if best_config['reduction_family'] == "PCA":
        reducer = make_pca(best_config['n_components'])
    else:
        reducer = make_pca_lda(best_config['n_components'])
    
    X_train_red = reducer.fit_transform(X_train, y_train)
    X_val_red = reducer.transform(X_val)
    
    # Fit classifier
    from sklearn.base import clone
    clf = clone(best_config['classifier_template'])
    clf.fit(X_train_red, y_train)
    
    # Predict
    y_pred = clf.predict(X_val_red)
    y_proba = clf.predict_proba(X_val_red)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    # Metrics
    fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
    fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
    fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
    fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
    if y_proba is not None:
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
    
    print(f"  Fold {fold_num}: Acc={fold_metrics['accuracy'][-1]:.4f}, "
          f"Prec={fold_metrics['precision'][-1]:.4f}, "
          f"Rec={fold_metrics['recall'][-1]:.4f}, "
          f"F1={fold_metrics['f1'][-1]:.4f}")

print(f"\n  5-Fold Averages:")
for metric_name, values in fold_metrics.items():
    if values:
        print(f"    {metric_name:12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# ============================================================================
# 8. TRAIN FINAL MODEL ON ALL DATA
# ============================================================================
print(f"\n[7/9] Training final model on complete dataset...")

# Fit reducer on all data
if best_config['reduction_family'] == "PCA":
    final_reducer = make_pca(best_config['n_components'])
else:
    final_reducer = make_pca_lda(best_config['n_components'])

X_reduced = final_reducer.fit_transform(X_scaled, y)

# Fit classifier on all data
from sklearn.base import clone
final_classifier = clone(best_config['classifier_template'])
final_classifier.fit(X_reduced, y)

# Final predictions for confusion matrix
y_pred_final = final_classifier.predict(X_reduced)
cm = confusion_matrix(y, y_pred_final)

print(f"✓ Final model trained on {X.shape[0]} samples")
print(f"\n  Confusion Matrix (on training data):")
print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
print(f"\n  Classification Report:")
print(classification_report(y, y_pred_final, target_names=["Correct", "Hallucinated"]))

# ============================================================================
# 9. SAVE FINAL MODEL AND ARTIFACTS
# ============================================================================
print(f"\n[8/9] Saving model artifacts...")

# Save preprocessing pipeline
joblib.dump(vt, os.path.join(MODELS_DIR, "variance_threshold_final.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_final.pkl"))
joblib.dump(final_reducer, os.path.join(MODELS_DIR, "reduction_final.pkl"))
print(f"  ✓ Saved: variance_threshold_final.pkl")
print(f"  ✓ Saved: scaler_final.pkl")
print(f"  ✓ Saved: reduction_final.pkl")

# Save classifier with descriptive name
model_filename = f"best_model_final.pkl"
joblib.dump(final_classifier, os.path.join(MODELS_DIR, model_filename))
print(f"  ✓ Saved: {model_filename}")

# Save metadata
metadata = pd.DataFrame([{
    "reduction_method": best_config['reduction_family'],
    "n_components": best_config['n_components'],
    "classifier_type": best_config['classifier_family'],
    "classifier_config": best_config['classifier_name'],
    "cv_accuracy_mean": best_config['mean_accuracy'],
    "cv_accuracy_std": best_config['std_accuracy'],
    "model_filename": model_filename,
}])
metadata.to_csv(os.path.join(MODELS_DIR, "reduction_metadata.csv"), index=False)
print(f"  ✓ Saved: reduction_metadata.csv")

# Save transformed training data
df_reduced = pd.concat([
    meta.reset_index(drop=True),
    pd.DataFrame({"label": y}),
    pd.DataFrame(X_reduced, columns=[f"PC{i}" for i in range(X_reduced.shape[1])]),
], axis=1)
output_xlsx = os.path.join(DATA_DIR, "training_data", "training_data_final.xlsx")
df_reduced.to_excel(output_xlsx, index=False)
print(f"  ✓ Saved: training_data/training_data_final.xlsx")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print(f"\n[9/9] Training complete!")
print(f"\n{'='*80}")
print(f"  FINAL MODEL SUMMARY")
print(f"{'='*80}")
print(f"  Total configurations tested: {total_combinations}")
print(f"  Winner: {best_config['reduction_name']} + {best_config['classifier_name']}")
print(f"  Cross-validation accuracy: {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
print(f"  Model saved as: models/{model_filename}")
print(f"{'='*80}")
print(f"\n✓ Ready for inference with src/app.py")
