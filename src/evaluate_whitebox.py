import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

def main():
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    try:
        # Load saved components
        model = joblib.load("models/lr_model_final.pkl")
        scaler = joblib.load("models/scaler_final.pkl")
        pca = joblib.load("models/reduction_final.pkl")

        # Load your dataset (replace with actual loading)
        try:
            X = np.load("X.npy")   # features
            y_true = np.load("y.npy")  # labels
        except FileNotFoundError:
            import pandas as pd
            print("X.npy and y.npy not found, attempting to load from Excel feature file instead...")
            df = pd.read_excel("data/processed/features_shuffled_final.xlsx")
            feature_cols = [c for c in df.columns if c.startswith("v_")] + ["seq_len", "target_index"]
            X = df[feature_cols].values.astype(np.float64)
            y_true = df["label"].values.astype(int)

        # Preprocess
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # Predictions
        y_pred = model.predict(X_pca)
        y_prob = model.predict_proba(X_pca)[:, 1]

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print(f"LogisticRegression (PCA=150):  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

        # ==========================================
        # Generate Graphs for Whitebox Methodology
        # ==========================================

        # 1. Confusion Matrix
        plt.figure(figsize=(7, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        metrics_text = f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        plt.title(f'Confusion Matrix\n{metrics_text}', fontsize=11, pad=15)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.close()

        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curve.png')
        plt.close()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='purple', lw=2, label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/pr_curve.png')
        plt.close()

        print("Whitebox methodology graphs successfully generated and saved to the 'results/' directory:")
        print("  - results/confusion_matrix.png")
        print("  - results/roc_curve.png")
        print("  - results/pr_curve.png")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the models ('models/lr_model_final.pkl', 'models/scaler_final.pkl', 'models/reduction_final.pkl') "
              "and the dataset ('X.npy'/'y.npy' or 'data/processed/features_shuffled_final.xlsx') exist.")

if __name__ == "__main__":
    main()
