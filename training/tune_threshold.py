"""
tune_threshold.py
=================
Improvement 4 — Data-driven similarity threshold tuning.

Reads the completed evaluation results (evaluation_results.xlsx produced by
evaluate_blackbox.py) and finds the FAISS similarity threshold that maximises
Youden's J statistic (TPR - FPR) on the ROC curve.

The optimal threshold replaces the fixed 0.35 in nli_utils.py.

Usage
-----
    python tune_threshold.py
    python tune_threshold.py --results data/results/evaluation_results.xlsx
    python tune_threshold.py --results data/results/evaluation_results.xlsx \
                              --plot                # saves roc_curve.png

Output
------
    Prints the optimal threshold and corresponding metrics.
    Optionally saves data/results/roc_curve.png

What it does
------------
For each possible threshold t (sweep over all observed max_similarity values):
  - Rows with max_similarity < t are predicted HALLUCINATION (label 1)
  - Rows with max_similarity >= t skip the threshold and let NLI decide
    (for the purpose of this sweep, we use the actual nli_verdict already
     in evaluation_results.xlsx)
  - Compute TPR (recall on label-1) and FPR (1 - specificity on label-0)
  - Youden's J = TPR - FPR

The threshold with the highest J is the recommended value.
Update SIMILARITY_THRESHOLD in nli_utils.py with this value.
"""

import argparse
import os
import numpy as np
import pandas as pd

DEFAULT_RESULTS = "data/results/evaluation_results.xlsx"
DEFAULT_PLOT    = "data/results/roc_curve.png"


def compute_metrics_at_threshold(df: pd.DataFrame, threshold: float):
    """
    Apply threshold to the dataframe and compute TP, TN, FP, FN.

    Logic:
      - If max_similarity < threshold  →  predicted = 1 (HALLUCINATION)
      - Else                           →  use the nli_verdict already in df
            HALLUCINATION → 1,  GROUNDED/UNCERTAIN → 0
    """
    def predict_row(row):
        if row["max_similarity"] < threshold:
            return 1
        return 1 if row["nli_verdict"] == "HALLUCINATION" else 0

    predicted = df.apply(predict_row, axis=1)
    true_labels = df["label"]

    TP = int(((predicted == 1) & (true_labels == 1)).sum())
    TN = int(((predicted == 0) & (true_labels == 0)).sum())
    FP = int(((predicted == 1) & (true_labels == 0)).sum())
    FN = int(((predicted == 0) & (true_labels == 1)).sum())

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # recall / sensitivity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0   # 1 - specificity
    acc = (TP + TN) / len(df) if len(df) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1   = (2 * prec * TPR / (prec + TPR)) if (prec + TPR) > 0 else 0.0

    return {"threshold": threshold, "TPR": TPR, "FPR": FPR,
            "J": TPR - FPR, "accuracy": acc, "precision": prec,
            "recall": TPR, "f1": f1, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def tune(results_path: str = DEFAULT_RESULTS,
         plot: bool = False,
         plot_path: str = DEFAULT_PLOT):

    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found: {results_path}")
        print("Run evaluate_blackbox.py first to generate this file.")
        return None

    df = pd.read_excel(results_path)
    required = {"label", "max_similarity", "nli_verdict"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in results file: {missing}")
        return None

    print(f"Loaded {len(df)} rows from {results_path}")
    label_counts = df["label"].value_counts().to_dict()
    print(f"Label distribution: {label_counts}")

    # Sweep thresholds over all unique max_similarity values
    # plus a small set of fixed candidates for coverage
    sim_values   = sorted(df["max_similarity"].unique().tolist())
    extra_values = [round(x, 2) for x in np.arange(0.0, 1.01, 0.01)]
    thresholds   = sorted(set(sim_values + extra_values))

    rows = []
    for t in thresholds:
        m = compute_metrics_at_threshold(df, t)
        rows.append(m)

    roc_df = pd.DataFrame(rows)

    # Best threshold = highest Youden's J
    best_idx = roc_df["J"].idxmax()
    best     = roc_df.loc[best_idx]

    sep  = "=" * 58
    sep2 = "-" * 58
    print(f"\n{sep}")
    print("  SIMILARITY THRESHOLD TUNING — RESULTS")
    print(sep)
    print(f"  Current default threshold : 0.35")
    print(f"  Optimal threshold (Youden): {best['threshold']:.4f}")
    print(sep2)
    print(f"  At optimal threshold {best['threshold']:.4f}:")
    print(f"    Accuracy   : {best['accuracy']*100:.2f}%")
    print(f"    Precision  : {best['precision']*100:.2f}%")
    print(f"    Recall     : {best['recall']*100:.2f}%  (hallucinations caught)")
    print(f"    F1 Score   : {best['f1']*100:.2f}%")
    print(f"    Youden J   : {best['J']:.4f}")
    print(f"    TPR        : {best['TPR']:.4f}  (sensitivity)")
    print(f"    FPR        : {best['FPR']:.4f}  (1 - specificity)")
    print(f"    TP={int(best['TP'])}  TN={int(best['TN'])}  "
          f"FP={int(best['FP'])}  FN={int(best['FN'])}")
    print(sep2)
    print(f"\n  ACTION: Open nli_utils.py and set:")
    print(f"    SIMILARITY_THRESHOLD = {best['threshold']:.4f}")
    print(sep)

    # Also print comparison at the original threshold
    orig = compute_metrics_at_threshold(df, 0.35)
    print(f"\n  Comparison at original threshold 0.35:")
    print(f"    Accuracy={orig['accuracy']*100:.2f}%  "
          f"Recall={orig['recall']*100:.2f}%  "
          f"F1={orig['f1']*100:.2f}%  J={orig['J']:.4f}")

    if plot:
        _save_roc_plot(roc_df, best["threshold"], plot_path)

    return float(best["threshold"])


def _save_roc_plot(roc_df: pd.DataFrame, best_threshold: float,
                   plot_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: ROC curve
        ax = axes[0]
        ax.plot(roc_df["FPR"], roc_df["TPR"], "b-", linewidth=1.5,
                label="ROC curve")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

        # Mark optimal point
        best_row = roc_df.loc[roc_df["J"].idxmax()]
        ax.scatter([best_row["FPR"]], [best_row["TPR"]],
                   color="red", zorder=5, s=80,
                   label=f"Optimal (t={best_threshold:.3f})")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve — Similarity Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Youden's J vs threshold
        ax2 = axes[1]
        ax2.plot(roc_df["threshold"], roc_df["J"], "g-", linewidth=1.5,
                 label="Youden's J")
        ax2.axvline(x=best_threshold, color="red", linestyle="--",
                    label=f"Optimal t={best_threshold:.3f}")
        ax2.axvline(x=0.35, color="orange", linestyle=":",
                    label="Current t=0.35")
        ax2.set_xlabel("Similarity Threshold")
        ax2.set_ylabel("Youden's J (TPR - FPR)")
        ax2.set_title("Youden's J vs Threshold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ROC plot saved -> {plot_path}")
    except ImportError:
        print("\n  [INFO] matplotlib not installed — skipping plot. "
              "Install with: pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal FAISS similarity threshold using Youden's J"
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Path to evaluation_results.xlsx")
    parser.add_argument("--plot", action="store_true",
                        help="Save ROC curve to data/results/roc_curve.png")
    parser.add_argument("--plot_path", default=DEFAULT_PLOT,
                        help="Where to save the ROC plot")
    args = parser.parse_args()

    tune(results_path=args.results, plot=args.plot, plot_path=args.plot_path)