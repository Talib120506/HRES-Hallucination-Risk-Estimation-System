"""
dedup_clean.py
==============
Cleans a TechManualQA Excel file by:
  1. Keeping only required columns
  2. Removing unanswerable questions
  3. Removing semantically similar duplicate questions (per document)
     using TF-IDF (question + answer) cosine similarity

Outputs:
  - <output_file>          : cleaned dataset
  - duplicates_report.xlsx : every removed pair with similarity score + answers

Usage:
  python dedup_clean.py --input TechManualQA_700_jsonl.xlsx
  python dedup_clean.py --input TechManualQA_700_jsonl.xlsx --threshold 0.60
  python dedup_clean.py --input TechManualQA_700_jsonl.xlsx --output my_clean.xlsx
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Config ────────────────────────────────────────────────────────────────────

KEEP_COLS = [
    "question_id",
    "doc_id",
    "question_text",
    "category",
    "gt_answer_snippet",
    "gt_page_number",
]

DEFAULT_THRESHOLD = 0.60   # cosine similarity above which a pair is a duplicate


# ── Step 1: Load & keep only needed columns ───────────────────────────────────

def load_and_trim(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[KEEP_COLS].copy()


# ── Step 2: Remove unanswerable rows ─────────────────────────────────────────

def remove_unanswerable(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["category"] != "Unanswerable"]
    df = df[df["gt_answer_snippet"] != "Not Answered"]
    df = df[df["gt_page_number"].notna()]
    df = df.reset_index(drop=True)
    print(f"  Unanswerable removed : {before - len(df):>4}  ({before} -> {len(df)})")
    return df


# ── Step 3: Semantic dedup per document ───────────────────────────────────────

def semantic_dedup(df: pd.DataFrame, threshold: float):
    """
    For each document, compute pairwise cosine similarity between
    (question_text + gt_answer_snippet) TF-IDF vectors.
    Greedily keep the first question in each similar pair, drop the rest.

    Returns:
        df_clean  : deduplicated dataframe
        report_df : dataframe of all removed pairs with scores + answers
    """
    to_drop = set()
    report  = []

    for doc_id in df["doc_id"].unique():
        subset = df[df["doc_id"] == doc_id].copy().reset_index()   # keeps original index

        questions = subset["question_text"].tolist()
        answers   = subset["gt_answer_snippet"].fillna("").astype(str).tolist()
        combined  = [f"{q} {a}" for q, a in zip(questions, answers)]

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        mat = vec.fit_transform(combined)
        sim = cosine_similarity(mat)

        for i in range(len(sim)):
            if subset.iloc[i]["index"] in to_drop:
                continue
            for j in range(i + 1, len(sim)):
                if subset.iloc[j]["index"] in to_drop:
                    continue
                if sim[i][j] >= threshold:
                    orig_idx = subset.iloc[j]["index"]
                    to_drop.add(orig_idx)
                    report.append({
                        "doc_id":         doc_id,
                        "similarity":     round(float(sim[i][j]), 3),
                        "kept_id":        subset.iloc[i]["question_id"],
                        "kept_q":         subset.iloc[i]["question_text"],
                        "kept_answer":    subset.iloc[i]["gt_answer_snippet"],
                        "dropped_id":     subset.iloc[j]["question_id"],
                        "dropped_q":      subset.iloc[j]["question_text"],
                        "dropped_answer": subset.iloc[j]["gt_answer_snippet"],
                    })

    df_clean   = df.drop(index=list(to_drop)).reset_index(drop=True)
    report_df  = pd.DataFrame(report)
    print(f"  Semantic dups removed: {len(to_drop):>4}  ({len(df)} -> {len(df_clean)})  [threshold={threshold}]")
    return df_clean, report_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean TechManualQA Excel file")
    parser.add_argument("--input",     required=True, help="Path to input .xlsx file")
    parser.add_argument("--output",    default=None,  help="Path for cleaned output .xlsx (default: auto-named)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold for duplicates (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    input_path = args.input
    threshold  = args.threshold

    # Auto-name output if not specified
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_clean.xlsx"

    report_path = os.path.splitext(output_path)[0] + "_duplicates_report.xlsx"

    print(f"\nInput  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Report : {report_path}\n")

    # Run pipeline
    print("Step 1: Loading and trimming columns...")
    df = load_and_trim(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Step 2: Removing unanswerable questions...")
    df = remove_unanswerable(df)

    print("Step 3: Removing semantic duplicates (per document)...")
    df_clean, report_df = semantic_dedup(df, threshold)

    # Save
    df_clean.to_excel(output_path, index=False, engine="openpyxl")
    report_df.to_excel(report_path, index=False, engine="openpyxl")

    print(f"\nDone.")
    print(f"  Final rows : {len(df_clean)}")
    print(f"  Saved      : {output_path}")
    print(f"  Report     : {report_path}")


if __name__ == "__main__":
    main()