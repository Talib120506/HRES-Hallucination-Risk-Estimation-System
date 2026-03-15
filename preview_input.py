"""
preview_input.py
================
Prints the input text fed into TinyLlama for the first 5 answerable rows.
No model loading, no forward pass — just shows what the input looks like.

Usage:
    python preview_input.py
    python preview_input.py --dataset TechManualQA_350.xlsx --pdfs_dir ./pdfs
"""

import os
import argparse
import pandas as pd

DEFAULT_DATASET   = "TechManualQA_350.xlsx"
DEFAULT_PDFS_DIR  = "./10 pdfs"
MAX_CONTEXT_CHARS = 1600


def extract_page(pdf_path: str, manual_page: int) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    idx = manual_page - 1

    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""

    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CONTEXT_CHARS]


def preview(dataset_path=DEFAULT_DATASET, pdfs_dir=DEFAULT_PDFS_DIR, n=5):
    df = (pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx")
          else pd.read_csv(dataset_path))

    shown = 0
    for _, row in df.iterrows():
        if shown >= n:
            break

        if row["category"] == "Unanswerable" or row["gt_answer_snippet"] == "Not Answered":
            continue
        if pd.isna(row["gt_page_number"]):
            continue

        pdf_path = os.path.join(pdfs_dir, row["doc_id"])
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}\n")
            continue

        context = extract_page(pdf_path, int(row["gt_page_number"]))
        if not context:
            continue

        text = f"{context}\n\nQuestion: {row['question_text']}\nAnswer: {row['gt_answer_snippet']}"

        print("=" * 70)
        print(f"Row        : {shown + 1}")
        print(f"Question ID: {row['question_id']}")
        print(f"Doc        : {row['doc_id']}  |  Page: {int(row['gt_page_number'])}")
        print(f"Category   : {row['category']}")
        print("-" * 70)
        print("INPUT TEXT:")
        print(text)
        print("=" * 70)
        print()

        shown += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default=DEFAULT_DATASET)
    parser.add_argument("--pdfs_dir", default=DEFAULT_PDFS_DIR)
    parser.add_argument("--n",        type=int, default=5)
    args = parser.parse_args()

    preview(args.dataset, args.pdfs_dir, args.n)