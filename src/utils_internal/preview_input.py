"""
preview_input.py
================
Prints the formatted input text fed into Gemma, tokenizes it, 
and identifies the exact target token extracted by the build_dataset logic.

Usage:
    python preview_input.py
"""

import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET   = os.path.join(BASE_DIR, "data", "raw", "TechManualQA_474 clean.xlsx")
DEFAULT_PDFS_DIR  = os.path.join(BASE_DIR, "resources", "pdfs")
MODEL_PATH        = os.path.join(BASE_DIR, "models", "gemma-2-2b-it")
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

def find_last_meaningful_token(input_ids: torch.Tensor, tokenizer) -> int:
    special_ids = set(tokenizer.all_special_ids)
    for i in range(len(input_ids) - 1, -1, -1):
        tok_id  = input_ids[i].item()
        tok_str = tokenizer.decode(tok_id).strip()

        if tok_id in special_ids:
            continue
        if not any(c.isalnum() for c in tok_str):
            continue

        return i
    return 0


def preview(dataset_path=DEFAULT_DATASET, pdfs_dir=DEFAULT_PDFS_DIR, n=5):
    df = (pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx")
          else pd.read_csv(dataset_path))

    print(f"\nLoading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Tokenizer loaded!\n")

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
            continue

        context = extract_page(pdf_path, int(row["gt_page_number"]))
        if not context:
            continue

        question = str(row['question_text'])
        gt_answer = str(row['gt_answer_snippet'])

        if str(context) == 'nan' or not context:
            user_msg = f"Question: {question}"
        else:
            user_msg = f"Context: {context}\n\nQuestion: {question}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": gt_answer}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1900)
        input_ids = inputs["input_ids"][0]
        
        target_index = find_last_meaningful_token(input_ids, tokenizer)
        target_token_id = input_ids[target_index].item()
        target_token_str = tokenizer.decode(target_token_id)

        print("=" * 70)
        print(f"Row {shown + 1}")
        print("-" * 70)
        print(f"QUESTION IN: {question}")
        print(f"ANSWER IN  : {gt_answer}")
        print(f"\nCONTEXT TAKEN (first 500 chars):")
        print(f"{context[:500]}...")
        print("-" * 70)
        print("EXTRACTION RESULT:")
        print(f"Last Token extracted for vector : '{target_token_str}'")
        print("=" * 70)
        print()

        shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default=DEFAULT_DATASET)
    parser.add_argument("--pdfs_dir", default=DEFAULT_PDFS_DIR)
    parser.add_argument("--n",        type=int, default=3)
    args = parser.parse_args()

    preview(args.dataset, args.pdfs_dir, args.n)
