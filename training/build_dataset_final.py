"""
build_dataset_final.py
======================
Builds the hallucination-detection feature dataset.

Both Excel files now share the same columns:
  question_text, gt_answer_snippet, doc_id, gt_page_number, question_id

For each answerable question:
  CORRECT (label = 0):  from correct answers.xlsx
    → context(doc_id, gt_page_number) + question_text + gt_answer_snippet
  INCORRECT (label = 1): from incorrect answers.xlsx
    → context(doc_id, gt_page_number) + question_text + gt_answer_snippet

Context is extracted from the PDF page indicated by doc_id + gt_page_number.
Unanswerable questions (NaN page number or "Not Answered") are skipped.
Output is saved to an Excel (.xlsx) file.
"""

import os
import gc
import logging
import argparse

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CORRECT_XLSX   = "../data/training_data/correct answers.xlsx"
INCORRECT_XLSX = "../data/training_data/incorrect answers.xlsx"
DEFAULT_PDFS   = "../resources/pdfs"
OUTPUT_XLSX    = "../data/processed/features_correct_incorrect.xlsx"
MODEL_PATH     = "../models/TinyLlama"
HIDDEN_DIM     = 2048
MAX_CTX_CHARS  = 1600


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_page(pdf_path: str, manual_page: int) -> str:
    """Extract text from a 1-indexed manual page number."""
    import fitz
    doc = fitz.open(pdf_path)
    idx = manual_page - 1
    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""
    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CTX_CHARS]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    logger.info("Model loaded on GPU")
    return tokenizer, model


# ── Forward pass ──────────────────────────────────────────────────────────────

def find_last_meaningful_token(input_ids: torch.Tensor, tokenizer) -> int:
    target_index = input_ids.shape[0] - 1
    while target_index > 0:
        tok_str = tokenizer.decode(input_ids[target_index]).strip()
        if any(c.isalnum() for c in tok_str) and "</s>" not in tok_str:
            break
        target_index -= 1
    return target_index


def get_last_hidden_state(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1900,
    ).to(model.device)

    input_ids    = inputs["input_ids"][0]
    seq_len      = input_ids.shape[0]
    target_index = find_last_meaningful_token(input_ids, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    vector = outputs.hidden_states[-1][0, target_index, :].float().cpu().numpy()

    del outputs, inputs
    torch.cuda.empty_cache()

    return {
        "vector":       vector,
        "seq_len":      seq_len,
        "target_index": target_index,
    }


# ── Build one row dict ───────────────────────────────────────────────────────

def make_row(qid, doc_id, question, answer, answer_type, label, result):
    row = {
        "question_id":  qid,
        "doc_id":       doc_id,
        "question":     question,
        "answer":       answer[:300],
        "answer_type":  answer_type,
        "label":        label,
        "seq_len":      result["seq_len"],
        "target_index": result["target_index"],
    }
    for i, v in enumerate(result["vector"]):
        row[f"v_{i}"] = round(float(v), 6)
    return row


# ── Process one set of answers ────────────────────────────────────────────────

def process_answers(df, answer_type, label, pdfs_dir,
                    tokenizer, model, all_rows, counters):
    """
    Process a dataframe of answers.
    Required columns: question_id, doc_id, question_text,
                      gt_answer_snippet, gt_page_number
    """
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Processing {answer_type} answers"):
        qid       = row["question_id"]
        doc_id    = row["doc_id"]
        question  = str(row["question_text"])
        answer    = str(row["gt_answer_snippet"])
        page_num  = row["gt_page_number"]

        # Skip unanswerable (NaN page or "Not Answered")
        if pd.isna(page_num) or answer == "Not Answered":
            counters["skipped_unanswerable"] += 1
            continue

        # Get page context from PDF
        pdf_path = os.path.join(pdfs_dir, doc_id)
        if not os.path.exists(pdf_path):
            logger.warning(f"[{qid}] PDF not found: {pdf_path}")
            counters["skipped_other"] += 1
            continue

        context = extract_page(pdf_path, int(page_num))
        if not context:
            logger.warning(f"[{qid}] empty page text - skipping")
            counters["skipped_other"] += 1
            continue

        # Build input: context + question + answer
        text = (f"{context}\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}")

        try:
            result = get_last_hidden_state(text, tokenizer, model)
            all_rows.append(make_row(
                qid, doc_id, question,
                answer, answer_type, label, result))
            counters["written"] += 1
        except Exception as e:
            logger.warning(f"[{qid}] {answer_type}-answer failed: {e}")
            counters["errors"] += 1

        gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset(correct_xlsx=CORRECT_XLSX,
                  incorrect_xlsx=INCORRECT_XLSX,
                  pdfs_dir=DEFAULT_PDFS,
                  output_xlsx=OUTPUT_XLSX,
                  model_path=MODEL_PATH):

    # ── Load both Excel files directly ────────────────────────────────────────
    df_correct   = pd.read_excel(correct_xlsx)
    df_incorrect = pd.read_excel(incorrect_xlsx)
    logger.info(f"Correct rows: {len(df_correct)}  |  Incorrect rows: {len(df_incorrect)}")

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model = load_model(model_path)

    all_rows = []
    counters = {"written": 0, "skipped_unanswerable": 0,
                "skipped_other": 0, "errors": 0}

    # ── Process correct answers (label = 0) ───────────────────────────────────
    process_answers(df_correct, "correct", 0, pdfs_dir,
                    tokenizer, model, all_rows, counters)

    # ── Process incorrect answers (label = 1) ─────────────────────────────────
    process_answers(df_incorrect, "incorrect", 1, pdfs_dir,
                    tokenizer, model, all_rows, counters)

    # ── Save to Excel ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    df_out.to_excel(output_xlsx, index=False, engine="openpyxl")

    label_counts = dict(df_out["label"].value_counts()) if len(df_out) > 0 else {}
    logger.info(
        f"Done - written: {counters['written']} rows to {output_xlsx} | "
        f"labels: {label_counts} | "
        f"unanswerable skipped: {counters['skipped_unanswerable']} | "
        f"other skipped: {counters['skipped_other']} | "
        f"errors: {counters['errors']}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build hallucination feature dataset from correct & incorrect answers"
    )
    parser.add_argument("--correct",   default=CORRECT_XLSX)
    parser.add_argument("--incorrect", default=INCORRECT_XLSX)
    parser.add_argument("--pdfs_dir",  default=DEFAULT_PDFS)
    parser.add_argument("--output",    default=OUTPUT_XLSX)
    parser.add_argument("--model",     default=MODEL_PATH)
    args = parser.parse_args()

    build_dataset(
        correct_xlsx   = args.correct,
        incorrect_xlsx = args.incorrect,
        pdfs_dir       = args.pdfs_dir,
        output_xlsx    = args.output,
        model_path     = args.model,
    )
