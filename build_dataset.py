"""
build_dataset.py
================
For each row in the dataset:

  ANSWERABLE (300 rows):
    1. Extract page text from PDF using gt_page_number
    2. Concatenate: page_text + question + gt_answer
    3. Forward pass through TinyLlama
    4. Extract hidden state at last layer, last meaningful token
    5. Save to CSV with label = 0

  UNANSWERABLE (50 rows):
    — placeholder, handled separately in a future step —

Requirements:
    pip install PyMuPDF openpyxl torch transformers tqdm pandas
"""

import os
import gc
import csv
import logging
import argparse

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET   = "TechManualQA_350.xlsx"
DEFAULT_PDFS_DIR  = "./10 pdfs"
OUTPUT_CSV        = "hallucination_features.csv"
MODEL_PATH        = "./models/TinyLlama"
HIDDEN_DIM        = 2048
MAX_CONTEXT_CHARS = 1600

CSV_FIELDS = (
    ["question_id", "doc_id", "category", "question",
     "gt_answer", "label", "seq_len", "target_index"] +
    [f"v_{i}" for i in range(HIDDEN_DIM)]
)


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_page(pdf_path: str, manual_page: int) -> str:
    """
    Extract text from the exact manual page number (1-indexed).
    manual_page maps directly to 0-based PDF index: idx = manual_page - 1
    No offset search — the dataset gt_page_number is trusted as-is.
    """
    import fitz
    doc = fitz.open(pdf_path)
    idx = manual_page - 1   # convert 1-indexed manual page to 0-based PDF index

    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""

    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CONTEXT_CHARS]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch: "
                           "pip install torch --index-url https://download.pytorch.org/whl/cu121")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded on GPU!")
    return tokenizer, model


# ── Forward pass ──────────────────────────────────────────────────────────────

def find_last_meaningful_token(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Walk backwards through token IDs to find the last token that contains
    at least one alphanumeric character and is not a special token like </s>.
    Returns the 0-based index within input_ids.
    """
    target_index = input_ids.shape[0] - 1

    while target_index > 0:
        tok_str = tokenizer.decode(input_ids[target_index]).strip()
        if any(char.isalnum() for char in tok_str) and "</s>" not in tok_str:
            break
        target_index -= 1

    return target_index


def get_last_hidden_state(text: str, tokenizer, model) -> dict:
    """
    Tokenize text, run a single forward pass, return the hidden state
    at the last meaningful token (last alphanumeric, not punctuation
    or special tokens).

    Input:  plain text string  (context + question + answer)
    Output: 2048-D vector from the last layer at that token position
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1900,
    ).to(model.device)

    input_ids    = inputs["input_ids"][0]        # (seq_len,)
    seq_len      = input_ids.shape[0]
    target_index = find_last_meaningful_token(input_ids, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Last layer, target token position → (2048,)
    vector = outputs.hidden_states[-1][0, target_index, :].float().cpu().numpy()

    del outputs, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "vector"       : vector,
        "seq_len"      : seq_len,
        "target_index" : target_index,
    }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def open_csv(path: str, append: bool):
    mode         = "a" if append else "w"
    write_header = not (append and os.path.exists(path)
                        and os.path.getsize(path) > 0)
    fh     = open(path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    return fh, writer


def get_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=["question_id"])["question_id"])
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset(dataset_path=DEFAULT_DATASET,
                  pdfs_dir=DEFAULT_PDFS_DIR,
                  output_csv=OUTPUT_CSV,
                  model_path=MODEL_PATH,
                  resume=True):

    df = (pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx")
          else pd.read_csv(dataset_path))
    logger.info(f"Loaded {len(df)} rows")

    done = get_done_ids(output_csv) if resume else set()
    logger.info(f"Already done: {len(done)}")

    tokenizer, model = load_model(model_path)
    fh, writer = open_csv(output_csv, append=resume)

    written = skipped_unanswerable = skipped_other = errors = 0

    # ── TEST MODE: first 3 rows only ──────────────────────────────────────────
    try:
        for _, row in tqdm(df.head(3).iterrows(), total=3):
            qid       = row["question_id"]
            doc_id    = row["doc_id"]
            question  = str(row["question_text"])
            gt_answer = str(row["gt_answer_snippet"])
            category  = row["category"]
            page_num  = row["gt_page_number"]

            print(f"\n{'='*60}")
            print(f"[{qid}] {question}")

            if qid in done:
                print(f"  -> already done, skipping")
                continue

            if category == "Unanswerable" or gt_answer == "Not Answered":
                print(f"  -> unanswerable, skipping")
                skipped_unanswerable += 1
                continue

            if pd.isna(page_num):
                print(f"  -> no page number, skipping")
                skipped_other += 1
                continue

            pdf_path = os.path.join(pdfs_dir, doc_id)
            if not os.path.exists(pdf_path):
                print(f"  -> PDF not found: {pdf_path}")
                skipped_other += 1
                continue

            context = extract_page(pdf_path, int(page_num))
            if not context:
                print(f"  -> empty page text, skipping")
                skipped_other += 1
                continue

            text = f"{context}\n\nQuestion: {question}\nAnswer: {gt_answer}"

            try:
                result = get_last_hidden_state(text, tokenizer, model)
                print(f"  -> seq_len={result['seq_len']}  target_index={result['target_index']}")
                print(f"  -> vector[:5] = {result['vector'][:5]}")

                out = {
                    "question_id"  : qid,
                    "doc_id"       : doc_id,
                    "category"     : category,
                    "question"     : question,
                    "gt_answer"    : gt_answer[:300],
                    "label"        : 0,
                    "seq_len"      : result["seq_len"],
                    "target_index" : result["target_index"],
                }
                for i, v in enumerate(result["vector"]):
                    out[f"v_{i}"] = round(float(v), 6)

                writer.writerow(out)
                fh.flush()
                written += 1
                print(f"  -> written to CSV")

            except Exception as e:
                logger.warning(f"[{qid}] failed: {e}")
                errors += 1

            gc.collect()

    # ── FULL RUN (all rows) — commented out for now ────────────────────────────
    # try:
    #     for _, row in tqdm(df.iterrows(), total=len(df)):
    #         qid       = row["question_id"]
    #         doc_id    = row["doc_id"]
    #         question  = str(row["question_text"])
    #         gt_answer = str(row["gt_answer_snippet"])
    #         category  = row["category"]
    #         page_num  = row["gt_page_number"]
    #
    #         if qid in done:
    #             continue
    #
    #         if category == "Unanswerable" or gt_answer == "Not Answered":
    #             skipped_unanswerable += 1
    #             continue
    #
    #         if pd.isna(page_num):
    #             logger.warning(f"[{qid}] no page number — skipping")
    #             skipped_other += 1
    #             continue
    #
    #         pdf_path = os.path.join(pdfs_dir, doc_id)
    #         if not os.path.exists(pdf_path):
    #             logger.warning(f"[{qid}] PDF not found: {pdf_path}")
    #             skipped_other += 1
    #             continue
    #
    #         context = extract_page(pdf_path, int(page_num))
    #         if not context:
    #             logger.warning(f"[{qid}] empty page text")
    #             skipped_other += 1
    #             continue
    #
    #         text = f"{context}\n\nQuestion: {question}\nAnswer: {gt_answer}"
    #
    #         try:
    #             result = get_last_hidden_state(text, tokenizer, model)
    #             out = {
    #                 "question_id"  : qid,
    #                 "doc_id"       : doc_id,
    #                 "category"     : category,
    #                 "question"     : question,
    #                 "gt_answer"    : gt_answer[:300],
    #                 "label"        : 0,
    #                 "seq_len"      : result["seq_len"],
    #                 "target_index" : result["target_index"],
    #             }
    #             for i, v in enumerate(result["vector"]):
    #                 out[f"v_{i}"] = round(float(v), 6)
    #             writer.writerow(out)
    #             fh.flush()
    #             written += 1
    #         except Exception as e:
    #             logger.warning(f"[{qid}] failed: {e}")
    #             errors += 1
    #
    #         gc.collect()

    finally:
        fh.close()

    logger.info(
        f"Done — written: {written} | "
        f"unanswerable (pending): {skipped_unanswerable} | "
        f"skipped other: {skipped_other} | "
        f"errors: {errors}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default=DEFAULT_DATASET)
    parser.add_argument("--pdfs_dir",  default=DEFAULT_PDFS_DIR)
    parser.add_argument("--output",    default=OUTPUT_CSV)
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--resume",    action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    build_dataset(
        dataset_path = args.dataset,
        pdfs_dir     = args.pdfs_dir,
        output_csv   = args.output,
        model_path   = args.model,
        resume       = not args.no_resume,
    )