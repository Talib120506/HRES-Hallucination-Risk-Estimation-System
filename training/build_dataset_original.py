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
DEFAULT_DATASET   = "../data/raw/TechManualQA_474 clean.xlsx"
DEFAULT_PDFS_DIR  = "../resources/pdfs"
OUTPUT_XLSX       = "../data/training_data/correct answers.xlsx"
MODEL_PATH        = "../models/gemma-2-2b-it"
HIDDEN_DIM        = 2304
MAX_CONTEXT_CHARS = 1600

# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_page(pdf_path: str, manual_page: int) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    idx = manual_page - 1   # convert 1-indexed to 0-based PDF index

    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""

    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CONTEXT_CHARS]

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "vector"       : vector,
        "seq_len"      : seq_len,
        "target_index" : target_index,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def get_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_excel(path, usecols=["question_id"])["question_id"])
    except Exception:
        return set()

def build_dataset(dataset_path=DEFAULT_DATASET,
                  pdfs_dir=DEFAULT_PDFS_DIR,
                  output_xlsx=OUTPUT_XLSX,
                  model_path=MODEL_PATH,
                  resume=True):

    df = (pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx")
          else pd.read_csv(dataset_path))
    logger.info(f"Loaded {len(df)} rows")

    done = get_done_ids(output_xlsx) if resume else set()
    logger.info(f"Already done: {len(done)}")

    tokenizer, model = load_model(model_path)
    
    results = []
    
    if resume and os.path.exists(output_xlsx):
        try:
            old_df = pd.read_excel(output_xlsx)
            results = old_df.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not load previous data from {output_xlsx}: {e}")

    written = skipped_unanswerable = skipped_other = errors = 0

    try:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            qid       = row["question_id"]
            doc_id    = row["doc_id"]
            question  = str(row["question_text"])
            gt_answer = str(row["gt_answer_snippet"])
            category  = row["category"]
            page_num  = row["gt_page_number"]

            if qid in done:
                continue

            if category == "Unanswerable" or gt_answer == "Not Answered":
                skipped_unanswerable += 1
                continue

            if pd.isna(page_num):
                logger.warning(f"[{qid}] no page number — skipping")
                skipped_other += 1
                continue

            pdf_path = os.path.join(pdfs_dir, doc_id)
            if not os.path.exists(pdf_path):
                logger.warning(f"[{qid}] PDF not found: {pdf_path}")
                skipped_other += 1
                continue

            context = extract_page(pdf_path, int(page_num))
            if not context:
                logger.warning(f"[{qid}] empty page text")
                skipped_other += 1
                continue

            # Ensure we're using a string instead of a float like 'nan'
            if str(context) == 'nan' or not context:
                user_msg = f"Question: {question}"
            else:
                user_msg = f"Context: {context}\n\nQuestion: {question}"

            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": gt_answer}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)

            try:
                result = get_last_hidden_state(text, tokenizer, model)
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
                
                results.append(out)
                done.add(qid)
                written += 1
                
                # Save periodically if needed, but for 474 rows, saving every 50 is fine.
                if written % 50 == 0:
                    pd.DataFrame(results).to_excel(output_xlsx, index=False)
                    
            except Exception as e:
                logger.warning(f"[{qid}] failed: {e}")
                errors += 1

            gc.collect()

    finally:
        if len(results) > 0:
            pd.DataFrame(results).to_excel(output_xlsx, index=False)
            logger.info(f"Final save to {output_xlsx}")

    logger.info(
        f"Done — written: {written} | "
        f"unanswerable (pending): {skipped_unanswerable} | "
        f"skipped other: {skipped_other} | "
        f"errors: {errors}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default=DEFAULT_DATASET)
    parser.add_argument("--pdfs_dir",  default=DEFAULT_PDFS_DIR)
    parser.add_argument("--output",    default=OUTPUT_XLSX)
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--resume",    action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    build_dataset(
        dataset_path = args.dataset,
        pdfs_dir     = args.pdfs_dir,
        output_xlsx  = args.output,
        model_path   = args.model,
        resume       = not args.no_resume,
    )
