"""
build_dataset_optimized.py
==========================
Optimized for GPU Batch Processing and faster I/O.
Extracts the final hidden state vectors for hallucination detection.
"""

import os
import gc
import logging
import argparse

import pandas as pd
import torch
import fitz  # PyMuPDF
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CORRECT_XLSX   = "../data/training_data/correct answers.xlsx"
INCORRECT_XLSX = "../data/training_data/Incorrect answers.xlsx"
DEFAULT_PDFS   = "../resources/pdfs"
OUTPUT_XLSX    = "../data/processed/features_correct_incorrect.xlsx"
MODEL_PATH     = "../models/gemma-2-2b-it"
MAX_CTX_CHARS  = 1600

# GPU OPTIMIZATIONS
# RTX 3050 (4GB) is memory constrained. Keep batch size small.
BATCH_SIZE     = 1


# ── PDF extraction (Cached) ───────────────────────────────────────────────────
_pdf_cache = {}

def get_cached_page_text(pdf_path: str, manual_page: int) -> str:
    """Extracts text, keeping the PDF open in memory to prevent slow disk I/O."""
    if pdf_path not in _pdf_cache:
        if not os.path.exists(pdf_path):
            return ""
        _pdf_cache[pdf_path] = fitz.open(pdf_path)
    
    doc = _pdf_cache[pdf_path]
    idx = int(manual_page) - 1
    
    if idx < 0 or idx >= len(doc):
        return ""
        
    return doc[idx].get_text().strip()[:MAX_CTX_CHARS]


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
        
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token exists for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # OPTIMIZATION: 4-bit Quantization (Crucial for 4GB RTX 3050)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="cuda",
            attn_implementation="flash_attention_2" # Massive speedup if supported
        )
        logger.info("Model loaded in 4-bit with Flash Attention 2 on GPU")
    except Exception as e:
        logger.warning(f"Flash Attention not supported, falling back to standard attention: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="cuda",
        )
        
    model.eval()
    return tokenizer, model


# ── Forward pass (Batched) ────────────────────────────────────────────────────
def find_last_meaningful_token(input_ids: torch.Tensor, tokenizer) -> int:
    special_ids = set(tokenizer.all_special_ids)
    for i in range(len(input_ids) - 1, -1, -1):
        tok_id  = input_ids[i].item()
        tok_str = tokenizer.decode(tok_id).strip()

        if tok_id in special_ids or not any(c.isalnum() for c in tok_str):
            continue
        return i
    return 0


def get_last_hidden_state_batched(texts: list, tokenizer, model) -> list:
    """Processes multiple texts at the exact same time on the GPU."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,       # Required for batching
        truncation=True,
        max_length=1900,
    ).to(model.device)

    # OPTIMIZATION: inference_mode is faster and uses less memory than no_grad
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    batch_results = []
    
    for i in range(len(texts)):
        input_ids    = inputs["input_ids"][i]
        seq_len      = (input_ids != tokenizer.pad_token_id).sum().item()
        target_index = find_last_meaningful_token(input_ids, tokenizer)
        
        # Extract vector and move it to CPU RAM immediately to free VRAM
        vector = outputs.hidden_states[-1][i, target_index, :].float().cpu().numpy()
        
        batch_results.append({
            "vector": vector,
            "seq_len": seq_len,
            "target_index": target_index,
        })

    del outputs, inputs
    return batch_results


# ── Process one set of answers ────────────────────────────────────────────────
def make_row(item, result):
    row = {
        "question_id":  item["qid"],
        "doc_id":       item["doc_id"],
        "question":     item["question"],
        "answer":       item["answer"][:300],
        "answer_type":  item["answer_type"],
        "label":        item["label"],
        "seq_len":      result["seq_len"],
        "target_index": result["target_index"],
    }
    for i, v in enumerate(result["vector"]):
        row[f"v_{i}"] = round(float(v), 6)
    return row


def process_answers(df, answer_type, label, pdfs_dir,
                    tokenizer, model, all_rows, done_set, counters, output_xlsx):
    
    # STEP 1: PRE-PROCESS DATA ON CPU
    valid_items = []
    logger.info(f"Preparing {answer_type} data...")
    
    for _, row in df.iterrows():
        qid = row["question_id"]
        if f"{qid}_{label}" in done_set:
            continue

        answer = str(row.get("hallucinated_answer" if label == 1 else "gt_answer_snippet", ""))
        page_num = row.get("gt_page_number", float('nan'))
        
        if pd.isna(page_num) or answer == "Not Answered" or not answer:
            counters["skipped_unanswerable"] += 1
            continue

        pdf_path = os.path.join(pdfs_dir, str(row["doc_id"]))
        context = get_cached_page_text(pdf_path, page_num)
        
        if not context:
            counters["skipped_other"] += 1
            continue

        question = str(row["question_text"])
        user_msg = f"Question: {question}" if not context else f"Context: {context}\n\nQuestion: {question}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer}
        ]
        
        valid_items.append({
            "qid": qid,
            "doc_id": row["doc_id"],
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "label": label,
            "formatted_text": tokenizer.apply_chat_template(messages, tokenize=False)
        })

    # STEP 2: BATCH INFERENCE ON GPU
    if not valid_items:
        logger.info(f"No new {answer_type} items to process.")
        return

    logger.info(f"Starting GPU inference for {len(valid_items)} items (Batch Size: {BATCH_SIZE})...")
    
    for i in tqdm(range(0, len(valid_items), BATCH_SIZE), desc=f"Batches ({answer_type})"):
        batch_items = valid_items[i : i + BATCH_SIZE]
        batch_texts = [item["formatted_text"] for item in batch_items]
        
        try:
            results = get_last_hidden_state_batched(batch_texts, tokenizer, model)
            
            for item, res in zip(batch_items, results):
                all_rows.append(make_row(item, res))
                done_set.add(f"{item['qid']}_{label}")
                counters["written"] += 1

            # Save periodically
            if counters["written"] > 0 and counters["written"] % 100 == 0:
                pd.DataFrame(all_rows).to_excel(output_xlsx, index=False, engine="openpyxl")
                # Only empty cache during saves to prevent massive slowdowns
                torch.cuda.empty_cache() 
                gc.collect()

        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            counters["errors"] += len(batch_items)
            # If a batch fails, clear VRAM to prevent cascading OOM errors
            torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────
def build_dataset(correct_xlsx=CORRECT_XLSX,
                  incorrect_xlsx=INCORRECT_XLSX,
                  pdfs_dir=DEFAULT_PDFS,
                  output_xlsx=OUTPUT_XLSX,
                  model_path=MODEL_PATH,
                  resume=True):

    # Load both Excel files directly
    df_correct   = pd.read_excel(correct_xlsx)
    df_incorrect = pd.read_excel(incorrect_xlsx)
    logger.info(f"Loaded | Correct rows: {len(df_correct)}  |  Incorrect rows: {len(df_incorrect)}")

    all_rows = []
    done_set = set()
    
    # Resume logic
    if resume and os.path.exists(output_xlsx):
        try:
            old_df = pd.read_excel(output_xlsx)
            all_rows = old_df.to_dict("records")
            for r in all_rows:
                done_set.add(f"{r['question_id']}_{r['label']}")
            logger.info(f"Already done (resumed): {len(done_set)} rows")
        except Exception as e:
            logger.warning(f"Could not load previous data from {output_xlsx}: {e}")

    # Load model
    tokenizer, model = load_model(model_path)

    counters = {"written": 0, "skipped_unanswerable": 0,
                "skipped_other": 0, "errors": 0}

    try:
        # Process correct answers (label = 0)
        process_answers(df_correct, "correct", 0, pdfs_dir,
                        tokenizer, model, all_rows, done_set, counters, output_xlsx)

        # Process incorrect answers (label = 1)
        process_answers(df_incorrect, "incorrect", 1, pdfs_dir,
                        tokenizer, model, all_rows, done_set, counters, output_xlsx)
    finally:
        # Save final Excel file
        if len(all_rows) > 0:
            logger.info("Saving final dataset to disk...")
            df_out = pd.DataFrame(all_rows)
            df_out.to_excel(output_xlsx, index=False, engine="openpyxl")
            
            label_counts = dict(df_out["label"].value_counts())
            logger.info(
                f"Done - written this session: {counters['written']} rows to {output_xlsx}\n"
                f"Final labels count: {label_counts} | "
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
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    build_dataset(
        correct_xlsx   = args.correct,
        incorrect_xlsx = args.incorrect,
        pdfs_dir       = args.pdfs_dir,
        output_xlsx    = args.output,
        model_path     = args.model,
        resume         = not args.no_resume,
    )