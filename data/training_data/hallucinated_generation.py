"""
generate_hallucinated_answers_groq.py
======================================
Takes the cleaned TechManualQA dataset and generates a plausible but
factually incorrect answer for each question using the Groq API.

Uses pre-built FAISS indexes from nli_index/ (index.faiss + meta.pkl)
to retrieve the most relevant context chunk per question — no PDF or
page number extraction needed at runtime.
"""

import os
import gc
import time
import pickle
import argparse
import logging

import numpy as np
import pandas as pd
import faiss
import torch
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT   = os.path.join(BASE_DIR, "data", "raw", "TechManualQA_474 clean.xlsx")
DEFAULT_OUTPUT  = os.path.join(BASE_DIR, "data", "training_data", "Incorrect answers.xlsx")
NLI_INDEX_DIR   = os.path.join(BASE_DIR, "models", "nli_index")

EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
TOP_K           = 3        # top chunks to retrieve and concatenate as context
MAX_CTX_CHARS   = 1800     # cap context sent to the LLM
MODEL           = "llama-3.3-70b-versatile" # High-capability Groq model
MAX_RETRIES     = 5
RETRY_DELAY     = 20


# ── Load pre-built FAISS index ────────────────────────────────────────────────

_index_cache: dict = {}

def load_doc_index(doc_id: str, nli_index_dir: str):
    """
    Load index.faiss + meta.pkl for a document.
    meta.pkl contains: {"chunks": [...], "page_nums": [...]}
    Returns (faiss_index, chunks, page_nums) or None if not found.
    """
    if doc_id in _index_cache:
        return _index_cache[doc_id]

    folder_name = os.path.splitext(doc_id)[0]
    index_path  = os.path.join(nli_index_dir, folder_name, "index.faiss")
    meta_path   = os.path.join(nli_index_dir, folder_name, "meta.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        logger.warning(f"Index not found for {doc_id} — expected at {index_path}")
        return None

    idx = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    result = (idx, meta["chunks"], meta["page_nums"])
    _index_cache[doc_id] = result
    logger.info(f"Loaded index: {doc_id} ({len(meta['chunks'])} chunks)")
    return result


# ── Retrieve best context via FAISS ──────────────────────────────────────────

def get_context(doc_id: str, question: str, answer: str,
                embedder, nli_index_dir: str,
                top_k: int = TOP_K,
                max_chars: int = MAX_CTX_CHARS) -> str:
    """
    Query the pre-built FAISS index with question+answer.
    Returns the top-k most relevant chunks concatenated.
    """
    loaded = load_doc_index(doc_id, nli_index_dir)
    if loaded is None:
        return ""

    idx, chunks, page_nums = loaded

    # Question + answer gives richer retrieval signal than question alone
    query = f"{question} {answer}"
    q_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    k = min(top_k, idx.ntotal)
    _, indices = idx.search(q_vec, k)

    retrieved, seen = [], set()
    for i in indices[0]:
        if i >= 0 and chunks[i] not in seen:
            retrieved.append(chunks[i])
            seen.add(chunks[i])

    context = "\n\n".join(retrieved)
    return context[:max_chars]


# ── Groq API call ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a dataset generation assistant for hallucination detection research.
Your job is to generate a WRONG answer that:
1. Is on the same topic as the correct answer
2. Matches the FORMAT and LENGTH of the correct answer exactly
3. Sounds plausible and realistic at first glance
4. Contains a subtle but clear factual error (wrong number, wrong part, wrong step, wrong condition, wrong specification, etc.)
5. Does NOT simply negate the correct answer — it must be a believable alternative
6. Is grounded in the same domain as the source document context

Return ONLY the wrong answer text. No explanation, no prefix, no quotes."""


def generate_hallucinated_answer(client: OpenAI,
                                  question: str,
                                  correct_answer: str,
                                  context: str) -> str:
    """Call Groq to generate a plausible but wrong answer."""

    user_message = (
        f"SOURCE DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"CORRECT ANSWER: {correct_answer}\n\n"
        f"Generate a WRONG answer that is on-topic, matches the exact format "
        f"and length of the correct answer, but contains a subtle factual error."
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.7 
            )
            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            logger.warning(f"Rate limit hit, waiting {wait}s...")
            time.sleep(wait)

        except openai.APIError as e:
            logger.warning(f"API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate hallucinated answers using pre-built FAISS indexes and Groq"
    )
    parser.add_argument("--input",     default=DEFAULT_INPUT,  help="Clean input .xlsx")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT, help="Output .xlsx path")
    parser.add_argument("--nli_index", default=NLI_INDEX_DIR,  help="Path to nli_index/ folder")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip rows that already have a hallucinated_answer")
    args = parser.parse_args()

    # Load the existing progress file if we are resuming
    if os.path.exists(args.output):
        df = pd.read_excel(args.output)
        logger.info(f"Resuming! Loaded {len(df)} rows from {args.output}")
    else:
        df = pd.read_excel(args.input)
        logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Add output columns if not present
    if "hallucinated_answer" not in df.columns:
        df["hallucinated_answer"] = ""
    if "hallucinated" not in df.columns:
        df["hallucinated"] = 1

    # Load embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedder ({EMBED_MODEL}) on {device}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # Init Groq client via OpenAI SDK
    api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    counters = {"written": 0, "skipped": 0, "failed": 0}

    for idx, row in df.iterrows():
        
        # DYNAMIC CHECK: If the row already has a hallucinated answer, skip it
        existing_answer = str(row.get("hallucinated_answer", "")).strip()
        if existing_answer != "" and existing_answer != "nan":
            counters["skipped"] += 1
            continue

        qid      = row["question_id"]
        doc_id   = row["doc_id"]
        question = str(row["question_text"])
        answer   = str(row["gt_answer_snippet"])

        # Retrieve context from pre-built FAISS index
        context = get_context(doc_id, question, answer, embedder, args.nli_index)
        if not context:
            logger.warning(f"[{qid}] no context — skipping")
            counters["failed"] += 1
            continue

        # Generate hallucinated answer
        hallucinated = generate_hallucinated_answer(client, question, answer, context)
        if not hallucinated:
            logger.warning(f"[{qid}] API returned empty — skipping")
            counters["failed"] += 1
            continue

        df.at[idx, "hallucinated_answer"] = hallucinated
        df.at[idx, "hallucinated"]        = 1
        counters["written"] += 1

        logger.info(
            f"[{idx+1}/{len(df)}] {qid}\n"
            f"  Correct     : {answer[:80]}\n"
            f"  Hallucinated: {hallucinated[:80]}"
        )

        # Checkpoint save every 10 rows
        if counters["written"] > 0 and counters["written"] % 10 == 0:
            df.to_excel(args.output, index=False, engine="openpyxl")
            logger.info(f"  Checkpoint -> {args.output}")

        # Proactively slow down to avoid hitting the API limit
        time.sleep(2) 

        gc.collect()

    # Final save
    df.to_excel(args.output, index=False, engine="openpyxl")
    logger.info(
        f"\nDone — written: {counters['written']} | "
        f"skipped (already filled): {counters['skipped']} | "
        f"failed: {counters['failed']}\n"
        f"Saved -> {args.output}"
    )

if __name__ == "__main__":
    main()

