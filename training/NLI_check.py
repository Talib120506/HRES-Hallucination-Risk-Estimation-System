"""
NLI_check.py
============
Hallucination verification: full-document FAISS retrieval + NLI.

Pipeline:
  1. Extract all PDF pages with PyMuPDF, clean and chunk them
  2. Embed all chunks with all-MiniLM-L6-v2 on GPU
  3. Build a FAISS IndexFlatIP per document (cosine similarity)
  4. For each question, retrieve candidate chunks
     - If gt_page_number is available, restrict search to that page's chunks
     - Otherwise, search across the entire document
  5. Run sentence-level NLI on each candidate chunk (with sliding windows)
  6. Pick the chunk+sentence combo with the highest entailment
  7. Output verdict: GROUNDED / UNCERTAIN / HALLUCINATION

Requirements:
    pip install faiss-cpu sentence-transformers transformers
    pip install PyMuPDF pandas openpyxl tqdm torch numpy
"""

import os
import re
import gc
import logging
import argparse

import fitz
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -- Config --------------------------------------------------------------------
DEFAULT_DATASET  = "../data/raw/TechManualQA_350.xlsx"
DEFAULT_PDFS_DIR = "../resources/pdfs"
OUTPUT_CSV       = "../data/results/nli_results.csv"

EMBED_MODEL   = "all-MiniLM-L6-v2"
NLI_MODEL     = "cross-encoder/nli-deberta-v3-small"
CHUNK_SIZE    = 100    # words per chunk
CHUNK_OVERLAP = 30     # word overlap
TOP_K         = 5      # chunks to retrieve (used when no page filter)

# deberta-v3-small label order: contradiction=0, entailment=1, neutral=2
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
VERDICT_MAP   = {
    "entailment"   : "GROUNDED",
    "neutral"      : "UNCERTAIN",
    "contradiction": "HALLUCINATION",
}


# -- PDF extraction ------------------------------------------------------------

def extract_pdf_pages(pdf_path: str) -> list:
    """Extract all non-empty pages. Returns [{page_num, text}]."""
    doc, pages = fitz.open(pdf_path), []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 30:
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


# -- Text processing ----------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove PDF noise: page numbers, language headers, figure references."""
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r'\d{1,3}', line):                          # page number
            continue
        if re.match(r'^(en|de|fr|es|it)\s+\S', line) and len(line) < 40:  # header
            continue
        if re.match(r'^\(?(figure|fig\.?)\s', line, re.IGNORECASE): # figure ref
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def chunk_text(text: str) -> list:
    """
    Split into overlapping chunks at sentence boundaries.
    Each chunk is a self-contained block of text -- no mid-sentence cuts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current = [], []
    for sentence in sentences:
        current.extend(sentence.split())
        if len(current) >= CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP:]

    if current:
        chunks.append(" ".join(current))

    return chunks


# -- FAISS document index ------------------------------------------------------

class FAISSDocIndex:
    """
    Full-document FAISS index.
    Stores all chunk embeddings + text + page numbers.
    Retrieval can optionally be scoped to a single page.
    """

    def __init__(self):
        self.index     = None
        self.vecs      = None   # (n, dim) normalized embeddings
        self.chunks    = []
        self.page_nums = []

    def build(self, chunks: list, page_nums: list, embedder):
        vecs = embedder.encode(
            chunks,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            device="cuda",
        ).astype(np.float32)

        dim        = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        self.vecs      = vecs
        self.chunks    = chunks
        self.page_nums = page_nums

    def retrieve(self, query_vec: np.ndarray, top_k: int = TOP_K,
                 page_filter: int = None):
        """
        Retrieve the top-k candidate chunks for a query.

        page_filter (int or None):
          - If set, only consider chunks from that page number.
            Returns ALL chunks from that page (sorted by similarity).
          - If None, FAISS searches across the entire document for top-k.

        Returns (list_of_chunks, list_of_page_nums).
        """
        if page_filter is not None:
            mask = [i for i, p in enumerate(self.page_nums) if p == page_filter]
            if not mask:
                return [], [page_filter]

            subset_vecs = self.vecs[mask]
            scores = (subset_vecs @ query_vec.T).squeeze()
            if scores.ndim == 0:
                scores = scores.reshape(1)
            order = np.argsort(-scores)
            return (
                [self.chunks[mask[i]] for i in order],
                [page_filter],
            )

        # No page filter -- search entire document
        _, indices = self.index.search(query_vec, top_k)
        idx = [i for i in indices[0] if i >= 0]
        if not idx:
            return [], []
        return (
            [self.chunks[i] for i in idx],
            [self.page_nums[i] for i in idx],
        )


def build_doc_index(pdf_path: str, embedder) -> FAISSDocIndex:
    """Build a full-document FAISS index from all pages."""
    chunks, page_nums = [], []
    for page in extract_pdf_pages(pdf_path):
        for chunk in chunk_text(clean_text(page["text"])):
            chunks.append(chunk)
            page_nums.append(page["page_num"])

    if not chunks:
        return None

    doc_index = FAISSDocIndex()
    doc_index.build(chunks, page_nums, embedder)
    return doc_index


# -- Embedding model -----------------------------------------------------------

def load_embedder():
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL, device="cuda")


# -- NLI model -----------------------------------------------------------------

_nli_tok   = None
_nli_model = None


def load_nli():
    global _nli_tok, _nli_model
    if _nli_model is not None:
        return _nli_tok, _nli_model
    logger.info(f"Loading NLI model: {NLI_MODEL}")
    _nli_tok   = AutoTokenizer.from_pretrained(NLI_MODEL)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    _nli_model.eval().to("cuda")
    logger.info("NLI model ready on GPU")
    return _nli_tok, _nli_model


def _nli_batch(premises: list, hypothesis: str) -> torch.Tensor:
    """Run NLI on a batch of (premise, hypothesis) pairs. Returns probs (n, 3)."""
    tok, model = load_nli()
    encodings = tok(
        premises,
        [hypothesis] * len(premises),
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        return torch.softmax(model(**encodings).logits, dim=-1).cpu()


def run_nli(premise: str, hypothesis: str) -> dict:
    """
    Sentence-level NLI with sliding windows.

    The small NLI model fails on long premises (~100 words) because
    irrelevant text in the chunk poisons the verdict. Fix:

    1. Split the chunk into individual sentences.
    2. Build candidate premises: each single sentence + each pair of
       consecutive sentences (window=2) + each triple (window=3).
       This handles answers that span 2-3 sentences in the manual.
    3. If the hypothesis (answer) contains multiple steps (newlines),
       split it into individual steps and check each step against
       each candidate premise. The final entailment score for multi-step
       answers is the minimum across steps (all steps must be grounded).
    4. Pick the candidate with the highest entailment score.
    """
    # Split chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', premise.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        sentences = [premise]

    # Build candidate premises: windows of 1, 2, and 3 consecutive sentences,
    # plus the full chunk itself (handles numbered lists like "1. ... 2. ..."
    # where sentence splitting breaks the structure)
    candidates = list(sentences)
    for w in (2, 3):
        for i in range(len(sentences) - w + 1):
            candidates.append(" ".join(sentences[i:i+w]))
    candidates.append(premise)  # full chunk as fallback

    # Check if answer is multi-step
    answer_steps = [s.strip() for s in re.split(r'\\n|\n', hypothesis) if s.strip()]
    if len(answer_steps) <= 1:
        answer_steps = [hypothesis]

    if len(answer_steps) == 1:
        # Single answer: pick candidate with best entailment
        all_probs = _nli_batch(candidates, hypothesis)
        best_idx = int(all_probs[:, 1].argmax())
        probs = all_probs[best_idx].tolist()
    else:
        # Multi-step answer: for each candidate, the entailment score is
        # the minimum entailment across all steps (all must be supported).
        # Pick the candidate with the highest such minimum.
        min_ent   = None
        sum_probs = None
        count     = 0
        for step in answer_steps:
            step_probs = _nli_batch(candidates, step)
            if min_ent is None:
                min_ent   = step_probs[:, 1].clone()
                sum_probs = step_probs.clone()
            else:
                min_ent = torch.min(min_ent, step_probs[:, 1])
                sum_probs += step_probs
            count += 1
        # Pick candidate with highest minimum entailment across steps
        best_idx = int(min_ent.argmax())
        # Average the probs across steps for reporting
        probs = (sum_probs[best_idx] / count).tolist()

    label = NLI_LABEL_MAP[probs.index(max(probs))]
    return {
        "label"        : label,
        "verdict"      : VERDICT_MAP[label],
        "entailment"   : round(probs[1], 4),
        "neutral"      : round(probs[2], 4),
        "contradiction": round(probs[0], 4),
    }


# -- Main ----------------------------------------------------------------------

def verify_dataset(dataset_path: str = DEFAULT_DATASET,
                   pdfs_dir    : str = DEFAULT_PDFS_DIR,
                   output_csv  : str = OUTPUT_CSV):

    df = (pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx")
          else pd.read_csv(dataset_path))

    df = df[
        (df["category"] != "Unanswerable") &
        (df["gt_answer_snippet"] != "Not Answered")
    ].reset_index(drop=True)
    logger.info(f"Rows to verify: {len(df)}")

    embedder = load_embedder()
    load_nli()

    # Build full-document FAISS indexes (lazy, one per doc_id)
    doc_indexes = {}

    # ── FULL RUN ──────────────────────────────────────────────────────────────
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
        qid       = row["question_id"]
        doc_id    = row["doc_id"]
        question  = str(row["question_text"])
        gt_answer = str(row["gt_answer_snippet"])

        pdf_path = os.path.join(pdfs_dir, doc_id)
        if not os.path.exists(pdf_path):
            logger.warning(f"[{qid}] PDF not found: {pdf_path}")
            continue

        # Build index for this doc if not already built
        if doc_id not in doc_indexes:
            idx = build_doc_index(pdf_path, embedder)
            if idx is None:
                logger.warning(f"[{qid}] No chunks from {doc_id}")
                continue
            doc_indexes[doc_id] = idx
            logger.info(f"Built index: {doc_id} ({len(idx.chunks)} chunks)")

        # Embed the question AND the answer (answer is from the PDF, so it
        # matches source chunks much better than the question phrasing)
        q_vec = embedder.encode(
            [question], normalize_embeddings=True, device="cuda",
        ).astype(np.float32)
        a_vec = embedder.encode(
            [gt_answer], normalize_embeddings=True, device="cuda",
        ).astype(np.float32)

        # Determine page filter (use gt_page_number if available)
        gt_page = row.get("gt_page_number")
        page_filter = None
        if gt_page is not None and not (isinstance(gt_page, float) and pd.isna(gt_page)):
            page_filter = int(gt_page)

        # Retrieve candidate chunks using BOTH question and answer embeddings.
        # Always include full-document retrieval (no page filter) as fallback,
        # in case gt_page_number is wrong or the answer spans a different page.
        seen = set()
        candidate_chunks, pages = [], []

        def _merge(chunks, pgs):
            for chunk in chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    candidate_chunks.append(chunk)
            for p in pgs:
                if p not in pages:
                    pages.append(p)

        # 1) Answer-based retrieval (full document) — highest priority
        a_all_chunks, a_all_pages = doc_indexes[doc_id].retrieve(
            a_vec, page_filter=None,
        )
        _merge(a_all_chunks, a_all_pages)

        # 2) Question-based retrieval (full document)
        q_all_chunks, q_all_pages = doc_indexes[doc_id].retrieve(
            q_vec, page_filter=None,
        )
        _merge(q_all_chunks, q_all_pages)

        # 3) Page-filtered retrieval (if page number available)
        if page_filter is not None:
            a_pg_chunks, a_pg_pages = doc_indexes[doc_id].retrieve(
                a_vec, page_filter=page_filter,
            )
            q_pg_chunks, q_pg_pages = doc_indexes[doc_id].retrieve(
                q_vec, page_filter=page_filter,
            )
            _merge(a_pg_chunks, a_pg_pages)
            _merge(q_pg_chunks, q_pg_pages)

        if not candidate_chunks:
            logger.warning(f"[{qid}] No chunks retrieved")
            continue

        # Run NLI on each candidate chunk, pick the one with best entailment
        best_nli   = None
        best_chunk = candidate_chunks[0]
        for chunk in candidate_chunks:
            nli = run_nli(premise=chunk, hypothesis=gt_answer)
            if best_nli is None or nli["entailment"] > best_nli["entailment"]:
                best_nli   = nli
                best_chunk = chunk

        results.append({
            "question_id"      : qid,
            "doc_id"           : doc_id,
            "category"         : row["category"],
            "question"         : question,
            "answer"           : gt_answer,
            "retrieved_context": best_chunk,
            "retrieved_pages"  : ", ".join(str(p) for p in pages),
            "nli_label"        : best_nli["label"],
            "verdict"          : best_nli["verdict"],
            "entailment"       : best_nli["entailment"],
            "neutral"          : best_nli["neutral"],
            "contradiction"    : best_nli["contradiction"],
        })

        gc.collect()

    out_df = pd.DataFrame(results)
    if out_df.empty:
        print("\nNo results -- check PDFs exist in pdfs_dir.")
        return out_df

    out_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(out_df)} rows -> {output_csv}")

    # Summary
    print("\n" + "=" * 55)
    print("VERDICT SUMMARY")
    print("=" * 55)
    for verdict, count in out_df["verdict"].value_counts().items():
        print(f"  {verdict:<15} {count:>4}  ({count/len(out_df)*100:.1f}%)")
    print("=" * 55)

    # Sample output
    print("\nRESULTS")
    print("-" * 55)
    for _, r in out_df.iterrows():
        print(f"Q  : {r['question'][:70]}")
        print(f"A  : {r['answer'][:80]}")
        print(f"PG : {r['retrieved_pages']}")
        print(f"CTX: {r['retrieved_context'][:120]}...")
        print(f"NLI: entailment={r['entailment']}  "
              f"neutral={r['neutral']}  "
              f"contradiction={r['contradiction']}")
        icon = {"GROUNDED": "OK", "UNCERTAIN": "??", "HALLUCINATION": "!!"}
        print(f"  -> {icon.get(r['verdict'], '?')} {r['verdict']}")
        print("-" * 55)

    return out_df


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hallucination verification: FAISS retrieval + NLI"
    )
    parser.add_argument("--dataset",  default=DEFAULT_DATASET)
    parser.add_argument("--pdfs_dir", default=DEFAULT_PDFS_DIR)
    parser.add_argument("--output",   default=OUTPUT_CSV)
    args = parser.parse_args()

    verify_dataset(
        dataset_path = args.dataset,
        pdfs_dir     = args.pdfs_dir,
        output_csv   = args.output,
    )
