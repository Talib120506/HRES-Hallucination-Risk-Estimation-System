import os
import re
import gc
import logging
import pickle

import fitz
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "models", "nli_index")

EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "D:\\Hallucination test\\models/nli_finetuned/best"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE     = 100
CHUNK_OVERLAP  = 30
TOP_K_RETRIEVAL      = 10   # candidates retrieved before re-ranking
TOP_K_AFTER_RERANK   = 5    # candidates passed to NLI after re-ranking

# Relaxed similarity threshold from strict 0.7411 so edge cases don't fail early
SIMILARITY_THRESHOLD = 0.70

# Feature toggles — set to False to disable individual improvements
USE_QUESTION_AWARE_HYPOTHESIS = True   # Improvement 2
USE_HYBRID_RETRIEVAL          = True   # Improvement 3  (needs rank_bm25)
USE_RERANKER                  = True   # Improvement 5  (cross-encoder re-rank)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "models", "nli_index")

# DeBERTa label order: contradiction=0, entailment=1, neutral=2
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
VERDICT_MAP   = {
    "entailment":    "GROUNDED",
    "neutral":       "UNCERTAIN",
    "contradiction": "HALLUCINATION",
    "UNCERTAIN_LEANING_GROUNDED": "UNCERTAIN",
}

ENTAILMENT_THRESHOLD = 0.95

def nli_verdict_from_scores(entailment: float, neutral: float, contradiction: float) -> str:
    """
    Entailment-threshold verdict (replaces argmax verdict).
    Only GROUNDED if entailment >= 0.99. Everything below is UNCERTAIN or HALLUCINATION.
    Data shows: label-0 avg entailment=0.946, label-1 avg entailment=0.617.
    """
    if entailment >= ENTAILMENT_THRESHOLD:
        return "GROUNDED"
    elif contradiction > entailment and contradiction > neutral:
        return "HALLUCINATION"
    elif neutral >= entailment and neutral >= contradiction:
        return "UNCERTAIN"
    else:
        return "UNCERTAIN_LEANING_GROUNDED"

# ── Model singletons ──────────────────────────────────────────────────────────

_embedder   = None
_nli_tok    = None
_nli_model  = None
_reranker   = None

def get_embedder():
    global _embedder
    if _embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedder: {EMBED_MODEL} on {device}")
        _embedder = SentenceTransformer(EMBED_MODEL, device=device)
    return _embedder

def get_nli():
    global _nli_tok, _nli_model
    if _nli_model is None:
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        _nli_tok   = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        _nli_model.eval().to(device)
    return _nli_tok, _nli_model

def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANKER:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Re-ranker model: {RERANK_MODEL} on {device}")
        _reranker = CrossEncoder(RERANK_MODEL, device=device)
    return _reranker

# ── PDF processing ────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: str) -> list:
    """Extract all non-empty pages. Returns [{page_num, text}]."""
    doc, pages = fitz.open(pdf_path), []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 30:
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages

def clean_text(text: str) -> str:
    """Remove PDF noise: page numbers, language headers, figure references."""
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r'\d{1,3}', line):
            continue
        if re.match(r'^(en|de|fr|es|it)\s+\S', line) and len(line) < 40:
            continue
        if re.match(r'^\(?(figure|fig\.?)\s', line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return " ".join(cleaned)

def chunk_text(text: str) -> list:
    """Split into overlapping ~100-word chunks at sentence boundaries."""
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

# ── FAISS index ───────────────────────────────────────────────────────────────

class FAISSDocIndex:
    """Full-document FAISS index with BM25 support for hybrid retrieval."""
    def __init__(self):
        self.index     = None
        self.vecs      = None
        self.chunks    = []
        self.page_nums = []
        self.bm25      = None

    def build(self, chunks: list, page_nums: list, embedder):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vecs = embedder.encode(
            chunks, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True, device=device,
        ).astype(np.float32)
        dim        = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        self.vecs      = vecs
        self.chunks    = chunks
        self.page_nums = page_nums

    def _get_bm25(self):
        if self.bm25 is None:
            if USE_HYBRID_RETRIEVAL:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [chunk.lower().split() for chunk in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
        return self.bm25

    def retrieve_dense(self, query_vec: np.ndarray, top_k: int) -> list:
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i >= 0:
                results.append((self.chunks[i], float(d)))
        return results

    def retrieve_bm25(self, query_text: str, top_k: int) -> list:
        bm25 = self._get_bm25()
        if not bm25:
            return []
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_n]

    def retrieve_hybrid(self, query_vec: np.ndarray, query_text: str,
                        top_k: int = TOP_K_RETRIEVAL) -> list:
        dense_results = self.retrieve_dense(query_vec, top_k)
        bm25_results = self.retrieve_bm25(query_text, top_k)
        
        # RRF Fusion
        rrf_scores = {}
        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1.0 / (60 + rank)
        for rank, (chunk, _) in enumerate(bm25_results):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1.0 / (60 + rank)
            
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_chunks[:top_k]]

    def retrieve(self, query_vec: np.ndarray, query_text: str = "",
                 top_k: int = TOP_K_RETRIEVAL) -> list:
        if USE_HYBRID_RETRIEVAL:
            return self.retrieve_hybrid(query_vec, query_text, top_k)
        else:
            return [c for c, _ in self.retrieve_dense(query_vec, top_k)]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "meta.pkl"), "wb") as f:
            pickle.dump({"vecs": self.vecs, "chunks": self.chunks, "page_nums": self.page_nums}, f)

    def load(self, save_dir: str):
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "meta.pkl"), "rb") as f:
            data = pickle.load(f)
            self.vecs      = data.get("vecs")
            self.chunks    = data["chunks"]
            self.page_nums = data["page_nums"]

def build_or_load_index(pdf_path: str, doc_id: str, embedder,
                        index_dir: str = INDEX_DIR) -> FAISSDocIndex:
    save_dir = os.path.join(index_dir, doc_id.replace(".pdf", ""))
    idx = FAISSDocIndex()
    if os.path.exists(os.path.join(save_dir, "index.faiss")) and os.path.exists(os.path.join(save_dir, "meta.pkl")):
        idx.load(save_dir)
        return idx
    chunks, page_nums = [], []
    for page in extract_pdf_pages(pdf_path):
        for chunk in chunk_text(clean_text(page["text"])):
            chunks.append(chunk)
            page_nums.append(page["page_num"])
    if not chunks:
        return None
    idx.build(chunks, page_nums, embedder)
    idx.save(save_dir)
    return idx

def _build_hypothesis(question: str, answer: str) -> str:
    if USE_QUESTION_AWARE_HYPOTHESIS and question:
        return f"Given the question: {question}  The answer is: {answer}"
    return answer.strip()

def _nli_batch(premises: list, hypothesis: str, batch_size: int = 16) -> torch.Tensor:
    tok, model = get_nli()
    device = next(model.parameters()).device
    all_probs = []
    for i in range(0, len(premises), batch_size):
        batch = premises[i:i + batch_size]
        enc = tok(
            batch, [hypothesis] * len(batch),
            return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu()
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0)

def run_nli(premise: str, hypothesis: str) -> dict:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', premise) if len(s.strip()) > 5]
    if not sents:
        sents = [premise]
    windows = list(sents)
    for i in range(len(sents) - 1):
        windows.append(sents[i] + " " + sents[i + 1])
    for i in range(len(sents) - 2):
        windows.append(sents[i] + " " + sents[i + 1] + " " + sents[i + 2])
    windows.append(premise)
    windows = list(set(windows))   # deduplicate
    probs_batch = _nli_batch(windows, hypothesis)
    best_idx    = probs_batch[:, 1].argmax().item()
    best_probs  = probs_batch[best_idx].tolist()
    label       = NLI_LABEL_MAP[best_probs.index(max(best_probs))]
    return {
        "label":         label,
        "verdict":       VERDICT_MAP[label],
        "entailment":    round(best_probs[1], 4),
        "neutral":       round(best_probs[2], 4),
        "contradiction": round(best_probs[0], 4),
    }

def rerank_chunks(chunks: list, question: str, answer: str,
                  top_k: int = TOP_K_AFTER_RERANK) -> list:
    if not USE_RERANKER or len(chunks) <= top_k:
        return chunks[:top_k]
    reranker = get_reranker()
    query = f"{question} {answer}".strip()
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]

def blackbox_predict_unified(doc_index: FAISSDocIndex,
                              question: str,
                              answer: str,
                              similarity_threshold: float = SIMILARITY_THRESHOLD
                              ) -> dict:
    embedder = get_embedder()
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    q_vec = embedder.encode([question], normalize_embeddings=True, device=device).astype(np.float32)
    a_vec = embedder.encode([answer], normalize_embeddings=True, device=device).astype(np.float32)

    seen = set()
    candidate_chunks = []
    max_score = -1.0

    for vec, text in [(a_vec, answer), (q_vec, question)]:
        chunks = doc_index.retrieve(vec, text, TOP_K_RETRIEVAL)
        # Also find max_score via dense
        dense_scores = doc_index.retrieve_dense(vec, TOP_K_RETRIEVAL)
        for _, sc in dense_scores:
            max_score = max(max_score, sc)
            
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                candidate_chunks.append(chunk)

    if not candidate_chunks:
        return {
            "verdict": "UNCERTAIN", "entailment": 0.0, "neutral": 1.0, "contradiction": 0.0,
            "retrieved_context": "[no chunks]", "max_similarity": max_score,
            "method_flags": {"similarity_threshold": similarity_threshold}
        }

    if max_score < similarity_threshold:
        return {
            "verdict": "UNCERTAIN", "entailment": 0.0, "neutral": 1.0, "contradiction": 0.0,
            "retrieved_context": f"[UNSUPPORTED] max_sim={max_score:.3f} < threshold={similarity_threshold}",
            "max_similarity": max_score,
            "method_flags": {"similarity_threshold": similarity_threshold}
        }

    candidate_chunks = rerank_chunks(candidate_chunks, question, answer, top_k=TOP_K_AFTER_RERANK)
    hypothesis = _build_hypothesis(question, answer)

    best_nli   = None
    best_chunk = candidate_chunks[0]
    for chunk in candidate_chunks:
        nli = run_nli(premise=chunk, hypothesis=hypothesis)
        if best_nli is None or nli["entailment"] > best_nli["entailment"]:
            best_nli   = nli
            best_chunk = chunk

    gc.collect()
    return {
        "verdict":           nli_verdict_from_scores(best_nli["entailment"], best_nli["neutral"], best_nli["contradiction"]),
        "entailment":        best_nli["entailment"],
        "neutral":           best_nli["neutral"],
        "contradiction":     best_nli["contradiction"],
        "retrieved_context": best_chunk[:500],
        "max_similarity":    round(max_score, 4),
        "method_flags": {
            "hybrid_retrieval":          USE_HYBRID_RETRIEVAL,
            "reranker":                  USE_RERANKER,
            "question_aware_hypothesis": USE_QUESTION_AWARE_HYPOTHESIS,
            "similarity_threshold":      similarity_threshold,
        },
    }
