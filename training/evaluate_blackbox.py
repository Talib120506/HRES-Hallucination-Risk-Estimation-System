"""
evaluate_blackbox.py
====================
Evaluates the blackbox (NLI) pipeline from NLI_check.py and app.py against
all labeled Q&A pairs in the features dataset (label 0 = correct, label 1 = incorrect).

Key logic:
  - Label 0 (correct answers): answers actually exist in the source PDFs.
    The blackbox SHOULD return GROUNDED (high entailment).
    Ground truth for evaluation = GROUNDED.

  - Label 1 (incorrect / hallucinated): answers are fabricated.
    The blackbox SHOULD return HALLUCINATION or UNCERTAIN.
    Ground truth for evaluation = HALLUCINATION.

Prediction mapping for binary metrics:
  GROUNDED        -> predicted CORRECT  (0)
  UNCERTAIN       -> predicted CORRECT  (0)   [conservative: doubt ≠ proof of hallucination]
  HALLUCINATION   -> predicted HALLUCINATED (1)

Outputs
-------
  evaluation_results.xlsx   – full per-row results with verdicts, scores, correctness
  evaluation_summary.txt    – classification report, confusion matrix, key metrics

Usage
-----
  python evaluate_blackbox.py [--dataset PATH] [--pdfs_dir PATH] [--output_xlsx PATH] [--output_txt PATH]

Defaults assume you run from inside the project root (same as NLI_check.py):
  dataset   : data/processed/features_correct_incorrect.xlsx
  pdfs_dir  : resources/pdfs
"""

import os
import re
import gc
import logging
import argparse
import pickle
import fitz
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (mirrors NLI_check.py exactly for consistency)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET   = os.path.join(BASE_DIR, "data", "processed", "features_correct_incorrect.xlsx")
DEFAULT_PDFS_DIR  = os.path.join(BASE_DIR, "resources", "pdfs")
DEFAULT_OUT_XLSX  = os.path.join(BASE_DIR, "data", "results", "evaluation_results.xlsx")
DEFAULT_OUT_TXT   = os.path.join(BASE_DIR, "data", "results", "evaluation_summary.txt")

EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
NLI_MODEL      = "cross-encoder/nli-deberta-v3-base"
CHUNK_SIZE     = 100
CHUNK_OVERLAP  = 30
TOP_K          = 5
INDEX_DIR      = os.path.join(BASE_DIR, "models", "nli_index")

# DeBERTa-v3 label order: contradiction=0, entailment=1, neutral=2
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
VERDICT_MAP   = {
    "entailment":    "GROUNDED",
    "neutral":       "UNCERTAIN",
    "contradiction": "HALLUCINATION",
}

# Threshold below which similarity => automatic HALLUCINATION (mirrors app.py)
SIMILARITY_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_embedder  = None
_nli_tok   = None
_nli_model = None


def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return _embedder


def get_nli():
    global _nli_tok, _nli_model
    if _nli_model is None:
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        _nli_tok   = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        _nli_model.eval().to(device)
        logger.info(f"NLI model ready on {device}")
    return _nli_tok, _nli_model


# ---------------------------------------------------------------------------
# PDF helpers  (identical to NLI_check.py)
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: str) -> list:
    doc, pages = fitz.open(pdf_path), []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 30:
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
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


# ---------------------------------------------------------------------------
# FAISS index  (identical to NLI_check.py — load cache if available)
# ---------------------------------------------------------------------------

class FAISSDocIndex:
    def __init__(self):
        self.index     = None
        self.vecs      = None
        self.chunks    = []
        self.page_nums = []

    def build(self, chunks, page_nums, embedder):
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

    def retrieve(self, query_vec, top_k=TOP_K, page_filter=None):
        if page_filter is not None:
            mask = [i for i, p in enumerate(self.page_nums) if p == page_filter]
            if not mask:
                return [], [page_filter]
            subset_vecs = self.vecs[mask]
            scores = (subset_vecs @ query_vec.T).squeeze()
            if scores.ndim == 0:
                scores = scores.reshape(1)
            order = np.argsort(-scores)
            return ([self.chunks[mask[i]] for i in order], [page_filter])
        _, indices = self.index.search(query_vec, top_k)
        idx = [i for i in indices[0] if i >= 0]
        if not idx:
            return [], []
        return ([self.chunks[i] for i in idx], [self.page_nums[i] for i in idx])

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "meta.pkl"), "wb") as f:
            pickle.dump({"vecs": self.vecs, "chunks": self.chunks, "page_nums": self.page_nums}, f)

    def load(self, save_dir):
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "meta.pkl"), "rb") as f:
            data = pickle.load(f)
            self.vecs      = data.get("vecs")
            self.chunks    = data["chunks"]
            self.page_nums = data["page_nums"]
        if self.vecs is None and self.index is not None:
            try:
                self.vecs = faiss.rev_swig_ptr(
                    self.index.get_xb(),
                    self.index.ntotal * self.index.d
                ).reshape(self.index.ntotal, self.index.d)
            except Exception as e:
                logger.warning(f"Could not reconstruct vecs from FAISS: {e}")


def get_doc_index(pdf_path, doc_id, embedder):
    save_dir   = os.path.join(INDEX_DIR, doc_id.replace(".pdf", ""))
    doc_index  = FAISSDocIndex()
    index_file = os.path.join(save_dir, "index.faiss")
    meta_file  = os.path.join(save_dir, "meta.pkl")
    if os.path.exists(index_file) and os.path.exists(meta_file):
        logger.info(f"Loading cached index: {save_dir}")
        doc_index.load(save_dir)
        return doc_index

    chunks, page_nums = [], []
    for page in extract_pdf_pages(pdf_path):
        for chunk in chunk_text(clean_text(page["text"])):
            chunks.append(chunk)
            page_nums.append(page["page_num"])
    if not chunks:
        return None
    doc_index.build(chunks, page_nums, embedder)
    doc_index.save(save_dir)
    return doc_index


# ---------------------------------------------------------------------------
# NLI helpers  (identical to NLI_check.py run_nli with sliding windows)
# ---------------------------------------------------------------------------

def _nli_batch(premises, hypothesis, batch_size=16):
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


def run_nli(premise, hypothesis):
    """
    Advanced sliding-window NLI (mirrors NLI_check.py exactly).
    Uses 1-, 2-, 3-sentence windows + full chunk to find best entailment.
    """
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', premise) if len(s.strip()) > 5]
    if not sents:
        sents = [premise]
    windows = list(sents)
    for i in range(len(sents) - 1):
        windows.append(sents[i] + " " + sents[i + 1])
    for i in range(len(sents) - 2):
        windows.append(sents[i] + " " + sents[i + 1] + " " + sents[i + 2])
    windows.append(premise)
    windows = list(set(windows))

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


# ---------------------------------------------------------------------------
# Per-row blackbox prediction  (mirrors app.py blackbox_predict)
# ---------------------------------------------------------------------------

def blackbox_predict(doc_index: FAISSDocIndex, question, answer):
    """
    Returns (verdict_str, entailment, neutral, contradiction, retrieved_context, max_sim)
    """
    embedder = get_embedder()
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    q_vec = embedder.encode([question], normalize_embeddings=True, device=device).astype(np.float32)
    a_vec = embedder.encode([answer],   normalize_embeddings=True, device=device).astype(np.float32)

    seen, candidate_chunks = set(), []
    max_score = -1.0

    for query_vec in [a_vec, q_vec]:
        distances, indices = doc_index.index.search(query_vec, min(TOP_K, len(doc_index.chunks)))
        for dist, i in zip(distances[0], indices[0]):
            if float(dist) > max_score:
                max_score = float(dist)
            if i >= 0 and doc_index.chunks[i] not in seen:
                seen.add(doc_index.chunks[i])
                candidate_chunks.append(doc_index.chunks[i])

    if not candidate_chunks:
        return "HALLUCINATION", 0.0, 0.0, 1.0, "[no chunks]", max_score

    # Similarity threshold check (mirrors app.py)
    if max_score < SIMILARITY_THRESHOLD:
        return (
            "HALLUCINATION", 0.0, 0.0, 1.0,
            f"[UNSUPPORTED] max_sim={max_score:.3f} < threshold={SIMILARITY_THRESHOLD}",
            max_score,
        )

    best_nli  = None
    best_chunk = candidate_chunks[0]
    for chunk in candidate_chunks:
        nli = run_nli(premise=chunk, hypothesis=answer)
        if best_nli is None or nli["entailment"] > best_nli["entailment"]:
            best_nli  = nli
            best_chunk = chunk

    return (
        best_nli["verdict"],
        best_nli["entailment"],
        best_nli["neutral"],
        best_nli["contradiction"],
        best_chunk[:500],
        max_score,
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def verdict_to_binary(verdict: str) -> int:
    """
    Map NLI verdict to binary prediction.
    GROUNDED / UNCERTAIN  -> 0 (predicted correct)
    HALLUCINATION         -> 1 (predicted hallucinated)
    """
    return 1 if verdict == "HALLUCINATION" else 0


def expected_label_for(label: int) -> str:
    """
    label 0 (correct answer)   -> ground truth verdict = GROUNDED
    label 1 (incorrect answer) -> ground truth verdict = HALLUCINATION
    """
    return "GROUNDED" if label == 0 else "HALLUCINATION"


def compute_metrics(y_true, y_pred):
    """Binary classification metrics without sklearn dependency."""
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc       = (TP + TN) / len(y_true) if y_true else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=acc, precision=precision, recall=recall,
                f1=f1, specificity=specificity)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(dataset_path=DEFAULT_DATASET,
             pdfs_dir=DEFAULT_PDFS_DIR,
             output_xlsx=DEFAULT_OUT_XLSX,
             output_txt=DEFAULT_OUT_TXT):

    # ── Load dataset ──────────────────────────────────────────────────────
    df = pd.read_excel(dataset_path)
    logger.info(f"Loaded {len(df)} rows from {dataset_path}")

    # Columns we need: question_id, doc_id, question, answer, label
    required = {"question_id", "doc_id", "question", "answer", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    label_counts = df["label"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_counts}")

    # ── Load models ───────────────────────────────────────────────────────
    embedder = get_embedder()
    get_nli()   # warm up NLI model

    # ── Build / load FAISS indexes per doc ───────────────────────────────
    doc_indexes = {}
    for doc_id in df["doc_id"].unique():
        pdf_path = os.path.join(pdfs_dir, doc_id)
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found, will skip rows for: {pdf_path}")
            continue
        idx = get_doc_index(pdf_path, doc_id, embedder)
        if idx is not None:
            doc_indexes[doc_id] = idx
            logger.info(f"  {doc_id}: {len(idx.chunks)} chunks")

    # ── Per-row evaluation ────────────────────────────────────────────────
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        qid      = row["question_id"]
        doc_id   = row["doc_id"]
        question = str(row["question"])
        answer   = str(row["answer"])
        label    = int(row["label"])           # 0 = correct, 1 = hallucinated

        if doc_id not in doc_indexes:
            logger.warning(f"[{qid}] No index for {doc_id}, skipping")
            continue

        verdict, ent, neu, con, ctx, max_sim = blackbox_predict(
            doc_indexes[doc_id], question, answer
        )

        pred_label = verdict_to_binary(verdict)
        is_correct = (pred_label == label)

        # For label-0 rows: correct answer IS in PDF → expected GROUNDED
        # For label-1 rows: hallucinated answer NOT in PDF → expected HALLUCINATION
        expected_verdict = expected_label_for(label)

        records.append({
            "question_id":       qid,
            "doc_id":            doc_id,
            "label":             label,
            "label_name":        "correct" if label == 0 else "hallucinated",
            "question":          question,
            "answer":            answer,
            "expected_verdict":  expected_verdict,
            "nli_verdict":       verdict,
            "predicted_label":   pred_label,
            "is_correct":        is_correct,
            "entailment":        ent,
            "neutral":           neu,
            "contradiction":     con,
            "max_similarity":    round(max_sim, 4),
            "retrieved_context": ctx,
        })

        gc.collect()

    out_df = pd.DataFrame(records)
    if out_df.empty:
        logger.error("No results produced. Check PDFs and dataset paths.")
        return

    # ── Save full results ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    out_df.to_excel(output_xlsx, index=False)
    logger.info(f"Saved full results -> {output_xlsx}")

    # ── Compute metrics ───────────────────────────────────────────────────
    y_true = out_df["label"].tolist()
    y_pred = out_df["predicted_label"].tolist()
    m      = compute_metrics(y_true, y_pred)

    # Per-label accuracy
    label0_df = out_df[out_df["label"] == 0]
    label1_df = out_df[out_df["label"] == 1]
    acc0 = label0_df["is_correct"].mean() if len(label0_df) else 0
    acc1 = label1_df["is_correct"].mean() if len(label1_df) else 0

    # Verdict distribution
    verdict_counts = out_df["nli_verdict"].value_counts()

    # Average NLI scores by label
    avg_ent_label0 = label0_df["entailment"].mean() if len(label0_df) else 0
    avg_ent_label1 = label1_df["entailment"].mean() if len(label1_df) else 0
    avg_sim_label0 = label0_df["max_similarity"].mean() if len(label0_df) else 0
    avg_sim_label1 = label1_df["max_similarity"].mean() if len(label1_df) else 0

    # ── Build summary report ──────────────────────────────────────────────
    sep  = "=" * 62
    sep2 = "-" * 62
    lines = [
        sep,
        "  BLACKBOX (NLI) EVALUATION SUMMARY",
        sep,
        f"  Dataset          : {dataset_path}",
        f"  Total rows       : {len(out_df)}",
        f"  Label 0 (correct): {len(label0_df)}",
        f"  Label 1 (halluc.): {len(label1_df)}",
        sep2,
        "  OVERALL METRICS",
        sep2,
        f"  Accuracy         : {m['accuracy']*100:.2f}%",
        f"  Precision        : {m['precision']*100:.2f}%   (of predicted hallucinations, how many are truly hallucinated)",
        f"  Recall           : {m['recall']*100:.2f}%   (of true hallucinations, how many were caught)",
        f"  F1 Score         : {m['f1']*100:.2f}%",
        f"  Specificity      : {m['specificity']*100:.2f}%  (of true correct answers, how many were spared)",
        sep2,
        "  CONFUSION MATRIX (positive = label 1 = HALLUCINATED)",
        sep2,
        f"  True  Pos (TP)   : {m['TP']:>4}  Hallucinated, correctly flagged",
        f"  True  Neg (TN)   : {m['TN']:>4}  Correct answers, correctly passed",
        f"  False Pos (FP)   : {m['FP']:>4}  Correct answers, wrongly flagged as hallucination",
        f"  False Neg (FN)   : {m['FN']:>4}  Hallucinations, missed (labeled GROUNDED/UNCERTAIN)",
        sep2,
        "  PER-CLASS ACCURACY",
        sep2,
        f"  Label 0 (correct answers in PDF)  : {acc0*100:.2f}%",
        f"  Label 1 (hallucinated answers)    : {acc1*100:.2f}%",
        sep2,
        "  VERDICT DISTRIBUTION",
        sep2,
    ]
    for verdict, count in verdict_counts.items():
        pct = count / len(out_df) * 100
        lines.append(f"  {verdict:<18}: {count:>4}  ({pct:.1f}%)")

    lines += [
        sep2,
        "  AVERAGE NLI SCORES BY TRUE LABEL",
        sep2,
        f"  Label 0 (correct) avg entailment : {avg_ent_label0:.4f}",
        f"  Label 1 (halluc.) avg entailment : {avg_ent_label1:.4f}",
        f"  Label 0 avg FAISS similarity     : {avg_sim_label0:.4f}",
        f"  Label 1 avg FAISS similarity     : {avg_sim_label1:.4f}",
        sep2,
        "  OVERFITTING / GENERALIZATION ANALYSIS",
        sep2,
        "  The blackbox pipeline has NO trainable parameters:",
        "  BAAI/bge-small-en-v1.5 and DeBERTa NLI are both frozen",
        "  pretrained models. There is no fitting to this dataset,",
        "  so traditional overfitting does not apply.",
        "",
        "  GENERALIZATION RISKS:",
        "",
        "  1. SIMILARITY THRESHOLD (0.35): A fixed threshold that was",
        "     not tuned on a held-out set may be sub-optimal. If the",
        "     threshold is too high, correct answers with lower semantic",
        "     overlap (e.g. paraphrased facts) are wrongly flagged.",
        "",
        "  2. CHUNK SIZE (100 words): Answers requiring multi-page context",
        "     may never be fully captured in a single chunk. The sliding",
        "     window mitigates but does not eliminate this.",
        "",
        "  3. NLI HYPOTHESIS PHRASING: Only the answer text is used as",
        "     the NLI hypothesis. Including the question can increase",
        "     false entailments (a sentence that mentions the topic but",
        "     gives a different spec value scores high entailment).",
        "",
        "  4. DOMAIN SHIFT: DeBERTa was not fine-tuned on product manuals.",
        "     Technical jargon, units, and model numbers may reduce NLI",
        "     reliability compared to general-domain benchmarks.",
        "",
        "  5. FALSE NEGATIVES ON PARAPHRASED HALLUCINATIONS: Hallucinated",
        "     answers that sound plausible and use vocabulary from the PDF",
        "     can slip through with UNCERTAIN verdict.",
        sep2,
        "  IMPROVEMENTS FOR ROBUSTNESS",
        sep2,
        "  - Tune the similarity threshold on a held-out validation split",
        "    (e.g. ROC analysis on FAISS scores for label 0 vs 1).",
        "  - Fine-tune DeBERTa NLI on product-manual (premise, answer) pairs",
        "    using the existing labeled dataset.",
        "  - Add a question-aware NLI hypothesis:",
        "    'Given question: {q}, the answer is: {a}' as the hypothesis.",
        "  - Increase TOP_K retrieval and add re-ranking with a cross-encoder",
        "    before NLI to improve chunk selection quality.",
        "  - Augment with keyword matching (BM25) alongside dense retrieval",
        "    to better handle numerical specs and model numbers.",
        sep,
    ]

    summary_text = "\n".join(lines)
    print(summary_text)

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    logger.info(f"Saved summary -> {output_txt}")

    # ── Per-doc breakdown ─────────────────────────────────────────────────
    print("\n" + sep2)
    print("  PER-DOCUMENT ACCURACY")
    print(sep2)
    for doc_id, grp in out_df.groupby("doc_id"):
        doc_acc  = grp["is_correct"].mean() * 100
        n        = len(grp)
        n0, n1   = (grp["label"] == 0).sum(), (grp["label"] == 1).sum()
        print(f"  {doc_id:<30} acc={doc_acc:.1f}%  (n={n}, label0={n0}, label1={n1})")

    return out_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackbox NLI evaluation over labeled dataset")
    parser.add_argument("--dataset",      default=DEFAULT_DATASET,  help="Path to features_correct_incorrect.xlsx")
    parser.add_argument("--pdfs_dir",     default=DEFAULT_PDFS_DIR, help="Path to resources/pdfs/")
    parser.add_argument("--output_xlsx",  default=DEFAULT_OUT_XLSX, help="Output Excel file path")
    parser.add_argument("--output_txt",   default=DEFAULT_OUT_TXT,  help="Output summary text file path")
    args = parser.parse_args()

    evaluate(
        dataset_path = args.dataset,
        pdfs_dir     = args.pdfs_dir,
        output_xlsx  = args.output_xlsx,
        output_txt   = args.output_txt,
    )