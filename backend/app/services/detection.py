"""
Detection Service
Contains whitebox (HRES) and blackbox (NLI) prediction pipelines
"""
import gc
import re
import numpy as np
import torch
import faiss
from ..services.model_loader import get_llama, get_embedder, get_nli, get_classifiers
from ..utils.pdf_utils import extract_all_text, chunk_text, clean_text

# Constants
HIDDEN_DIM = 2048
MAX_CTX_CHARS = 1600
TOP_K = 5
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
VERDICT_MAP = {
    "entailment": "GROUNDED",
    "neutral": "UNCERTAIN",
    "contradiction": "HALLUCINATION"
}


# ── Whitebox Pipeline ────────────────────────────────────────────────────────

def find_last_meaningful_token(input_ids, tokenizer):
    """Find the last token that contains alphanumeric characters"""
    target_index = input_ids.shape[0] - 1
    while target_index > 0:
        tok_str = tokenizer.decode(input_ids[target_index]).strip()
        if any(c.isalnum() for c in tok_str) and "</s>" not in tok_str:
            break
        target_index -= 1
    return target_index


def extract_hidden_state(context_text, question, answer):
    """Extract hidden states from TinyLlama for the given context, question, and answer"""
    tokenizer, model = get_llama()
    text = f"{context_text}\n\nQuestion: {question}\nAnswer: {answer}"
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1900
    ).to(model.device)
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]
    target_index = find_last_meaningful_token(input_ids, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    vector = outputs.hidden_states[-1][0, target_index, :].float().cpu().numpy()
    del outputs, inputs
    torch.cuda.empty_cache()
    gc.collect()
    return vector, seq_len, target_index


def whitebox_predict(pdf_path, question, answer):
    """
    Run whitebox pipeline: extract hidden states -> classify
    Returns: (results_dict, error_string)
    """
    # Build FAISS index over the full document
    embedder = get_embedder()
    pages = extract_all_text(pdf_path)
    if not pages:
        return None, "Could not extract text from PDF"

    all_chunks, all_page_nums = [], []
    for page in pages:
        for chunk in chunk_text(clean_text(page["text"])):
            all_chunks.append(chunk)
            all_page_nums.append(page["page_num"])

    if not all_chunks:
        return None, "No text chunks extracted from PDF"

    vecs = embedder.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        device="cuda",
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Retrieve the single best chunk using the answer as query
    # (mirrors training: context = the page containing the answer)
    a_vec = embedder.encode(
        [answer], normalize_embeddings=True, device="cuda"
    ).astype(np.float32)
    _, indices = index.search(a_vec, 1)
    best_idx = int(indices[0][0])
    context = all_chunks[best_idx][:MAX_CTX_CHARS]

    vector, seq_len, target_index = extract_hidden_state(context, question, answer)

    features = np.concatenate([vector, [seq_len, target_index]]).reshape(1, -1)

    svm, xgb, scaler, reduction, vt = get_classifiers()

    features_vt = vt.transform(features)
    features_scaled = scaler.transform(features_vt)
    features_reduced = reduction.transform(features_scaled)

    results = {}
    if svm is not None:
        svm_pred = svm.predict(features_reduced)[0]
        svm_proba = svm.predict_proba(features_reduced)[0]
        results["SVM"] = {
            "prediction": int(svm_pred),
            "label": "HALLUCINATED" if svm_pred == 1 else "CORRECT",
            "confidence": float(max(svm_proba)),
            "prob_correct": float(svm_proba[0]),
            "prob_hallucinated": float(svm_proba[1]),
        }
    if xgb is not None:
        xgb_pred = xgb.predict(features_reduced)[0]
        xgb_proba = xgb.predict_proba(features_reduced)[0]
        results["XGBoost"] = {
            "prediction": int(xgb_pred),
            "label": "HALLUCINATED" if xgb_pred == 1 else "CORRECT",
            "confidence": float(max(xgb_proba)),
            "prob_correct": float(xgb_proba[0]),
            "prob_hallucinated": float(xgb_proba[1]),
        }

    return results, None


# ── Blackbox Pipeline ────────────────────────────────────────────────────────

def nli_batch(premises, hypothesis):
    """Run NLI model on a batch of premises against a single hypothesis"""
    tok, model = get_nli()
    encodings = tok(
        premises,
        [hypothesis] * len(premises),
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(**encodings).logits, dim=-1).cpu()
    del encodings
    torch.cuda.empty_cache()
    return probs


def run_nli_on_chunk(premise, hypothesis):
    """
    Run NLI scoring on a chunk of text (premise) against the hypothesis (answer)
    Returns dict with label, verdict, and probability scores
    """
    sentences = re.split(r'(?<=[.!?])\s+', premise.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        sentences = [premise]

    # Create candidate spans: individual sentences, 2-sentence windows, 3-sentence windows, full premise
    candidates = list(sentences)
    for w in (2, 3):
        for i in range(len(sentences) - w + 1):
            candidates.append(" ".join(sentences[i : i + w]))
    candidates.append(premise)

    # Split answer into steps if it has multiple lines
    answer_steps = [s.strip() for s in re.split(r'\\n|\n', hypothesis) if s.strip()]
    if len(answer_steps) <= 1:
        answer_steps = [hypothesis]

    if len(answer_steps) == 1:
        # Single-step answer: find best matching candidate
        all_probs = nli_batch(candidates, hypothesis)
        best_idx = int(all_probs[:, 1].argmax())
        probs = all_probs[best_idx].tolist()
    else:
        # Multi-step answer: aggregate across steps
        min_ent = None
        sum_probs = None
        count = 0
        for step in answer_steps:
            step_probs = nli_batch(candidates, step)
            if min_ent is None:
                min_ent = step_probs[:, 1].clone()
                sum_probs = step_probs.clone()
            else:
                min_ent = torch.min(min_ent, step_probs[:, 1])
                sum_probs += step_probs
            count += 1
        best_idx = int(min_ent.argmax())
        probs = (sum_probs[best_idx] / count).tolist()

    label = NLI_LABEL_MAP[probs.index(max(probs))]
    return {
        "label": label,
        "verdict": VERDICT_MAP[label],
        "entailment": round(probs[1], 4),
        "neutral": round(probs[2], 4),
        "contradiction": round(probs[0], 4),
    }


def blackbox_predict(pdf_path, question, answer):
    """
    Run blackbox pipeline: FAISS retrieval + NLI
    Returns: (result_dict, error_string)
    """
    embedder = get_embedder()

    pages = extract_all_text(pdf_path)
    if not pages:
        return None, "Could not extract text from PDF"

    all_chunks, all_page_nums = [], []
    for page in pages:
        for chunk in chunk_text(clean_text(page["text"])):
            all_chunks.append(chunk)
            all_page_nums.append(page["page_num"])

    if not all_chunks:
        return None, "No text chunks extracted from PDF"

    vecs = embedder.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        device="cuda",
    ).astype(np.float32)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    q_vec = embedder.encode(
        [question], normalize_embeddings=True, device="cuda"
    ).astype(np.float32)
    a_vec = embedder.encode(
        [answer], normalize_embeddings=True, device="cuda"
    ).astype(np.float32)

    # Retrieve top-K chunks using both question and answer
    seen = set()
    candidate_chunks = []

    for query_vec in [a_vec, q_vec]:
        _, indices = index.search(query_vec, TOP_K)
        for i in indices[0]:
            if i >= 0 and all_chunks[i] not in seen:
                seen.add(all_chunks[i])
                candidate_chunks.append(all_chunks[i])

    if not candidate_chunks:
        return None, "No relevant chunks found"

    # Run NLI on each candidate chunk and select the best one
    best_nli = None
    best_chunk = candidate_chunks[0]
    for chunk in candidate_chunks:
        nli = run_nli_on_chunk(premise=chunk, hypothesis=answer)
        if best_nli is None or nli["entailment"] > best_nli["entailment"]:
            best_nli = nli
            best_chunk = chunk

    result = {
        "verdict": best_nli["verdict"],
        "entailment": best_nli["entailment"],
        "neutral": best_nli["neutral"],
        "contradiction": best_nli["contradiction"],
        "retrieved_context": best_chunk[:500],
    }
    gc.collect()
    return result, None
