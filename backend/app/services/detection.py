"""
Detection Service
Contains whitebox (HRES) and blackbox (NLI) prediction pipelines

This module mirrors the exact detection logic from src/app.py to ensure
consistent behavior between the Gradio frontend and React/FastAPI backend.
"""
import gc
import re
import numpy as np
import torch
import faiss
from ..services.model_loader import get_llama, get_embedder, get_nli, get_classifiers
from ..utils.pdf_utils import extract_all_text, chunk_text, clean_text

# Constants (matching app.py exactly)
HIDDEN_DIM = 2304
MAX_CTX_CHARS = 1600
TOP_K = 5
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
VERDICT_MAP = {
    "entailment": "GROUNDED",
    "neutral": "UNCERTAIN",
    "contradiction": "HALLUCINATION"
}

# PDF cache for faster sequential lookups (matching app.py)
_pdf_cache = {}


def get_pdf_index(pdf_path):
    """Caches extracted PDF text and its FAISS index to drastically speed up sequential lookups."""
    global _pdf_cache
    if pdf_path in _pdf_cache:
        return _pdf_cache[pdf_path], None

    embedder = get_embedder()
    pages = extract_all_text(pdf_path)
    if not pages:
        return None, "Could not extract text from PDF"

    all_chunks = []
    for page in pages:
        for chunk in chunk_text(clean_text(page["text"])):
            all_chunks.append(chunk)

    if not all_chunks:
        return None, "No text chunks extracted from PDF"

    vecs = embedder.encode(all_chunks, batch_size=128, show_progress_bar=False,
                           normalize_embeddings=True, device="cuda").astype(np.float32)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    _pdf_cache[pdf_path] = {"chunks": all_chunks, "index": index}
    return _pdf_cache[pdf_path], None


# ── Whitebox Pipeline (matching app.py exactly) ──────────────────────────────

def find_last_meaningful_token(input_ids, tokenizer):
    """Find the last token that contains alphanumeric characters (matching app.py)"""
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for i in range(len(input_ids) - 1, -1, -1):
        tok_id = input_ids[i].item()
        tok_str = tokenizer.decode(tok_id).strip()

        if tok_id in special_ids:
            continue
        if not any(c.isalnum() for c in tok_str):
            continue

        return i

    return 0


def extract_hidden_state(context_text, question, answer):
    """Extract hidden states from Gemma for the given context, question, and answer.
    
    Uses the chat template format to match app.py exactly.
    """
    tokenizer, model = get_llama()
    
    if str(context_text) == 'nan' or not context_text:
        user_msg = f"Question: {question}"
    else:
        user_msg = f"Context: {context_text}\n\nQuestion: {question}"

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1900).to(model.device)
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
    Run whitebox pipeline: extract hidden states -> classify.
    
    Returns: (results_dict, error_string)
    
    The results_dict contains a single classifier result with:
    - model: classifier name
    - prediction: 0 or 1
    - label: "CORRECT" or "HALLUCINATED"
    - confidence: max probability
    - prob_correct: probability of correct
    - prob_hallucinated: probability of hallucinated
    """
    cache_data, err = get_pdf_index(pdf_path)
    if err:
        return None, err
    index, all_chunks = cache_data["index"], cache_data["chunks"]

    # Retrieve the single best chunk using the answer as query
    # (mirrors training: context = the page containing the answer)
    embedder = get_embedder()
    a_vec = embedder.encode(
        [answer], normalize_embeddings=True, device="cuda",
    ).astype(np.float32)
    _, indices = index.search(a_vec, 1)
    best_idx = int(indices[0][0])
    context = all_chunks[best_idx][:MAX_CTX_CHARS]

    vector, seq_len, target_index = extract_hidden_state(context, question, answer)

    features = np.concatenate([vector, [seq_len, target_index]]).reshape(1, -1)

    classifier, classifier_name, scaler, reduction, vt = get_classifiers()

    features_vt = vt.transform(features)
    features_scaled = scaler.transform(features_vt)
    features_reduced = reduction.transform(features_scaled)

    # Make prediction with the loaded classifier
    pred = classifier.predict(features_reduced)[0]
    proba = classifier.predict_proba(features_reduced)[0]
    
    result = {
        "model": classifier_name,
        "prediction": int(pred),
        "label": "HALLUCINATED" if pred == 1 else "CORRECT",
        "confidence": float(max(proba)),
        "prob_correct": float(proba[0]),
        "prob_hallucinated": float(proba[1]),
    }

    return result, None


# ── Blackbox Pipeline (matching app.py exactly) ──────────────────────────────

def nli_batch(premises, hypothesis):
    """Run NLI model on a batch of premises against a single hypothesis"""
    tok, model = get_nli()
    encodings = tok(premises, [hypothesis] * len(premises),
                    return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(**encodings).logits, dim=-1).cpu()
    del encodings
    torch.cuda.empty_cache()
    return probs


def run_nli_on_chunk(premise, hypothesis):
    """Run NLI on a single premise-hypothesis pair (matching app.py)
    
    Enhanced to also check individual sentences within the chunk when
    entailment is low, which helps when the answer is buried in a noisy chunk.
    """
    import re
    
    # First, try the full chunk as premise
    probs = nli_batch([premise], hypothesis)[0].tolist()
    best_probs = probs
    
    # If entailment is low, also try individual sentences within the chunk
    # This helps when the relevant statement is buried in a noisy chunk
    if probs[1] < 0.5:  # entailment < 50%
        sentences = re.split(r'(?<=[.!?])\s+', premise.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if sentences:
            # Batch process all sentences at once for efficiency
            all_probs = nli_batch(sentences, hypothesis)
            for sent_probs in all_probs:
                sent_probs = sent_probs.tolist()
                if sent_probs[1] > best_probs[1]:  # higher entailment
                    best_probs = sent_probs
    
    label = NLI_LABEL_MAP[best_probs.index(max(best_probs))]
    
    return {
        "label": label,
        "verdict": VERDICT_MAP[label],
        "entailment": round(best_probs[1], 4),
        "neutral": round(best_probs[2], 4),
        "contradiction": round(best_probs[0], 4),
    }


def blackbox_predict(pdf_path, question, answer):
    """
    Run blackbox pipeline: FAISS retrieval + NLI.
    
    Returns: (result_dict, error_string)
    """
    cache_data, err = get_pdf_index(pdf_path)
    if err:
        return None, err
    index, all_chunks = cache_data["index"], cache_data["chunks"]

    embedder = get_embedder()
    q_vec = embedder.encode([question], normalize_embeddings=True,
                            device="cuda").astype(np.float32)
    a_vec = embedder.encode([answer], normalize_embeddings=True,
                            device="cuda").astype(np.float32)

    seen = set()
    candidate_chunks = []
    max_score = -1.0

    # Reduce TOP_K search width to speed up (checking top 3 is usually enough)
    for query_vec in [a_vec, q_vec]:
        distances, indices = index.search(query_vec, min(3, len(all_chunks)))
        for dist, i in zip(distances[0], indices[0]):
            if dist > max_score:
                max_score = float(dist)
            if i >= 0 and all_chunks[i] not in seen:
                seen.add(all_chunks[i])
                candidate_chunks.append(all_chunks[i])

    if not candidate_chunks:
        return None, "No relevant chunks found"

    # METHOD 1: FAISS Similarity Score Thresholding
    # If the semantic similarity is too low, the document doesn't contain the answer.
    SIMILARITY_THRESHOLD = 0.35
    if max_score < SIMILARITY_THRESHOLD:
        result = {
            "verdict": "HALLUCINATION",
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 1.0,
            "retrieved_context": f"[UNSUPPORTED] Context irrelevant. Highest semantic similarity ({max_score:.2f}) is below threshold ({SIMILARITY_THRESHOLD}).",
        }
        gc.collect()
        return result, None

    best_nli = None
    best_chunk = candidate_chunks[0]
    
    # Use answer as hypothesis directly (full Q&A format causes false positives)
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
