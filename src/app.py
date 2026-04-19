"""
app.py
======
Gradio frontend for HRES Hallucination Detection.

Two detection pipelines:
  - Whitebox (HRES): Gemma hidden-state extraction -> PCA -> SVM/XGBoost
  - Blackbox (NLI):  FAISS retrieval from PDF -> DeBERTa NLI scoring
        
User provides: PDF file + Question + Answer
App returns: combined verdict from both pipelines.
"""

import os
import gc
import re
import tempfile

import numpy as np
import pandas as pd
import torch
import fitz
import faiss
import joblib
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "gemma-2-2b-it")
PDFS_DIR       = os.path.join(BASE_DIR, "resources", "pdfs")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler_final.pkl")
REDUCTION_PATH = os.path.join(MODELS_DIR, "reduction_final.pkl")
REDUCTION_META_PATH = os.path.join(MODELS_DIR, "reduction_metadata.csv")
VT_PATH        = os.path.join(MODELS_DIR, "variance_threshold_final.pkl")

HIDDEN_DIM     = 2304
MAX_CTX_CHARS  = 1600
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from nli_utils import (
    get_embedder, get_nli, build_or_load_index,
    blackbox_predict_unified, SIMILARITY_THRESHOLD,
    NLI_LABEL_MAP, VERDICT_MAP,
)

# ── Lazy-loaded globals ──────────────────────────────────────────────────────
_llama_tok = None
_llama_model = None
_embedder = None
_nli_tok = None
_nli_model = None
_classifier = None
_classifier_name = None
_scaler = None
_reduction = None
_reduction_method = None
_vt = None
_pdf_cache = {}


# ── Loaders (called once on first use) ───────────────────────────────────────

def get_llama():
    global _llama_tok, _llama_model
    if _llama_model is None:
        _llama_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")
        _llama_model.eval()
    return _llama_tok, _llama_model






def get_classifiers():
    """
    Dynamically load the best trained classifier from metadata.
    Falls back to legacy SVM/XGB/Ada/LR models if best_model_final.pkl not found.
    """
    global _classifier, _classifier_name, _scaler, _reduction, _reduction_method, _vt
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
        _reduction = joblib.load(REDUCTION_PATH)
        _vt = joblib.load(VT_PATH)

        # Load reduction metadata to know which method was used
        if os.path.exists(REDUCTION_META_PATH):
            meta_df = pd.read_csv(REDUCTION_META_PATH)
            _reduction_method = meta_df.get('reduction_method', meta_df.get('method', ['PCA'])).iloc[0]
            
            # Try to load the best model from metadata
            if 'model_filename' in meta_df.columns:
                model_file = meta_df['model_filename'].iloc[0]
                model_path = os.path.join(MODELS_DIR, model_file)
                if os.path.exists(model_path):
                    _classifier = joblib.load(model_path)
                    _classifier_name = meta_df.get('classifier_type', meta_df.get('classifier_config', ['Best Model'])).iloc[0]
                    print(f"✓ Loaded best model: {_classifier_name} from {model_file}")
                    return _classifier, _classifier_name, _scaler, _reduction, _vt
        else:
            _reduction_method = "PCA"
        
        # Fallback: try to load any available model (backward compatibility)
        model_candidates = [
            ("ensemble_stacking.pkl", "Stacking Ensemble"),
            ("svm_model_final.pkl", "SVM"),
            ("ada_model_final.pkl", "AdaBoost"),
            ("lr_model_final.pkl", "Logistic Regression"),
            ("xgb_model_final.pkl", "XGBoost"),
        ]
        
        for model_file, model_name in model_candidates:
            model_path = os.path.join(MODELS_DIR, model_file)
            if os.path.exists(model_path):
                _classifier = joblib.load(model_path)
                _classifier_name = model_name
                print(f"✓ Loaded classifier: {model_name} from {model_file}")
                break
        
        if _classifier is None:
            raise FileNotFoundError(
                "No classifier model found! Run training/train_unified.py first."
            )
    
    return _classifier, _classifier_name, _scaler, _reduction, _vt


# ── PDF helpers ──────────────────────────────────────────────────────────────

def extract_page_text(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    idx = page_num - 1
    if idx < 0 or idx >= len(doc):
        doc.close()
        return ""
    text = doc[idx].get_text().strip()
    doc.close()
    return text[:MAX_CTX_CHARS]


# ── Whitebox: hidden-state extraction ────────────────────────────────────────

def find_last_meaningful_token(input_ids, tokenizer):
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for i in range(len(input_ids) - 1, -1, -1):
        tok_id  = input_ids[i].item()
        tok_str = tokenizer.decode(tok_id).strip()

        if tok_id in special_ids:
            continue
        if not any(c.isalnum() for c in tok_str):
            continue

        return i

    return 0


def extract_hidden_state(context_text, question, answer):
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
    """Run whitebox pipeline: extract hidden states -> classify."""
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

    svm, xgb, scaler, reduction, vt = get_classifiers()

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


# ── Blackbox: NLI pipeline ───────────────────────────────────────────────────

def nli_batch(premises, hypothesis):
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
    """Blackbox prediction — now delegates to nli_utils.py."""
    embedder = get_embedder()

    # Build or load the FAISS index for this PDF
    doc_id = os.path.basename(pdf_path)
    doc_index = build_or_load_index(pdf_path, doc_id, embedder,
                                    index_dir=os.path.join(MODELS_DIR, "nli_index"))
    if doc_index is None:
        return None, "Could not extract text from PDF"

    result = blackbox_predict_unified(
        doc_index  = doc_index,
        question   = question,
        answer     = answer,
        similarity_threshold = SIMILARITY_THRESHOLD,
    )

    return result, None

def _html_error(msg):
    return f"""<div class="error-card"><span class="error-icon">!</span> {msg}</div>"""


def predict(pdf_file, question, answer, progress=gr.Progress()):
    if pdf_file is None:
        err = _html_error("Please upload a PDF file.")
        return err, "", ""
    if not question or not question.strip():
        err = _html_error("Please enter a question.")
        return err, "", ""
    if not answer or not answer.strip():
        err = _html_error("Please enter an answer to verify.")
        return err, "", ""

    pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file

    # ── Whitebox ──
    try:
        progress(0.1, desc="Running Whitebox (HRES) pipeline...")
        wb_results, wb_err = whitebox_predict(pdf_path, question, answer)
    except Exception as e:
        wb_results, wb_err = None, f"Whitebox pipeline failed: {e}"

    # ── Blackbox ──
    try:
        progress(0.5, desc="Running Blackbox (NLI) pipeline...")
        bb_results, bb_err = blackbox_predict(pdf_path, question, answer)
    except Exception as e:
        bb_results, bb_err = None, f"Blackbox pipeline failed: {e}"

    progress(0.9, desc="Generating results...")

    # ── Format whitebox output ──
    wb_text = ""
    if wb_err:
        wb_text = f"""<div class="error-card"><span class="error-icon">!</span> {wb_err}</div>"""
    elif wb_results:
        model_name = wb_results.get("model", "Classifier")
        res = wb_results
        is_hall = res["label"] == "HALLUCINATED"
        badge_cls = "badge-danger" if is_hall else "badge-success"
        icon = "&#10005;" if is_hall else "&#10003;"
        conf = res["confidence"] * 100
        p_correct = res["prob_correct"] * 100
        p_hall = res["prob_hallucinated"] * 100

        fill_cls = "fill-danger" if is_hall else "fill-success"
        wb_text = f"""
<div class="model-result-card">
  <div class="model-header">
    <span class="model-name">{model_name}</span>
    <span class="badge {badge_cls}">{icon} {res["label"]}</span>
  </div>
  <div class="confidence-section">
    <div class="confidence-label">Confidence <span class="confidence-value">{conf:.1f}%</span></div>
    <div class="progress-track">
      <div class="progress-fill {fill_cls}" style="width:{conf:.0f}%"></div>
    </div>
  </div>
  <div class="prob-grid">
    <div class="prob-item">
      <div class="prob-label">P(Correct)</div>
      <div class="prob-bar-track">
        <div class="prob-bar fill-success" style="width:{p_correct:.0f}%"></div>
      </div>
      <div class="prob-value">{p_correct:.1f}%</div>
    </div>
    <div class="prob-item">
      <div class="prob-label">P(Hallucinated)</div>
      <div class="prob-bar-track">
        <div class="prob-bar fill-danger" style="width:{p_hall:.0f}%"></div>
      </div>
      <div class="prob-value">{p_hall:.1f}%</div>
    </div>
  </div>
</div>"""
    else:
        wb_text = """<div class="empty-state">No classifiers available.</div>"""

    # ── Format blackbox output ──
    bb_text = ""
    if bb_err:
        bb_text = f"""<div class="error-card"><span class="error-icon">!</span> {bb_err}</div>"""
    elif bb_results:
        verdict = bb_results["verdict"]
        verdict_cls = {"GROUNDED": "badge-success", "UNCERTAIN": "badge-warning",
                       "HALLUCINATION": "badge-danger"}.get(verdict, "badge-muted")
        verdict_icon = {"GROUNDED": "&#10003;", "UNCERTAIN": "?",
                        "HALLUCINATION": "&#10005;"}.get(verdict, "")
        ent = bb_results["entailment"] * 100
        neu = bb_results["neutral"] * 100
        con = bb_results["contradiction"] * 100
        ctx = bb_results["retrieved_context"]

        bb_text = f"""
<div class="model-result-card">
  <div class="model-header">
    <span class="model-name">NLI Verdict</span>
    <span class="badge {verdict_cls}">{verdict_icon} {verdict}</span>
  </div>
  <div class="nli-bars">
    <div class="nli-row">
      <div class="nli-label">Entailment</div>
      <div class="nli-bar-track">
        <div class="nli-bar fill-success" style="width:{ent:.0f}%"></div>
      </div>
      <div class="nli-value">{ent:.1f}%</div>
    </div>
    <div class="nli-row">
      <div class="nli-label">Neutral</div>
      <div class="nli-bar-track">
        <div class="nli-bar fill-warning" style="width:{neu:.0f}%"></div>
      </div>
      <div class="nli-value">{neu:.1f}%</div>
    </div>
    <div class="nli-row">
      <div class="nli-label">Contradiction</div>
      <div class="nli-bar-track">
        <div class="nli-bar fill-danger" style="width:{con:.0f}%"></div>
      </div>
      <div class="nli-value">{con:.1f}%</div>
    </div>
  </div>
  <div class="source-chunk">
    <div class="source-chunk-header">Best Matching Source</div>
    <div class="source-chunk-text">{ctx}</div>
  </div>
</div>"""

    # ── Combined verdict ──
    combined = ""
    wb_label = None
    bb_verdict = None
    wb_conf = 0
    bb_ent = 0

    if wb_results:
        wb_label = wb_results.get("label")
        wb_conf = wb_results.get("confidence", 0)

    if bb_results:
        bb_verdict = bb_results["verdict"]
        bb_ent = bb_results["entailment"]

    verdict_text = ""
    verdict_desc = ""
    verdict_cls = "verdict-muted"

    if wb_label and bb_verdict:
        wb_says_hall = wb_label == "HALLUCINATED"
        bb_says_hall = bb_verdict == "HALLUCINATION"
        bb_says_ok = bb_verdict == "GROUNDED"

        if wb_says_hall and bb_says_hall:
            verdict_text = "HALLUCINATED"
            verdict_desc = "Both pipelines agree: this answer is likely hallucinated."
            verdict_cls = "verdict-danger"
        elif not wb_says_hall and bb_says_ok:
            verdict_text = "CORRECT"
            verdict_desc = "Both pipelines agree: this answer appears grounded in the source."
            verdict_cls = "verdict-success"
        elif wb_says_hall and not bb_says_hall:
            if bb_verdict == "UNCERTAIN":
                verdict_text = "HALLUCINATED"   # keep this — now UNCERTAIN is meaningful
                verdict_desc = "Whitebox flagged hallucination; NLI entailment below threshold."
                verdict_cls = "verdict-danger"
            else:
                verdict_text = "UNCERTAIN"
                verdict_desc = "Whitebox flagged hallucination, but NLI says grounded. Manual review recommended."
                verdict_cls = "verdict-warning"
        elif not wb_says_hall and bb_says_hall:
            verdict_text = "UNCERTAIN"
            verdict_desc = "NLI flagged hallucination, but whitebox says correct. Manual review recommended."
            verdict_cls = "verdict-warning"
        else:
            verdict_text = "UNCERTAIN"
            verdict_desc = "Mixed signals from both pipelines. Manual verification recommended."
            verdict_cls = "verdict-warning"
    elif wb_label:
        verdict_text = wb_label
        verdict_desc = "Result from whitebox pipeline only (NLI unavailable)."
        verdict_cls = "verdict-danger" if wb_label == "HALLUCINATED" else "verdict-success"
    elif bb_verdict:
        verdict_text = bb_verdict
        verdict_desc = "Result from blackbox pipeline only (whitebox unavailable)."
        verdict_cls = {"GROUNDED": "verdict-success", "UNCERTAIN": "verdict-warning",
                       "HALLUCINATION": "verdict-danger"}.get(bb_verdict, "verdict-muted")
    else:
        verdict_text = "ERROR"
        verdict_desc = "Both pipelines failed. Please check your inputs."

    verdict_icon = {"HALLUCINATED": "&#10005;", "CORRECT": "&#10003;",
                    "UNCERTAIN": "&#9888;", "ERROR": "&#9888;",
                    "GROUNDED": "&#10003;", "HALLUCINATION": "&#10005;"
                    }.get(verdict_text, "")

    combined = f"""
<div class="verdict-card {verdict_cls}">
  <div class="verdict-icon">{verdict_icon}</div>
  <div class="verdict-label">{verdict_text}</div>
  <div class="verdict-desc">{verdict_desc}</div>
</div>"""

    if wb_label and bb_verdict:
        wb_icon = "&#10005;" if wb_label == "HALLUCINATED" else "&#10003;"
        bb_icon = "&#10005;" if bb_verdict == "HALLUCINATION" else ("&#10003;" if bb_verdict == "GROUNDED" else "?")
        combined += f"""
<div class="summary-table">
  <div class="summary-row summary-header">
    <div class="summary-cell">Pipeline</div>
    <div class="summary-cell">Method</div>
    <div class="summary-cell">Result</div>
  </div>
  <div class="summary-row">
    <div class="summary-cell"><span class="pipeline-tag tag-wb">Whitebox</span></div>
    <div class="summary-cell cell-method">Gemma hidden states + SVM</div>
    <div class="summary-cell"><strong>{wb_icon} {wb_label}</strong> ({wb_conf*100:.0f}%)</div>
  </div>
  <div class="summary-row">
    <div class="summary-cell"><span class="pipeline-tag tag-bb">Blackbox</span></div>
    <div class="summary-cell cell-method">FAISS + DeBERTa NLI</div>
    <div class="summary-cell"><strong>{bb_icon} {bb_verdict}</strong> (ent: {bb_ent*100:.0f}%)</div>
  </div>
</div>"""

    return combined, wb_text, bb_text


# ── Preloaded PDF selector ──────────────────────────────────────────────────

def get_preloaded_pdfs():
    if os.path.isdir(PDFS_DIR):
        return [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]
    return []


def predict_with_options(preloaded_pdf, question, answer,
                         progress=gr.Progress()):
    """Run prediction using a preloaded PDF."""
    try:
        if not preloaded_pdf or preloaded_pdf == "None":
            err = _html_error("Please select a document from the dropdown.")
            return err, "", ""

        pdf_path = os.path.join(PDFS_DIR, preloaded_pdf)

        class _FakePdf:
            def __init__(self, p): self.name = p
        return predict(_FakePdf(pdf_path), question, answer, progress)
    except Exception as e:
        err = _html_error(f"Unexpected error: {e}")
        return err, err, err


# ── Gradio UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Reset & Base ─────────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ── Hero Header ──────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 32px 20px 18px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.4em;
    font-weight: 800;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-size: 1em;
    color: #64748b;
    margin-top: 6px;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-wb {
    background: rgba(102,126,234,0.12);
    color: #667eea;
    border: 1px solid rgba(102,126,234,0.25);
}
.badge-bb {
    background: rgba(16,185,129,0.10);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.25);
}

/* ── Section Headers ──────────────────────────────────────── */
.section-header {
    font-size: 0.82em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94a3b8;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(148,163,184,0.2);
}

/* ── Input Panel ──────────────────────────────────────────── */
.input-panel {
    background: rgba(248,250,252,0.6);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: 14px;
    padding: 20px;
}

/* ── Verdict Card ─────────────────────────────────────────── */
.verdict-card {
    text-align: center;
    padding: 28px 20px;
    border-radius: 14px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    opacity: 0.06;
    pointer-events: none;
}
.verdict-danger {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fca5a5;
}
.verdict-success {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
}
.verdict-warning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fcd34d;
}
.verdict-muted {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #cbd5e1;
}
.verdict-icon {
    font-size: 36px;
    margin-bottom: 4px;
}
.verdict-danger .verdict-icon { color: #dc2626; }
.verdict-success .verdict-icon { color: #16a34a; }
.verdict-warning .verdict-icon { color: #d97706; }
.verdict-muted .verdict-icon { color: #64748b; }
.verdict-label {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.verdict-danger .verdict-label { color: #dc2626; }
.verdict-success .verdict-label { color: #16a34a; }
.verdict-warning .verdict-label { color: #d97706; }
.verdict-muted .verdict-label { color: #64748b; }
.verdict-desc {
    font-size: 0.88em;
    color: #64748b;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Summary Table ────────────────────────────────────────── */
.summary-table {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 10px;
    font-size: 0.88em;
}
.summary-row {
    display: grid;
    grid-template-columns: 120px 1fr 180px;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid #f1f5f9;
}
.summary-row:last-child { border-bottom: none; }
.summary-header {
    background: #f8fafc;
    font-weight: 700;
    color: #64748b;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.summary-cell { }
.cell-method {
    color: #64748b;
    font-size: 0.92em;
}
.pipeline-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.82em;
    font-weight: 600;
}
.tag-wb {
    background: rgba(102,126,234,0.12);
    color: #667eea;
}
.tag-bb {
    background: rgba(16,185,129,0.10);
    color: #059669;
}

/* ── Model Result Card ────────────────────────────────────── */
.model-result-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    background: #fff;
}
.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}
.model-name {
    font-weight: 700;
    font-size: 1em;
    color: #1e293b;
}

/* ── Badges ───────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82em;
    letter-spacing: 0.5px;
}
.badge-success {
    background: #dcfce7;
    color: #16a34a;
    border: 1px solid #86efac;
}
.badge-danger {
    background: #fee2e2;
    color: #dc2626;
    border: 1px solid #fca5a5;
}
.badge-warning {
    background: #fef3c7;
    color: #d97706;
    border: 1px solid #fcd34d;
}
.badge-muted {
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #cbd5e1;
}

/* ── Confidence / Progress ────────────────────────────────── */
.confidence-section {
    margin-bottom: 14px;
}
.confidence-label {
    font-size: 0.82em;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}
.confidence-value {
    font-weight: 700;
    color: #1e293b;
}
.progress-track {
    height: 8px;
    background: #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.fill-success { background: linear-gradient(90deg, #22c55e, #16a34a); }
.fill-danger { background: linear-gradient(90deg, #f87171, #dc2626); }
.fill-warning { background: linear-gradient(90deg, #fbbf24, #d97706); }

/* ── Probability Bars ─────────────────────────────────────── */
.prob-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.prob-item {
    display: grid;
    grid-template-columns: 120px 1fr 52px;
    align-items: center;
    gap: 10px;
}
.prob-label {
    font-size: 0.8em;
    color: #64748b;
    font-weight: 500;
}
.prob-bar-track {
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.prob-value {
    font-size: 0.8em;
    font-weight: 700;
    color: #1e293b;
    text-align: right;
}

/* ── NLI Bars ─────────────────────────────────────────────── */
.nli-bars {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 16px;
}
.nli-row {
    display: grid;
    grid-template-columns: 100px 1fr 52px;
    align-items: center;
    gap: 10px;
}
.nli-label {
    font-size: 0.8em;
    color: #64748b;
    font-weight: 500;
}
.nli-bar-track {
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
}
.nli-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.nli-value {
    font-size: 0.8em;
    font-weight: 700;
    color: #1e293b;
    text-align: right;
}

/* ── Source Chunk ──────────────────────────────────────────── */
.source-chunk {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 4px;
}
.source-chunk-header {
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 6px;
}
.source-chunk-text {
    font-size: 0.84em;
    line-height: 1.6;
    color: #475569;
    border-left: 3px solid #cbd5e1;
    padding-left: 12px;
}

/* ── Error & Empty ────────────────────────────────────────── */
.error-card {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 14px 18px;
    color: #dc2626;
    font-size: 0.9em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.error-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #dc2626;
    color: #fff;
    font-weight: 700;
    font-size: 0.82em;
    flex-shrink: 0;
}
.empty-state {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9em;
    padding: 30px 20px;
}

/* ── Placeholder State ────────────────────────────────────── */
.placeholder {
    text-align: center;
    padding: 40px 20px;
    color: #94a3b8;
}
.placeholder-icon {
    font-size: 36px;
    margin-bottom: 8px;
    opacity: 0.4;
}
.placeholder-text {
    font-size: 0.9em;
    line-height: 1.5;
}

/* ── How It Works ─────────────────────────────────────────── */
.how-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85em;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
.how-table th {
    background: #f8fafc;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.82em;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid #e2e8f0;
}
.how-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    color: #475569;
    vertical-align: top;
}
.how-table tr:last-child td { border-bottom: none; }
.how-table td:first-child {
    font-weight: 600;
    color: #334155;
    white-space: nowrap;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #667eea;
    color: #fff;
    font-size: 0.75em;
    font-weight: 700;
    margin-right: 6px;
}

/* ── Footer ───────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 16px;
    color: #94a3b8;
    font-size: 0.78em;
    border-top: 1px solid rgba(148,163,184,0.15);
    margin-top: 12px;
}

/* ── Analyze Button Override ──────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border-radius: 10px !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
button.primary:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── Dark Mode Overrides ──────────────────────────────────── */
.dark .model-result-card {
    background: #1e293b;
    border-color: #334155;
}
.dark .model-name { color: #e2e8f0; }
.dark .confidence-value, .dark .prob-value, .dark .nli-value { color: #e2e8f0; }
.dark .prob-label, .dark .nli-label, .dark .confidence-label { color: #94a3b8; }
.dark .progress-track, .dark .prob-bar-track, .dark .nli-bar-track { background: #334155; }
.dark .source-chunk { background: #1e293b; border-color: #334155; }
.dark .source-chunk-text { color: #cbd5e1; border-left-color: #475569; }
.dark .summary-table { border-color: #334155; }
.dark .summary-row { border-bottom-color: #334155; }
.dark .summary-header { background: #1e293b; }
.dark .cell-method { color: #94a3b8; }
.dark .how-table { border-color: #334155; }
.dark .how-table th { background: #1e293b; border-bottom-color: #334155; }
.dark .how-table td { border-bottom-color: #1e293b; color: #cbd5e1; }
.dark .how-table td:first-child { color: #e2e8f0; }
"""

def build_ui():
    preloaded = get_preloaded_pdfs()
    preloaded_choices = preloaded if preloaded else ["No PDFs found"]
    default_pdf = preloaded[0] if preloaded else "No PDFs found"

    with gr.Blocks(
        title="HRES - Hallucination Detector",
    ) as app:

        # ── Hero ──
        gr.HTML("""
<div class="hero">
  <div class="hero-title">HRES</div>
  <div class="hero-sub">Hallucination Detection via Hidden-State Representations</div>
  <div class="hero-badges">
    <span class="hero-badge badge-wb">Whitebox &middot; Model Internals</span>
    <span class="hero-badge badge-bb">Blackbox &middot; NLI Verification</span>
  </div>
</div>""")

        with gr.Row(equal_height=False):
            # ── Left Column: Inputs ──
            with gr.Column(scale=1, min_width=340):
                gr.HTML('<div class="section-header">Source Document</div>')
                preloaded_dd = gr.Dropdown(
                    choices=preloaded_choices,
                    value=default_pdf,
                    label="Select a manual",
                )

                gr.HTML('<div class="section-header" style="margin-top:16px">Question & Answer</div>')
                question = gr.Textbox(
                    label="Question",
                    lines=2,
                    placeholder="e.g. What is the maximum operating temperature?",
                )
                answer = gr.Textbox(
                    label="Answer to Verify",
                    lines=3,
                    placeholder="e.g. The maximum operating temperature is 300 degrees Celsius.",
                )
                run_btn = gr.Button(
                    "Analyze Answer",
                    variant="primary",
                    size="lg",
                )

            # ── Right Column: Outputs ──
            with gr.Column(scale=2, min_width=520):
                gr.HTML('<div class="section-header">Combined Verdict</div>')
                combined_out = gr.HTML(
                    value="""
<div class="placeholder">
  <div class="placeholder-icon">&#128269;</div>
  <div class="placeholder-text">Upload a PDF, enter a question and answer,<br>then click <strong>Analyze Answer</strong> to begin.</div>
</div>""")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.HTML("""
<div class="section-header">
  <span class="pipeline-tag tag-wb" style="margin-right:6px">Whitebox</span> HRES Pipeline
</div>
<div style="font-size:0.78em;color:#94a3b8;margin-top:-6px;margin-bottom:10px">
  Gemma hidden states &#8594; PCA &#8594; SVM / XGBoost
</div>""")
                        wb_out = gr.HTML(
                            value='<div class="empty-state">Waiting for analysis...</div>',
                        )
                    with gr.Column():
                        gr.HTML("""
<div class="section-header">
  <span class="pipeline-tag tag-bb" style="margin-right:6px">Blackbox</span> NLI Pipeline
</div>
<div style="font-size:0.78em;color:#94a3b8;margin-top:-6px;margin-bottom:10px">
  FAISS retrieval &#8594; DeBERTa NLI scoring
</div>""")
                        bb_out = gr.HTML(
                            value='<div class="empty-state">Waiting for analysis...</div>',
                        )

        with gr.Accordion("How does it work?", open=False):
            gr.HTML("""
<table class="how-table">
  <tr>
    <th>Step</th>
    <th>Whitebox (HRES)</th>
    <th>Blackbox (NLI)</th>
  </tr>
  <tr>
    <td><span class="step-num">1</span> Input</td>
    <td>Feed PDF context + Q&A into Gemma</td>
    <td>Extract and chunk all PDF pages</td>
  </tr>
  <tr>
    <td><span class="step-num">2</span> Encode</td>
    <td>Extract 2304-D hidden-state vector (last layer)</td>
    <td>Embed chunks with BAAI/bge-small-en-v1.5, build FAISS index</td>
  </tr>
  <tr>
    <td><span class="step-num">3</span> Process</td>
    <td>VarianceThreshold + StandardScaler + PCA</td>
    <td>Retrieve top-5 chunks for question and answer</td>
  </tr>
  <tr>
    <td><span class="step-num">4</span> Classify</td>
    <td>SVM (RBF) / XGBoost classifier</td>
    <td>DeBERTa NLI with sliding sentence windows</td>
  </tr>
  <tr>
    <td><span class="step-num">5</span> Output</td>
    <td>CORRECT / HALLUCINATED</td>
    <td>GROUNDED / UNCERTAIN / HALLUCINATION</td>
  </tr>
</table>""")

        gr.HTML("""
<div class="app-footer">
  HRES &middot; Hallucination Detection via Hidden-State Representations
  &middot; Dual-pipeline Whitebox + Blackbox Analysis
</div>""")

        run_btn.click(
            fn=predict_with_options,
            inputs=[preloaded_dd, question, answer],
            outputs=[combined_out, wb_out, bb_out],
        )

    return app


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_md,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )
