"""
Model Loader Module
Handles lazy loading of all ML models (TinyLlama, Embedder, NLI, Classifiers)
"""
import os
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "TinyLlama")
SVM_PATH = os.path.join(BASE_DIR, "models", "svm_model_final.pkl")
XGB_PATH = os.path.join(BASE_DIR, "models", "xgb_model_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_final.pkl")
REDUCTION_PATH = os.path.join(BASE_DIR, "models", "reduction_final.pkl")
REDUCTION_META_PATH = os.path.join(BASE_DIR, "models", "reduction_metadata.csv")
VT_PATH = os.path.join(BASE_DIR, "models", "variance_threshold_final.pkl")

# Constants
EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# ── Lazy-loaded globals ──────────────────────────────────────────────────────
_llama_tok = None
_llama_model = None
_embedder = None
_nli_tok = None
_nli_model = None
_svm = None
_xgb = None
_scaler = None
_reduction = None
_reduction_method = None
_vt = None


def get_llama():
    """Load TinyLlama model and tokenizer"""
    global _llama_tok, _llama_model
    if _llama_model is None:
        _llama_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="cuda"
        )
        _llama_model.eval()
    return _llama_tok, _llama_model


def get_embedder():
    """Load sentence transformer embedder"""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    return _embedder


def get_nli():
    """Load NLI model and tokenizer"""
    global _nli_tok, _nli_model
    if _nli_model is None:
        _nli_tok_temp = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model_temp = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_tok, _nli_model = _nli_tok_temp, _nli_model_temp
        _nli_model.eval().to("cuda")
    return _nli_tok, _nli_model


def get_classifiers():
    """Load SVM, XGBoost, scaler, and dimensionality reduction models"""
    global _svm, _xgb, _scaler, _reduction, _reduction_method, _vt
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
        _reduction = joblib.load(REDUCTION_PATH)
        _vt = joblib.load(VT_PATH)

        # Load reduction metadata to know which method was used
        if os.path.exists(REDUCTION_META_PATH):
            meta_df = pd.read_csv(REDUCTION_META_PATH)
            _reduction_method = meta_df["method"].iloc[0]
        else:
            # Fallback: assume PCA if metadata not found
            _reduction_method = "PCA"

        # Load classifier models
        if os.path.exists(SVM_PATH):
            _svm = joblib.load(SVM_PATH)
        if os.path.exists(XGB_PATH):
            _xgb = joblib.load(XGB_PATH)
    return _svm, _xgb, _scaler, _reduction, _vt
