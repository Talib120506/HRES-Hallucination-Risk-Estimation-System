"""
Model Loader Module
Handles lazy loading of all ML models (Gemma, Embedder, NLI, Classifiers)

This module mirrors the exact loading logic from src/app.py to ensure
consistent behavior between the Gradio frontend and React/FastAPI backend.
"""
import os
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ── Paths (matching app.py exactly) ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gemma-2-2b-it")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_final.pkl")
REDUCTION_PATH = os.path.join(MODELS_DIR, "reduction_final.pkl")
REDUCTION_META_PATH = os.path.join(MODELS_DIR, "reduction_metadata.csv")
VT_PATH = os.path.join(MODELS_DIR, "variance_threshold_final.pkl")

# Constants (matching app.py exactly)
HIDDEN_DIM = 2304
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

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


def get_llama():
    """Load Gemma model and tokenizer (matching app.py)"""
    global _llama_tok, _llama_model
    if _llama_model is None:
        _llama_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="cuda"
        )
        _llama_model.eval()
    return _llama_tok, _llama_model


def get_embedder():
    """Load sentence transformer embedder (BAAI/bge-small-en-v1.5)"""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    return _embedder


def get_nli():
    """Load NLI model and tokenizer"""
    global _nli_tok, _nli_model
    if _nli_model is None:
        _nli_tok = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_tok, _nli_model = _nli_model_tok, _nli_tok
        _nli_model.eval().to("cuda")
    return _nli_tok, _nli_model


def get_classifiers():
    """
    Dynamically load the best trained classifier from metadata.
    Falls back to legacy SVM/XGB/Ada/LR models if best_model_final.pkl not found.
    
    This matches the exact logic from app.py's get_classifiers() function.
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
                    print(f"[OK] Loaded best model: {_classifier_name} from {model_file}")
                    return _classifier, _classifier_name, _scaler, _reduction, _vt
            
            # Alternative: use 'classifier' field to determine which model file to load
            if 'classifier' in meta_df.columns:
                classifier_type = meta_df['classifier'].iloc[0]
                # Map classifier type to model file
                classifier_map = {
                    'LogisticRegression': ('lr_model_final.pkl', 'Logistic Regression'),
                    'SVM': ('svm_model_final.pkl', 'SVM'),
                    'XGBoost': ('xgb_model_final.pkl', 'XGBoost'),
                    'AdaBoost': ('ada_model_final.pkl', 'AdaBoost'),
                }
                if classifier_type in classifier_map:
                    model_file, model_name = classifier_map[classifier_type]
                    model_path = os.path.join(MODELS_DIR, model_file)
                    if os.path.exists(model_path):
                        _classifier = joblib.load(model_path)
                        _classifier_name = model_name
                        print(f"[OK] Loaded classifier from metadata: {_classifier_name} from {model_file}")
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
                print(f"[OK] Loaded classifier: {model_name} from {model_file}")
                break
        
        if _classifier is None:
            raise FileNotFoundError(
                "No classifier model found! Run training/train_unified.py first."
            )
    
    return _classifier, _classifier_name, _scaler, _reduction, _vt
