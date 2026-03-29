import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def loadmodel(model_path=None):
    if model_path is None:
        # Calculate relative path from src/ to models/TinyLlama
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(BASE_DIR, "models", "TinyLlama")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    print("Model loaded on GPU!\n")
    return tokenizer, model
