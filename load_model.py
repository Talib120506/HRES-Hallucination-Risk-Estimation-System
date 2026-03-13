import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def loadmodel(model_path=r"D:\Hallucination-MAIN\models\TinyLlama"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  
        device_map="cuda"
    )
    print("Model loaded on GPU!\n")
    return tokenizer, model