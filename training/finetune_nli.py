import os
import argparse
import logging
import pickle
import random
import numpy as np
import pandas as pd
import torch
import faiss
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add parent dir to path so we can resolve models/ etc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

DEFAULT_DATASET  = os.path.join(BASE_DIR, "data/processed/finetuning_nli_from_audit.xlsx")
DEFAULT_PDFS_DIR = os.path.join(BASE_DIR, "resources/pdfs")
INDEX_DIR        = os.path.join(BASE_DIR, "models/nli_index")
NLI_MODEL        = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models/nli_finetuned")

LABEL_TO_NLI = {0: 1, 1: 0}   # correct->entailment, hallucinated->contradiction

MAX_LENGTH  = 256
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15

class NLIDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_length: int = MAX_LENGTH):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.examples[idx]
        enc = self.tokenizer(
            premise, hypothesis,
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }

def load_faiss_index(doc_id: str):
    save_dir = os.path.join(INDEX_DIR, doc_id.replace(".pdf", ""))
    index_file = os.path.join(save_dir, "index.faiss")
    meta_file  = os.path.join(save_dir, "meta.pkl")
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        return None, None
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunks"]

def get_best_chunk(doc_id: str, answer: str, embedder, top_k: int = 3) -> str:
    index, chunks = load_faiss_index(doc_id)
    if index is None or not chunks:
        return ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_vec = embedder.encode([answer], normalize_embeddings=True, device=device).astype(np.float32)
    distances, indices = index.search(a_vec, min(top_k, len(chunks)))
    if indices[0][0] < 0:
        return ""
    return chunks[indices[0][0]]

def build_examples(dataset_path: str, embedder, use_question_aware: bool = True) -> list:
    df = pd.read_excel(dataset_path)
    logger.info(f"Loaded {len(df)} rows from {dataset_path}")
    examples = []
    skipped  = 0
    
    if "premise" in df.columns and "hypothesis" in df.columns:
        use_prebuilt = True
    else:
        use_prebuilt = False
        
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building examples"):
        if use_prebuilt:
            premise    = str(row["premise"]).strip()
            hypothesis = str(row["hypothesis"]).strip()
            label      = int(row["label"])
            
            if not premise or premise in ("[no chunks]", "") or len(premise) < 20:
                skipped += 1
                continue
            
            examples.append((premise, hypothesis, label))
        else:
            doc_id  = row["doc_id"]
            question = str(row["question"])
            answer   = str(row["answer"])
            label    = int(row["label"])

            premise = get_best_chunk(doc_id, answer, embedder)
            if not premise:
                skipped += 1
                continue

            if use_question_aware and question:
                hypothesis = f"Given the question: {question.strip()}  The answer is: {answer.strip()}"
            else:
                hypothesis = answer.strip()

            nli_class = LABEL_TO_NLI[label]
            examples.append((premise, hypothesis, nli_class))
            
    logger.info(f"Built {len(examples)} examples, skipped {skipped}")
    return examples

def train(dataset_path: str = DEFAULT_DATASET, output_dir: str = OUTPUT_MODEL_DIR,
          epochs: int = 3, batch_size: int = 4, lr: float = 5e-6,
          use_question_aware: bool = True, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    examples = build_examples(dataset_path, embedder, use_question_aware=use_question_aware)
    if not examples:
        logger.error("No examples built.")
        return

    random.shuffle(examples)
    n = len(examples)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    train_ex = examples[:n_train]
    val_ex   = examples[n_train:n_train + n_val]
    test_ex  = examples[n_train + n_val:]
    
    # 7. Check for data leakage
    train_pairs = set((ex[0], ex[1]) for ex in train_ex)
    clean_val_ex = []
    leak_count = 0
    for ex in val_ex:
        if (ex[0], ex[1]) in train_pairs:
            leak_count += 1
        else:
            clean_val_ex.append(ex)
    if leak_count > 0:
        logger.warning(f"Found {leak_count} leaking examples in validation set! Removed them.")
        val_ex = clean_val_ex
        
    # 6. Train/Val Split Logging
    n_train = len(train_ex)
    n_val   = len(val_ex)
    n_test  = len(test_ex)
    
    labels_train = [ex[2] for ex in train_ex]
    c_count = labels_train.count(0)
    e_count = labels_train.count(1)
    n_count = labels_train.count(2)
    
    logger.info(f"Training examples : {n_train}")
    logger.info(f"Validation examples: {n_val}")
    logger.info(f"Test examples      : {n_test}")
    logger.info(f"Label distribution in train: contradiction={c_count}, entailment={e_count}, neutral={n_count}")

    logger.info(f"Loading {NLI_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, num_labels=3)
    
    # 2. Add weight decay and dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.3
            
    model.to(device)

    train_ds = NLIDataset(train_ex, tokenizer)
    val_ds   = NLIDataset(val_ex, tokenizer)
    test_ds  = NLIDataset(test_ex, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 4. Gradient accumulation
    accumulation_steps = 4
    total_steps = (len(train_loader) // accumulation_steps) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad() # zero grad at start
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        val_acc = evaluate_split(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"  New best model saved -> {best_model_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s).")
            
        # 3. Early stopping
        if epochs_without_improvement >= 2:
            logger.info("Early stopping triggered")
            break

    logger.info("Evaluating best model...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    best_model.to(device).eval()
    test_acc = evaluate_split(best_model, test_loader, device)

    print("\n" + "-" * 44)
    print("FINE-TUNING COMPLETE")
    print(f"Best val acc : {best_val_acc*100:.2f}%")
    print(f"Test acc     : {test_acc*100:.2f}%")
    print(f"Model saved  : {best_model_path}")
    print("\nTO USE THE FINE-TUNED MODEL:")
    print("In src/nli_utils.py, change:")
    print("  NLI_MODEL = \"cross-encoder/nli-deberta-v3-base\"")
    print("to:")
    print(f"  NLI_MODEL = \"{best_model_path}\"")
    print("\nThen re-run the audit:")
    print("  python training/nli_audit_pipeline.py")
    print("\nTARGET: Recall > 0.75, F1 > 0.80")
    print("-" * 44 + "\n")

def evaluate_split(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to training dataset (xlsx)")
    parser.add_argument("--use_hard_cases_only", action="store_true",
                        help="--use_hard_cases_only is NOT recommended — causes overfitting on small datasets. Use full 573-row dataset instead.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()

    dataset_path = args.dataset
    if args.use_hard_cases_only:
        dataset_path = os.path.join(BASE_DIR, "data/processed/finetuning_hard_cases_only.xlsx")

    train(dataset_path=dataset_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
