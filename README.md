# Hallucination Detection with TinyLlama

A research pipeline for studying **LLM hallucination detection** using hidden-state embeddings. The project runs TinyLlama-1.1B locally on GPU, extracts last-layer hidden state vectors from model responses, and builds a labelled dataset that can train a downstream hallucination classifier like XGBoost

---

## Project Purpose

Large language models frequently generate plausible-sounding but factually incorrect answers ("hallucinations"). This project investigates whether the **internal representations** of a model — specifically the 2048-dimensional hidden state vector at the final transformer layer — can serve as features to detect hallucinated responses before they reach the user.

**Research pipeline:**

```
TechManualQA_Dataset.csv          (ground-truth Q&A pairs)
        |
        v
  TinyLlama model (local, GPU)
        |
        v
  Last-layer hidden state (2048-D vector) at final meaningful token
        |
        v
  hallucination_training_data.csv  (embeddings + labels → classifier input)
```

---

## Directory Structure

```
Hallucination-Main/
├── download_model.py          # One-time model download from Hugging Face
├── check.py                   # CUDA memory diagnostic
├── load_model.py              # Shared helper: loads tokenizer + model
├── chat.py                    # Interactive terminal chat with TinyLlama
├── tokenlog.py                # Chat with per-token probability + hidden state logging
├── read.py                    # Batch pipeline: extract hidden states → CSV
├── TechManualQA_Dataset.csv   # Input: 350 technical Q&A pairs with ground truth
├── hallucination_training_data.csv  # Output: hidden state embeddings per Q&A
├── .gitignore
└── README.md
```

> **Note:** `models/` and `env/` are excluded from version control (see `.gitignore`). The model (~2 GB) must be downloaded locally via `download_model.py`.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10 or higher (developed on 3.12) |
| CUDA GPU | Required for model inference (CUDA 12.x recommended) |
| GPU VRAM | ~2–3 GB minimum (TinyLlama at float16) |
| Disk Space | ~2 GB for model weights |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd Hallucination-Main
```

### 2. Create a virtual environment

```bash
python -m venv env

# Windows
.\env\Scripts\activate

# Linux / macOS
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate huggingface_hub safetensors pandas tqdm
```

> For a different CUDA version, replace `cu124` with your version (e.g., `cu118`, `cu121`). Use `nvcc --version` to check.

### 4. Download the model

```bash
python download_model.py
```

This downloads **TinyLlama-1.1B-Chat-v1.0** (~2 GB) into `./models/TinyLlama/`.

### 5. Verify CUDA

```bash
python check.py
```

Expected output shows GPU memory reserved before and after a test tensor allocation. If CUDA is unavailable, model loading in subsequent scripts will fail.

---

## Scripts

### `download_model.py`

Downloads the TinyLlama model from Hugging Face Hub to the local `models/TinyLlama/` directory. Only needs to be run once.

```bash
python download_model.py
```

---

### `load_model.py`

A **shared helper module** — not run directly. Exposes:

```python
from load_model import loadmodel
tokenizer, model = loadmodel()
```

Loads the tokenizer and `AutoModelForCausalLM` in `float16` onto CUDA. Imported by `chat.py`, `tokenlog.py`, and `read.py`.

---

### `check.py`

Minimal CUDA sanity check. Allocates a 1000×1000 random tensor on the GPU and prints reserved VRAM before and after.

```bash
python check.py
```

---

### `chat.py`

Interactive multi-turn terminal chat with TinyLlama. Maintains full conversation history and applies the model's chat template. Also prints the **token count** of each response.

```bash
python chat.py
```

- Type your message and press Enter to get a response.
- Type `exit` to quit.

**Generation settings:** `max_new_tokens=256`, `temperature=0.7`, `top_p=0.9`

---

### `tokenlog.py`

Extended chat interface with **per-response introspection**. For every reply, it prints:

- The full generated response
- The last meaningful token (last word before punctuation/EOS)
- That token's **softmax probability** from the logit scores
- The full **2048-dimensional hidden state vector** from the final transformer layer at that token position

```bash
python tokenlog.py
```

Useful for understanding the model's confidence and internal representation at the point of its final committed token.

---

### `read.py`

Batch dataset processing pipeline. Reads `TechManualQA_Dataset.csv`, runs each (question, ground-truth answer) pair through TinyLlama, extracts the 2048-D last-layer hidden state at the final meaningful answer token, and writes results to `hallucination_training_data.csv`.

```bash
python read.py
```

**Output columns:** `Question`, `Last_Token`, `v_0`, `v_1`, ..., `v_2047`

> Currently processes the first 5 rows as a test batch. Modify `df.head(5)` in `read.py` to process the full dataset.

---

## Datasets

### `TechManualQA_Dataset.csv`

350 Q&A pairs sourced from technical product manuals. Key columns:

| Column | Description |
|---|---|
| `question_text` | The question posed to the model |
| `gt_answer_snippet` | The verified ground-truth answer |
| `category` | Question type (e.g., "Specification Lookup") |
| `ragas_faithfulness` | RAGAS faithfulness score (0–1) |
| `ragas_correctness` | RAGAS correctness score (0–1) |
| `judge_score` | LLM-judge evaluation score |
| `passed_strict_check` | Boolean: did the answer pass strict grounding? |

### `hallucination_training_data.csv`

Output of `read.py`. Each row is one processed Q&A pair:

| Column | Description |
|---|---|
| `Question` | Input question text |
| `Last_Token` | Final meaningful token in the model's answer |
| `v_0` … `v_2047` | 2048-D hidden state vector (float32) |

These embeddings are intended as input features for a downstream hallucination classifier (e.g., logistic regression, small MLP, or SVM trained on labelled hallucinated vs. grounded responses).

---

## Model: TinyLlama-1.1B-Chat-v1.0

| Property | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | 1.1 billion |
| Hidden size | 2048 |
| Layers | 22 transformer layers |
| Attention heads | 32 (with 4 KV heads via GQA) |
| Vocabulary | 32,000 tokens (SentencePiece BPE) |
| Context window | 2048 tokens |
| Pretrained on | 3 trillion tokens |
| Fine-tuning | UltraChat-200k (SFT) + UltraFeedback (DPO) |
| License | Apache 2.0 |

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0+cu124 | Tensor ops and CUDA inference |
| `transformers` | 4.39.3 | Model and tokenizer loading |
| `accelerate` | 1.12.0 | `device_map="cuda"` dispatch |
| `huggingface_hub` | 0.36.2 | Model downloading |
| `safetensors` | 0.7.0 | Safe model weight loading |
| `pandas` | 3.0.1 | CSV dataset I/O |
| `numpy` | 2.4.2 | Numerical operations |
| `tqdm` | 4.67.3 | Progress bars |

---

## Next Steps / Future Work

- [ ] Process the full `TechManualQA_Dataset.csv` (currently 5-row test batch)
- [ ] Label embeddings as hallucinated vs. grounded using annotation columns (`judge_score`, `passed_strict_check`)
- [ ] Train a lightweight classifier on the 2048-D hidden state features
- [ ] Evaluate classifier accuracy on held-out Q&A pairs
- [ ] Extend pipeline to larger models or multiple LLM architectures

---

## License

This project is for research and educational purposes. TinyLlama model weights are distributed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
