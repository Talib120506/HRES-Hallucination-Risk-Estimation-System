# HRES - Hallucination Detection Using Hidden-State Representations

A research pipeline for detecting **LLM hallucinations** by combining two independent approaches:

1. **Hidden-State Embeddings (HRES)** - Extract 2048-dimensional internal representations from TinyLlama's final transformer layer and use them as features for a downstream classifier.
2. **Retrieval-Augmented NLI Verification** - Retrieve source document chunks via FAISS and verify answers using Natural Language Inference (NLI) to classify them as GROUNDED, UNCERTAIN, or HALLUCINATION.

---

## Table of Contents

- [What Is This Project About?](#what-is-this-project-about)
- [How Does It Work?](#how-does-it-work)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup (Step-by-Step)](#setup-step-by-step)
- [Python Scripts (Detailed)](#python-scripts-detailed)
- [Data Files (Excel and CSV)](#data-files-excel-and-csv)
- [The PDF Source Documents](#the-pdf-source-documents)
- [Model Details: TinyLlama-1.1B-Chat-v1.0](#model-details-tinyllama-11b-chat-v10)
- [All Dependencies](#all-dependencies)
- [Next Steps / Future Work](#next-steps--future-work)
- [License](#license)

---

## What Is This Project About?

Large Language Models (LLMs) like ChatGPT sometimes generate answers that **sound correct but are factually wrong**. These are called **hallucinations**. This project explores whether we can automatically detect hallucinated answers using two complementary techniques:

- **Approach 1 (Hidden States):** When a language model generates text, its internal "neurons" produce a vector of numbers at each layer. This project extracts those internal vectors and feeds them into a simpler classifier (like XGBoost or an SVM) to predict whether the answer is hallucinated.

- **Approach 2 (NLI Verification):** Given a question and an answer, retrieve the most relevant chunk of text from the original source PDF document, then use a separate NLI model to check if the source text **supports**, **contradicts**, or is **neutral** toward the given answer.

---

## How Does It Work?

### Pipeline 1: Hidden-State Feature Extraction

```
TechManualQA_350.xlsx          (350 Q&A pairs with ground truth)
        |
        v
  Source PDF pages             (extracted via PyMuPDF from "10 pdfs/" folder)
        |
        v
  Concatenate: page_text + question + ground_truth_answer
        |
        v
  TinyLlama forward pass      (local GPU inference, no text generation)
        |
        v
  Last-layer hidden state      (2048-D float vector at the last meaningful token)
        |
        v
  hallucination_features.csv   (300 rows of embeddings + metadata -> classifier input)
```

### Pipeline 2: FAISS Retrieval + NLI Verification

```
TechManualQA_350.xlsx          (350 Q&A pairs)
        |
        v
  Source PDFs -> Extract all pages -> Clean text -> Chunk into ~100-word blocks
        |
        v
  Embed all chunks with all-MiniLM-L6-v2 -> Build FAISS index per document
        |
        v
  For each question: retrieve top-k matching chunks
        |
        v
  Run NLI (DeBERTa-v3) on each chunk vs. the ground-truth answer
        |
        v
  nli_results.csv              (300 rows with verdict: GROUNDED / UNCERTAIN / HALLUCINATION)
```

---

## Directory Structure

```
Hallucination-Main/
│
├── .gitignore                   # Excludes env/, models/, __pycache__, large binaries
├── README.md                    # This file
│
├── ── Core Scripts ──
├── download_model.py            # One-time TinyLlama model download
├── check.py                     # CUDA GPU sanity check
├── load_model.py                # Shared helper: loads tokenizer + model onto GPU
├── chat.py                      # Interactive terminal chat with TinyLlama
├── tokenlog.py                  # Chat with per-token probability + hidden-state logging
├── read.py                      # Batch hidden-state extraction (early prototype)
├── build_dataset.py             # Full hidden-state extraction pipeline (PDF-aware)
├── preview_input.py             # Preview the text input fed to TinyLlama (no GPU needed)
├── NLI_check.py                 # FAISS retrieval + NLI verification pipeline
│
├── ── Data Files ──
├── TechManualQA_350.xlsx        # Master dataset: 350 Q&A pairs from 10 product manuals
├── hallucination_features.csv   # Output of build_dataset.py (300 rows, 2048-D vectors)
├── nli_results.csv              # Output of NLI_check.py (300 rows, NLI verdicts)
│
├── ── Directories ──
├── 10 pdfs/                     # 10 source PDF technical manuals
├── nli_index/                   # Cached FAISS indexes (one subfolder per PDF document)
├── models/                      # TinyLlama model weights (~2 GB, git-ignored)
└── env/                         # Python virtual environment (git-ignored)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher (developed on 3.12) |
| **CUDA GPU** | Required for TinyLlama and NLI model inference |
| **CUDA Toolkit** | CUDA 12.x recommended (check with `nvcc --version`) |
| **GPU VRAM** | ~2-3 GB minimum (TinyLlama at float16) |
| **Disk Space** | ~2 GB for TinyLlama weights + ~120 MB for PDFs |
| **Operating System** | Windows 10/11 (developed on), Linux, or macOS |

---

## Setup (Step-by-Step)

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/Hallucination-Main.git
cd Hallucination-Main
```

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

```bash
python -m venv env
```

Activate the environment:

```bash
# Windows (Command Prompt)
.\env\Scripts\activate

# Windows (Git Bash)
source env/Scripts/activate

# Linux / macOS
source env/bin/activate
```

> You should see `(env)` at the start of your terminal prompt when the environment is active.

### Step 3: Install PyTorch with CUDA Support

PyTorch must be installed **before** the other packages, and must match your CUDA version.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **Different CUDA version?** Replace `cu124` with your version:
> - CUDA 11.8 → `cu118`
> - CUDA 12.1 → `cu121`
> - CUDA 12.4 → `cu124`
>
> Run `nvcc --version` to check your CUDA version.

### Step 4: Install All Other Dependencies

```bash
pip install transformers accelerate huggingface_hub safetensors pandas openpyxl tqdm
pip install PyMuPDF sentence-transformers faiss-cpu numpy
```

### Step 5: Download the TinyLlama Model

```bash
python download_model.py
```

This downloads **TinyLlama-1.1B-Chat-v1.0** (~2 GB) from Hugging Face into `./models/TinyLlama/`. You only need to run this once.

### Step 6: Verify CUDA Works

```bash
python check.py
```

You should see GPU memory numbers printed before and after a test tensor allocation. If this fails, your CUDA installation needs fixing before proceeding.

---

## Python Scripts (Detailed)

### `download_model.py`

**Purpose:** Downloads the TinyLlama-1.1B-Chat-v1.0 model from Hugging Face Hub.

**How to run:**
```bash
python download_model.py
```

**What it does:**
- Uses `huggingface_hub.snapshot_download()` to download the full model repository.
- Saves everything to `./models/TinyLlama/` (tokenizer files, model weights in safetensors format, config).
- Only needs to be run **once**. After that, all other scripts load the model from disk.

**Key detail:** The download is approximately 2 GB. Make sure you have sufficient disk space and a stable internet connection.

---

### `check.py`

**Purpose:** A minimal sanity check to verify your GPU and CUDA are working correctly.

**How to run:**
```bash
python check.py
```

**What it does:**
1. Prints the amount of GPU memory reserved **before** any allocation.
2. Creates a 1000x1000 random tensor on the GPU (`torch.randn(..., device="cuda")`).
3. Prints the amount of GPU memory reserved **after** the allocation.

**Key detail:** If this script throws an error like `"CUDA is not available"`, it means PyTorch was not installed with CUDA support, or your GPU drivers are missing/outdated.

---

### `load_model.py`

**Purpose:** A shared helper module that loads the TinyLlama tokenizer and model. **Not meant to be run directly** — it is imported by other scripts.

**How it is used:**
```python
from load_model import loadmodel
tokenizer, model = loadmodel()
```

**What it does:**
- Loads the tokenizer using `AutoTokenizer.from_pretrained()`.
- Loads the model using `AutoModelForCausalLM.from_pretrained()` in **float16 precision** on the CUDA GPU.
- Returns both the tokenizer and model objects.

**Key details:**
- The model path is **hardcoded** to `D:\Hallucination-MAIN\models\TinyLlama`. If your project is in a different location, update this path.
- Float16 (half-precision) reduces VRAM usage by approximately half compared to float32 while maintaining good inference quality.
- Imported by: `chat.py`, `tokenlog.py`, and `read.py`.

---

### `chat.py`

**Purpose:** An interactive terminal-based chat interface with TinyLlama. Good for manually testing how the model responds to questions.

**How to run:**
```bash
python chat.py
```

**What it does:**
1. Loads the model via `load_model.py`.
2. Starts an infinite loop that waits for your input.
3. When you type a message, it applies the TinyLlama chat template (system prompt + conversation history), generates a response, and prints it.
4. After each response, it also prints the **exact token count** of the generated text.
5. Type `exit` to quit the loop.

**Key details:**
- **Multi-turn conversation:** The full message history is maintained in a list, so the model "remembers" earlier messages in the conversation.
- **Generation parameters:** `max_new_tokens=256` (max response length), `temperature=0.7` (moderate randomness), `top_p=0.9` (nucleus sampling).
- Useful for quick experiments — ask the model questions from the dataset and manually inspect the answers.

---

### `tokenlog.py`

**Purpose:** An extended chat interface that shows **per-response introspection data** — the model's internal confidence and hidden-state vector.

**How to run:**
```bash
python tokenlog.py
```

**What it does:**
1. Loads TinyLlama directly (not using `load_model.py`).
2. For each user message, it generates a response with `output_scores=True` and `output_hidden_states=True`.
3. After generating, it **backtracks** through the generated tokens to find the last "meaningful" token (skipping EOS, punctuation, whitespace).
4. For that target token, it prints:
   - The **complete generated response** (full text).
   - The **target token** itself (e.g., `"devices"`).
   - The **softmax probability** of that token (how confident the model was in choosing that word).
   - The full **2048-dimensional hidden-state vector** from the final transformer layer at that token position.

**Key details:**
- Contains commented-out earlier versions at the top of the file (single-pass experiments).
- The backtracking logic: walks backwards through generated tokens, skipping any token that is purely punctuation, whitespace, or the `</s>` end-of-sequence marker.
- The hidden state is extracted from `outputs.hidden_states[target_index][-1][0, -1, :]` — this means: at the generation step of the target token, take the last layer (`[-1]`), first batch item (`[0]`), last sequence position (`[-1]`).
- This script is **single-turn** (no conversation history), unlike `chat.py`.

---

### `read.py`

**Purpose:** An early prototype for batch hidden-state extraction. Reads Q&A pairs from a CSV and extracts hidden-state vectors.

**How to run:**
```bash
python read.py
```

**What it does:**
1. Loads TinyLlama directly.
2. Reads `TechManualQA_Dataset.csv` (the original CSV dataset, which has since been replaced by the Excel version).
3. For each Q&A pair, concatenates the question and ground-truth answer into a prompt string.
4. Runs a **forward pass** (not generation — just processes the text and extracts internal states).
5. Backtracks to find the last meaningful token, extracts its 2048-D hidden-state vector.
6. Saves results to `hallucination_training_data.csv`.

**Key details:**
- Currently limited to **5 rows** (`df.head(5)`) as a test batch.
- This was the first version of the pipeline. It does **not** include PDF page context — only the raw question + answer.
- **Superseded by `build_dataset.py`**, which adds PDF page text as context and processes the full dataset.
- Output columns: `Question`, `Last_Token`, `v_0` through `v_2047`.

---

### `build_dataset.py`

**Purpose:** The **main hidden-state extraction pipeline**. Reads Q&A pairs from the Excel dataset, extracts the relevant PDF page text, concatenates everything, runs a TinyLlama forward pass, and saves the 2048-D hidden-state vector for each row.

**How to run:**
```bash
python build_dataset.py
python build_dataset.py --dataset TechManualQA_350.xlsx --pdfs_dir "./10 pdfs" --output hallucination_features.csv
```

**What it does (step by step):**
1. Loads `TechManualQA_350.xlsx` (350 Q&A pairs).
2. For each **answerable** row (skips "Unanswerable" category and "Not Answered" entries):
   a. Looks up `gt_page_number` to find the relevant page in the source PDF.
   b. Uses **PyMuPDF** (`fitz`) to extract the text from that specific page.
   c. Truncates page text to 1600 characters (`MAX_CONTEXT_CHARS`) to stay within model limits.
   d. Builds the input string: `"{page_text}\n\nQuestion: {question}\nAnswer: {gt_answer}"`.
   e. Tokenizes the input and runs a **single forward pass** through TinyLlama (no generation).
   f. Walks backwards through the tokens to find the last meaningful token.
   g. Extracts the 2048-D hidden-state vector from the **last transformer layer** at that position.
   h. Writes one row to `hallucination_features.csv`.
3. Supports **resume mode** (`--resume` flag, on by default) — if the script crashes or is interrupted, it skips rows that are already in the output CSV.

**Key details:**
- Currently processing only the **first 3 rows** in test mode (`df.head(3)`). The full-run loop is commented out but ready to use.
- The label column is hardcoded to `0` for all rows (grounded). Unanswerable/hallucinated rows are handled separately.
- Uses `gc.collect()` and `torch.cuda.empty_cache()` after each row to prevent GPU memory buildup.
- Model weights are loaded in float16 for reduced VRAM usage.
- Input is truncated to `max_length=1900` tokens to stay within TinyLlama's 2048-token context window.

**Command-line arguments:**
| Argument | Default | Description |
|---|---|---|
| `--dataset` | `TechManualQA_350.xlsx` | Path to the input dataset |
| `--pdfs_dir` | `./10 pdfs` | Directory containing the source PDF files |
| `--output` | `hallucination_features.csv` | Output CSV file path |
| `--model` | `./models/TinyLlama` | Path to the local TinyLlama model |
| `--resume` | `True` | Skip rows already in the output CSV |
| `--no_resume` | `False` | Force reprocessing of all rows |

---

### `preview_input.py`

**Purpose:** A debugging/inspection tool that prints the exact text input that would be fed to TinyLlama — **without loading the model or using the GPU**. Useful for verifying that PDF extraction and text concatenation are working correctly.

**How to run:**
```bash
python preview_input.py
python preview_input.py --dataset TechManualQA_350.xlsx --pdfs_dir "./10 pdfs" --n 10
```

**What it does:**
1. Reads the dataset (Excel or CSV).
2. For the first `n` answerable rows:
   a. Extracts the relevant PDF page text using PyMuPDF.
   b. Builds the same concatenated string used by `build_dataset.py`.
   c. Prints it to the terminal along with metadata (question ID, document, page number, category).

**Key details:**
- **No GPU needed** — this script only does text processing.
- Defaults to showing the first 5 rows. Change with `--n`.
- Very useful for catching data issues before running the full (slow) GPU pipeline.

---

### `NLI_check.py`

**Purpose:** The **hallucination verification pipeline** using FAISS retrieval and Natural Language Inference. This is the second major component of the project — it independently verifies whether each answer is supported by the source document.

**How to run:**
```bash
python NLI_check.py
python NLI_check.py --dataset TechManualQA_350.xlsx --pdfs_dir "./10 pdfs" --output nli_results.csv
```

**What it does (step by step):**
1. **PDF Processing:** Extracts all pages from each PDF using PyMuPDF. Cleans the text (removes page numbers, language headers, figure references). Splits text into overlapping chunks of ~100 words with 30-word overlap at sentence boundaries.

2. **Embedding:** Embeds all chunks using the `all-MiniLM-L6-v2` sentence transformer model on GPU. Each chunk becomes a 384-dimensional vector.

3. **FAISS Indexing:** Builds one FAISS `IndexFlatIP` (inner product / cosine similarity) index per source document. These indexes are **cached** in `nli_index/` — one subfolder per PDF containing `index.faiss` and `meta.pkl`.

4. **Retrieval:** For each question-answer pair:
   - Embeds both the question and the answer as query vectors.
   - Retrieves candidate chunks using **three strategies** (merged and deduplicated):
     - Answer-based full-document retrieval (top-k most similar to the answer).
     - Question-based full-document retrieval (top-k most similar to the question).
     - Page-filtered retrieval using `gt_page_number` if available.
   - This multi-strategy approach ensures the best matching chunk is found even if the page number in the dataset is slightly off.

5. **NLI Scoring:** For each candidate chunk, runs sentence-level NLI using the `cross-encoder/nli-deberta-v3-small` model:
   - Splits the chunk into individual sentences.
   - Builds sliding windows of 1, 2, and 3 consecutive sentences as candidate premises (handles answers that span multiple sentences in the source).
   - For multi-step answers (containing newlines), scores each step independently and takes the **minimum entailment** across all steps.
   - Picks the chunk + sentence combination with the **highest entailment probability**.

6. **Verdict:** Maps the NLI label to a final verdict:
   - `entailment` → **GROUNDED** (answer is supported by the source)
   - `neutral` → **UNCERTAIN** (cannot confirm or deny)
   - `contradiction` → **HALLUCINATION** (answer contradicts the source)

7. **Output:** Saves all results to `nli_results.csv` and prints a summary table showing how many answers are GROUNDED, UNCERTAIN, or HALLUCINATION.

**Key details:**
- Uses **two separate models**: `all-MiniLM-L6-v2` for embedding (fast, 384-D) and `cross-encoder/nli-deberta-v3-small` for NLI classification (3-class: entailment, contradiction, neutral).
- DeBERTa label order is `[contradiction=0, entailment=1, neutral=2]` — this is hardcoded in `NLI_LABEL_MAP`.
- The NLI model is lazy-loaded (singleton pattern via global variables `_nli_tok` and `_nli_model`).
- Skips "Unanswerable" rows and "Not Answered" entries automatically.
- `TOP_K = 5` chunks are retrieved when no page filter is applied.

**Command-line arguments:**
| Argument | Default | Description |
|---|---|---|
| `--dataset` | `TechManualQA_350.xlsx` | Path to the input dataset |
| `--pdfs_dir` | `./10 pdfs` | Directory containing the source PDF files |
| `--output` | `nli_results.csv` | Output CSV file path |

---

## Data Files (Excel and CSV)

### `TechManualQA_350.xlsx` (Input Dataset)

The master dataset containing **350 Q&A pairs** sourced from 10 real technical product manuals. This is the single source of truth for all pipelines.

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Unique identifier (e.g., `bosch_oven_Q001`) |
| `doc_id` | string | Source PDF filename (e.g., `bosch_oven.pdf`) |
| `question_text` | string | The question posed about the product manual |
| `gt_answer_snippet` | string | The verified ground-truth answer extracted from the manual |
| `gt_page_number` | integer | The page number in the PDF where the answer is found |
| `category` | string | Question type: `Specification Lookup`, `Procedural`, `Safety Warning`, `Troubleshooting`, `Unanswerable` |
| `ragas_faithfulness` | float (0-1) | RAGAS faithfulness score |
| `ragas_correctness` | float (0-1) | RAGAS correctness score |
| `judge_score` | float | LLM-judge evaluation score |
| `passed_strict_check` | boolean | Whether the answer passed strict grounding checks |

**Important:** Of the 350 rows, approximately **300 are answerable** (have valid page numbers and ground-truth answers). The remaining ~50 are marked as "Unanswerable" and are skipped by both pipelines.

---

### `hallucination_features.csv` (Output of `build_dataset.py`)

Contains **300 rows** — one per answerable Q&A pair. Each row includes metadata, a label, and the 2048-dimensional hidden-state vector extracted from TinyLlama.

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Matches the ID in `TechManualQA_350.xlsx` |
| `doc_id` | string | Source PDF filename |
| `category` | string | Question type |
| `question` | string | The question text |
| `gt_answer` | string | Ground-truth answer (truncated to 300 chars) |
| `label` | integer | `0` = grounded (currently all rows are label 0) |
| `seq_len` | integer | Total number of tokens in the input sequence |
| `target_index` | integer | Token position of the last meaningful token |
| `v_0` through `v_2047` | float | The 2048 hidden-state values from TinyLlama's last layer |

**File size:** ~5.8 MB (2048 float columns per row).

**Usage:** These 2048 features (`v_0` to `v_2047`) are the input for training a downstream hallucination classifier (e.g., XGBoost, SVM, logistic regression, or a small neural network).

---

### `nli_results.csv` (Output of `NLI_check.py`)

Contains **300 rows** — one per verified Q&A pair. Each row shows whether the answer is supported by the source PDF according to the NLI model.

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Matches the ID in `TechManualQA_350.xlsx` |
| `doc_id` | string | Source PDF filename |
| `category` | string | Question type |
| `question` | string | The question text |
| `answer` | string | The ground-truth answer that was verified |
| `retrieved_context` | string | The best-matching chunk from the PDF |
| `retrieved_pages` | string | Comma-separated page numbers that were searched |
| `nli_label` | string | Raw NLI label: `entailment`, `neutral`, or `contradiction` |
| `verdict` | string | Human-readable: `GROUNDED`, `UNCERTAIN`, or `HALLUCINATION` |
| `entailment` | float | Probability that the answer is supported (0.0 to 1.0) |
| `neutral` | float | Probability that the evidence is neutral (0.0 to 1.0) |
| `contradiction` | float | Probability that the answer contradicts the source (0.0 to 1.0) |

---

## The PDF Source Documents

The `10 pdfs/` folder contains **10 real product manuals** used as the ground-truth source for all Q&A pairs:

| PDF File | Product | Size |
|---|---|---|
| `bosch_oven.pdf` | Bosch Oven | 3.6 MB |
| `dewalt_saw.pdf` | DeWalt Saw | 7.4 MB |
| `dyson_v12.pdf` | Dyson V12 Vacuum | 42.1 MB |
| `electrolux_oven.pdf` | Electrolux Oven | 1.7 MB |
| `hilti_hammer.pdf` | Hilti Hammer Drill | 2.4 MB |
| `laptop_lenovo_tc_x1.pdf` | Lenovo ThinkCentre X1 Laptop | 7.4 MB |
| `makita_drill.pdf` | Makita Drill | 3.8 MB |
| `omron_monitor.pdf` | Omron Blood Pressure Monitor | 1.5 MB |
| `prusa_3d-printer.pdf` | Prusa 3D Printer | 44.2 MB |
| `spot_boston_dynamics.pdf` | Boston Dynamics Spot | 8.6 MB |

---

## The `nli_index/` Directory

This directory contains **cached FAISS indexes** built by `NLI_check.py`. Each PDF gets its own subfolder (e.g., `nli_index/bosch_oven/`) containing:

- `index.faiss` — The FAISS inner-product index storing all chunk embeddings.
- `meta.pkl` — Pickled metadata (chunk texts and page numbers).

These indexes are built once and reused across runs to avoid re-processing PDFs.

---

## Model Details: TinyLlama-1.1B-Chat-v1.0

| Property | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | 1.1 billion |
| Hidden size | 2048 (this is the dimensionality of extracted vectors) |
| Layers | 22 transformer layers |
| Attention heads | 32 (with 4 KV heads via Grouped Query Attention) |
| Vocabulary | 32,000 tokens (SentencePiece BPE tokenizer) |
| Context window | 2048 tokens |
| Pretrained on | 3 trillion tokens |
| Fine-tuning | UltraChat-200k (SFT) + UltraFeedback (DPO) |
| License | Apache 2.0 |

---

## All Dependencies

### Core Dependencies

| Package | Purpose |
|---|---|
| `torch` | PyTorch — tensor operations, GPU inference, and all neural network computation |
| `transformers` | Hugging Face Transformers — loads TinyLlama tokenizer and model, and the DeBERTa NLI model |
| `accelerate` | Hugging Face Accelerate — enables `device_map="cuda"` for automatic GPU placement |
| `huggingface_hub` | Hugging Face Hub SDK — downloads the TinyLlama model from the Hugging Face repository |
| `safetensors` | Safe model weight loading format (used by TinyLlama's saved weights) |

### Data Processing

| Package | Purpose |
|---|---|
| `pandas` | DataFrame operations — reads/writes CSV and Excel files, data manipulation |
| `openpyxl` | Excel file engine — required by pandas to read `.xlsx` files |
| `numpy` | Numerical operations — array handling, used with FAISS embeddings |
| `tqdm` | Progress bars — shows processing progress during batch operations |

### PDF Processing

| Package | Purpose |
|---|---|
| `PyMuPDF` (imported as `fitz`) | Extracts text from PDF pages. Install with `pip install PyMuPDF` |

### NLI Pipeline (used by `NLI_check.py`)

| Package | Purpose |
|---|---|
| `sentence-transformers` | Sentence embedding — the `all-MiniLM-L6-v2` model that converts text chunks into 384-D vectors |
| `faiss-cpu` | Facebook AI Similarity Search — builds and queries vector indexes for fast chunk retrieval |

### Quick Install (All at Once)

```bash
# Step 1: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install everything else
pip install transformers accelerate huggingface_hub safetensors pandas openpyxl numpy tqdm PyMuPDF sentence-transformers faiss-cpu
```

---

## Next Steps / Future Work

- [ ] Process the full dataset in `build_dataset.py` (switch from `df.head(3)` to `df.iterrows()`)
- [ ] Generate hallucinated answers using TinyLlama and label them as `label=1`
- [ ] Train a lightweight classifier (XGBoost, SVM, or small MLP) on the 2048-D hidden-state features
- [ ] Combine HRES features with NLI verdicts for a multi-signal classifier
- [ ] Evaluate classifier accuracy on held-out Q&A pairs
- [ ] Extend pipeline to larger models or compare across multiple LLM architectures

---

## License

This project is for research and educational purposes. TinyLlama model weights are distributed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
