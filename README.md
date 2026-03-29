# HRES -- Hallucination Detection Using Hidden-State Representations

A dual-pipeline system for detecting LLM hallucinations by combining **whitebox** (model internal analysis) and **blackbox** (NLI text verification) approaches, with a Gradio web interface for interactive use.

---

## Table of Contents

- [What Is This Project About?](#what-is-this-project-about)
- [How It Works](#how-it-works)
  - [Whitebox Pipeline (HRES)](#whitebox-pipeline-hres)
  - [Blackbox Pipeline (NLI)](#blackbox-pipeline-nli)
  - [Combined Verdict](#combined-verdict)
- [Project Structure](#project-structure)
- [Complete File Reference](#complete-file-reference)
  - [Core Pipeline Scripts](#core-pipeline-scripts)
  - [Web Application](#web-application)
  - [Utility / Development Scripts](#utility--development-scripts)
- [Data Files](#data-files)
  - [Input Data (Excel)](#input-data-excel)
  - [Generated Features (Excel)](#generated-features-excel)
  - [Trained Model Artifacts (PKL)](#trained-model-artifacts-pkl)
  - [Other Data Outputs](#other-data-outputs)
- [Source PDFs](#source-pdfs)
- [Models Used](#models-used)
- [Prerequisites](#prerequisites)
- [Setup (Step-by-Step)](#setup-step-by-step)
- [Running the Application](#running-the-application)
- [Reproducing the Pipeline from Scratch](#reproducing-the-pipeline-from-scratch)
- [Dependencies](#dependencies)
- [License](#license)

---

## What Is This Project About?

Large Language Models (LLMs) sometimes generate answers that **sound correct but are factually wrong**. These are called **hallucinations**. This project detects hallucinated answers using two independent techniques:

**Whitebox Approach** -- Opens up the LLM (TinyLlama) and inspects its **internal neural activations** (hidden-state vectors). When a model produces a hallucinated answer, its internal state differs from when it produces a grounded one. A classifier (SVM/XGBoost) learns to distinguish these patterns.

**Blackbox Approach** -- Treats the LLM as a black box. Retrieves relevant text from the source PDF using semantic search (FAISS), then uses a Natural Language Inference model (DeBERTa) to check if the source **supports**, **contradicts**, or is **neutral** toward the answer.

Both results are combined into a single verdict: **CORRECT**, **HALLUCINATED**, or **UNCERTAIN**.

---

## How It Works

### Whitebox Pipeline (HRES)

```
PDF page text + Question + Answer
       |
       v
TinyLlama forward pass (no text generation, just processes the input)
       |
       v
Extract 2048-D hidden-state vector from the last transformer layer
at the last meaningful token position
       |
       v
VarianceThreshold  ->  StandardScaler  ->  PCA (dimensionality reduction)
       |
       v
SVM (RBF kernel) or XGBoost classifier
       |
       v
Output: CORRECT or HALLUCINATED  (with confidence %)
```

**Why it works:** The 2048-dimensional vector from TinyLlama's final layer encodes a "fingerprint" of how the model internally processed the input. Hallucinated answers produce different activation patterns than correct ones. The classifier learns to tell them apart.

**What "whitebox" means:** You need access to the model's internal weights and hidden states -- you're looking *inside* the model.

### Blackbox Pipeline (NLI)

```
PDF document
       |
       v
Extract all pages  ->  Clean text  ->  Chunk into ~100-word segments
       |
       v
Embed all chunks with all-MiniLM-L6-v2 (384-D sentence vectors)
       |
       v
Build FAISS index (cosine similarity search)
       |
       v
Retrieve top-5 chunks matching the question and answer
       |
       v
For each chunk: run DeBERTa NLI with sliding sentence windows
(1-sentence, 2-sentence, 3-sentence combinations + full chunk)
       |
       v
Pick the chunk with the highest entailment score
       |
       v
Output: GROUNDED, UNCERTAIN, or HALLUCINATION (with probabilities)
```

**Why it works:** If the source document supports the answer (entailment), it's grounded. If it contradicts it, it's hallucinated. If there's no clear signal, it's uncertain.

**What "blackbox" means:** You don't need access to the model that generated the answer. You only examine the input text and output text.

### Combined Verdict

| Whitebox says   | Blackbox says   | Combined Verdict |
|-----------------|-----------------|------------------|
| CORRECT         | GROUNDED        | **CORRECT**      |
| HALLUCINATED    | HALLUCINATION   | **HALLUCINATED** |
| HALLUCINATED    | GROUNDED/NEUTRAL| **UNCERTAIN**    |
| CORRECT         | HALLUCINATION   | **UNCERTAIN**    |
| Mixed           | Mixed           | **UNCERTAIN**    |

When the pipelines disagree, the system flags the answer as UNCERTAIN and recommends manual review.

---

## Project Structure

```text
Hallucination-Main-TinyLlama/
├── data/
│   ├── processed/                 # Extracted features, shuffled features
│   ├── raw/                       # Original datasets (TechManualQA_350.xlsx, training_pairs.xlsx)
│   ├── results/                   # Evaluation results (e.g. nli_results.csv)
│   └── training_data/             # Correct & Incorrect answers, final processed excel
├── models/
│   ├── nli_index/                 # Cached FAISS indexes per PDF
│   ├── TinyLlama/                 # TinyLlama-1.1B model weights
│   └── *.pkl / *.csv              # Trained ML models and transformers (pca_final.pkl, svm_model_final.pkl, etc.)
├── resources/
│   └── pdfs/                      # Source PDF technical manuals (e.g. bosch_oven.pdf, etc.)
├── src/
│   ├── app.py                     # Main Gradio Web UI combining both pipelines
│   ├── chat.py                    # Interactive chat with TinyLlama
│   ├── check.py                   # CUDA/GPU sanity check
│   ├── download_model.py          # Script to download TinyLlama
│   ├── load_model.py              # Shared model loader logic
│   └── utils_internal/            # Utility scripts (preview_input.py, read.py, tokenlog.py)
├── training/
│   ├── build_dataset_final.py     # Extract hidden-state features from TinyLlama
│   ├── build_dataset_original.py  # Original prototype dataset builder
│   ├── NLI_check.py               # Run blackbox NLI verification
│   ├── shuffle.py                 # Shuffle the feature dataset
│   └── train.py                   # Train SVM/XGBoost classifiers with grid search
├── env/                           # Python virtual environment (git-ignored)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Complete File Reference

### Core Pipeline Scripts

#### `build_dataset_final.py` -- Feature Extraction

**What it does:** Extracts hidden-state features from TinyLlama for both correct and incorrect answers. This creates the training data for the whitebox classifiers.

**How it works:**
1. Reads `correct answers.xlsx` (label=0) and `incorrect answers.xlsx` (label=1)
2. For each Q&A pair, extracts the relevant page from the source PDF using PyMuPDF
3. Builds input: `"{page_text}\n\nQuestion: {question}\nAnswer: {answer}"`
4. Runs a TinyLlama forward pass (no text generation)
5. Finds the last meaningful token (walks backwards, skipping EOS/punctuation)
6. Extracts the 2048-D hidden-state vector from the final transformer layer at that position
7. Saves all features to `features_correct_incorrect.xlsx`

**Key functions:**
- `extract_page(pdf_path, page)` -- Extracts text from a specific PDF page (max 1600 chars)
- `find_last_meaningful_token(input_ids, tokenizer)` -- Locates the target token position
- `get_last_hidden_state(text, tokenizer, model)` -- Runs forward pass, returns 2048-D vector + metadata
- `process_answers(df, answer_type, label, ...)` -- Processes all rows for a given label
- `build_dataset(correct_xlsx, incorrect_xlsx, ...)` -- Main orchestrator

**CLI arguments:** `--correct`, `--incorrect`, `--pdfs_dir`, `--output`, `--model`

**Input:** `correct answers.xlsx`, `incorrect answers.xlsx`, PDFs, TinyLlama model
**Output:** `features_correct_incorrect.xlsx` (each row = metadata + label + 2048 features + seq_len + target_index)

---

#### `shuffle.py` -- Dataset Shuffling

**What it does:** Randomly shuffles the rows of the feature dataset to prevent ordering bias during cross-validation.

```python
df = pd.read_excel("features_correct_incorrect.xlsx")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_excel("features_shuffled_final.xlsx", index=False)
```

**Input:** `features_correct_incorrect.xlsx`
**Output:** `features_shuffled_final.xlsx`

---

#### `train.py` -- Model Training with Grid Search

**What it does:** Trains SVM and XGBoost classifiers on the hidden-state features, with full hyperparameter grid search and 5-fold cross-validation. Saves the final trained model and preprocessing pipeline.

**Training pipeline (9 steps):**

1. **Load Data** -- Reads `features_shuffled_final.xlsx`. Separates features (`v_0`..`v_2047`, `seq_len`, `target_index` = 2050 raw features) from labels (0=correct, 1=hallucinated).

2. **VarianceThreshold** (threshold=1e-6) -- Removes features with near-zero variance.

3. **StandardScaler** -- Standardizes all features to zero mean and unit variance.

4. **PCA Variance Analysis** -- Fits full PCA and reports how many components capture 80%, 85%, 90%, 95%, 99% of variance.

5. **XGBoost Grid Search** -- 5-fold stratified CV across:
   - PCA components: [150, 200, 250, 300]
   - 2 configs varying: `max_depth` (1-2), `learning_rate` (0.005), `n_estimators` (1000 with early stopping at 80 rounds)
   - Heavy regularization: `reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`

6. **SVM Grid Search** -- 5-fold stratified CV across:
   - PCA components: [150, 200, 250, 300]
   - 8 configs: `C` in {1, 10, 50, 100} x `gamma` in {"scale", "auto"}
   - RBF kernel, `probability=True`

7. **Head-to-Head Comparison** -- Best XGBoost vs best SVM by mean CV accuracy. Ties default to XGBoost.

8. **Detailed Evaluation** -- Full metrics for both models: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report.

9. **Final Training & Save** -- Winner is retrained on ALL data. Saves:
   - `scaler_final.pkl`, `pca_final.pkl`, `variance_threshold_final.pkl`
   - `svm_model_final.pkl` or `xgb_model_final.pkl` (whichever won)
   - `training_data_final.xlsx` (PCA-transformed data with metadata)

**Input:** `features_shuffled_final.xlsx`
**Output:** 5 PKL files + `training_data_final.xlsx`

---

#### `NLI_check.py` -- Blackbox NLI Verification

**What it does:** Independently verifies whether each answer in the dataset is supported by its source PDF, using semantic retrieval (FAISS) and Natural Language Inference (DeBERTa).

**Pipeline:**
1. **PDF Processing** -- Extracts all pages with PyMuPDF; cleans text (removes page numbers, language headers, figure references); chunks into ~100-word overlapping segments (30-word overlap at sentence boundaries)
2. **Embedding** -- Encodes all chunks with `all-MiniLM-L6-v2` (384-D vectors) on GPU
3. **FAISS Indexing** -- Builds one `IndexFlatIP` (cosine similarity) index per PDF document; caches indexes in `nli_index/` for reuse
4. **Retrieval** -- For each Q&A pair, retrieves candidates using three strategies:
   - Answer-based full-document search (top-5)
   - Question-based full-document search (top-5)
   - Page-filtered search using known page number (if available)
   - Results are merged and deduplicated
5. **NLI Scoring** -- For each candidate chunk, runs DeBERTa NLI with sliding sentence windows (1, 2, 3 sentences + full chunk). Multi-step answers are scored per-step with minimum entailment. Picks the highest entailment score.
6. **Verdict** -- Maps NLI label to: entailment=GROUNDED, neutral=UNCERTAIN, contradiction=HALLUCINATION

**Key class:** `FAISSDocIndex` -- Manages per-document FAISS indexes with `build()` and `retrieve()` methods.

**CLI arguments:** `--dataset`, `--pdfs_dir`, `--output`

**Input:** `TechManualQA_350.xlsx`, PDFs in `./10 pdfs/`
**Output:** `nli_results.csv`

---

### Web Application

#### `app.py` -- Gradio Web Interface

**What it does:** Provides a web UI that runs both pipelines on user-provided PDFs and displays combined results.

**User workflow:**
1. Upload a PDF or select a preloaded manual from the dropdown
2. Enter a question about the document
3. Enter an answer to verify
4. Click "Analyze Answer"
5. View the combined verdict, whitebox details, and blackbox details

**Backend pipeline:**

- **`whitebox_predict(pdf_path, question, answer)`**
  1. Extracts first 3 pages of PDF text (max 1600 chars combined)
  2. Runs TinyLlama forward pass, extracts 2048-D hidden-state vector
  3. Applies saved preprocessing: VarianceThreshold -> StandardScaler -> PCA
  4. Classifies with SVM and/or XGBoost (only loads models whose feature dimensions match the PCA)
  5. Returns: CORRECT/HALLUCINATED with confidence and probability breakdown

- **`blackbox_predict(pdf_path, question, answer)`**
  1. Chunks all PDF pages, builds in-memory FAISS index
  2. Retrieves top-5 chunks for question and answer
  3. Runs DeBERTa NLI with sliding sentence windows
  4. Returns: GROUNDED/UNCERTAIN/HALLUCINATION with probabilities and matching source text

- **`predict()`** -- Runs both pipelines, combines verdicts, formats results as styled HTML

**Model loading:** All models (TinyLlama, MiniLM, DeBERTa, SVM, XGBoost, PCA, scaler) are **lazy-loaded** on first use via singleton patterns (`get_llama()`, `get_embedder()`, `get_nli()`, `get_classifiers()`).

**Error handling:** Both pipelines are wrapped in try-except blocks. Failures display styled error messages instead of crashing the UI.

**UI features:**
- Gradient-styled header with pipeline badges
- Color-coded verdict cards (green/red/amber)
- Confidence progress bars and probability breakdowns
- Retrieved source chunk display with styled quote block
- Side-by-side pipeline comparison table
- Dark mode support
- "How does it work?" accordion with step-by-step comparison table

**Run:**
```bash
python app.py
# Opens at http://127.0.0.1:7860
```

---

### Utility / Development Scripts

#### `download_model.py` -- Model Download

Downloads TinyLlama-1.1B-Chat-v1.0 from Hugging Face Hub to `./models/TinyLlama/`. Run once before using any pipeline script.

```bash
python download_model.py
```

---

#### `check.py` -- GPU Check

Verifies CUDA is working by allocating a test tensor on the GPU and printing memory usage.

```bash
python check.py
```

---

#### `load_model.py` -- Shared Model Loader

Helper module (not run directly) that loads TinyLlama tokenizer + model in float16 on CUDA. Imported by `chat.py` and other scripts.

```python
from load_model import loadmodel
tokenizer, model = loadmodel()
```

**Note:** Default model path is hardcoded to `D:\Hallucination-MAIN\models\TinyLlama`. Update if your project is in a different location.

---

#### `chat.py` -- Interactive Chat

Multi-turn terminal chat with TinyLlama. Maintains conversation history. Useful for manually testing model responses.

- Generation parameters: `max_new_tokens=256`, `temperature=0.7`, `top_p=0.9`
- Reports token count after each response
- Type `exit` to quit

```bash
python chat.py
```

---

#### `tokenlog.py` -- Token-Level Analysis

Extended chat that logs per-response introspection data:
- The target token (last meaningful token in the response)
- Its softmax probability (model confidence)
- The full 2048-D hidden-state vector at that position

Useful for understanding how hidden states differ between confident and uncertain responses.

```bash
python tokenlog.py
```

---

#### `preview_input.py` -- Input Preview (No GPU)

Prints the exact text that would be fed to TinyLlama for the first N answerable rows. **Does not load any model or use the GPU.** Useful for verifying PDF extraction and text concatenation.

```bash
python preview_input.py --n 10
```

---

#### `read.py` -- Early Prototype

First proof-of-concept for hidden-state extraction. Processes 5 rows from a CSV, extracts hidden states. Does NOT include PDF context -- only uses question + answer text. **Superseded by `build_dataset_final.py`.**

---

#### `build_dataset_original.py` -- Original Dataset Builder

Earlier version of `build_dataset_final.py`. Processes only correct answers (label=0) from `TechManualQA_350.xlsx`, outputs CSV, includes resume support. Currently in test mode (3 rows). **Superseded by `build_dataset_final.py`.**

---

## Data Files

### Input Data (Excel)

| File | Rows | Description |
|------|------|-------------|
| `TechManualQA_350.xlsx` | 350 | Master dataset with Q&A pairs from 10 product manuals |
| `correct answers.xlsx` | ~300 | Correct (grounded) answers with page references |
| `incorrect answers.xlsx` | ~300 | Hallucinated/incorrect answers |

**Columns in `TechManualQA_350.xlsx`:**

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | string | Unique ID (e.g., `bosch_oven_Q001`) |
| `doc_id` | string | Source PDF filename (e.g., `bosch_oven.pdf`) |
| `question_text` | string | The question about the product manual |
| `gt_answer_snippet` | string | Verified ground-truth answer |
| `gt_page_number` | integer | Page number where the answer is found |
| `category` | string | `Specification Lookup`, `Procedural`, `Safety Warning`, `Troubleshooting`, `Unanswerable` |
| `ragas_faithfulness` | float | RAGAS faithfulness score (0-1) |
| `ragas_correctness` | float | RAGAS correctness score (0-1) |
| `judge_score` | float | LLM-judge evaluation score |
| `passed_strict_check` | boolean | Whether the answer passed strict grounding checks |

**Columns in `correct answers.xlsx` / `incorrect answers.xlsx`:**

| Column | Description |
|--------|-------------|
| `question_id` | Matches ID in master dataset |
| `doc_id` | Source PDF filename |
| `question_text` | The question |
| `gt_answer_snippet` | The answer (correct or incorrect) |
| `gt_page_number` | PDF page number |

---

### Generated Features (Excel)

| File | Source Script | Description |
|------|-------------|-------------|
| `features_correct_incorrect.xlsx` | `build_dataset_final.py` | Raw features: metadata + label + 2048 hidden-state values + `seq_len` + `target_index` |
| `features_shuffled_final.xlsx` | `shuffle.py` | Same as above, rows shuffled with `random_state=42` |
| `training_data_final.xlsx` | `train.py` | PCA-transformed features with metadata and labels |

**Feature columns in `features_correct_incorrect.xlsx`:**

| Columns | Count | Description |
|---------|-------|-------------|
| `question_id`, `doc_id`, `question`, `answer`, `answer_type` | 5 | Metadata |
| `label` | 1 | 0 = correct, 1 = hallucinated |
| `v_0` through `v_2047` | 2048 | Hidden-state vector from TinyLlama's last layer |
| `seq_len` | 1 | Total number of tokens in the input |
| `target_index` | 1 | Token position of the last meaningful token |
| **Total** | **2056** | |

---

### Trained Model Artifacts (PKL)

| File | Contents | Saved by |
|------|----------|----------|
| `variance_threshold_final.pkl` | Fitted `VarianceThreshold` (removes near-zero-variance features) | `train.py` |
| `scaler_final.pkl` | Fitted `StandardScaler` (zero mean, unit variance) | `train.py` |
| `pca_final.pkl` | Fitted `PCA` (dimensionality reduction to N components) | `train.py` |
| `svm_model_final.pkl` | Trained SVM classifier (RBF kernel, probability=True) | `train.py` |
| `xgb_model_final.pkl` | Trained XGBoost classifier | `train.py` |

**Note:** `train.py` only saves the winning model's PKL file and its matching PCA. If both `svm_model_final.pkl` and `xgb_model_final.pkl` exist on disk, they may have been trained with different PCA dimensions. `app.py` handles this by checking each model's expected feature count against the PCA output dimension.

---

### Other Data Outputs

| File | Source | Description |
|------|--------|-------------|
| `hallucination_features.csv` | `build_dataset_original.py` | Early prototype output (300 rows, correct answers only) |
| `nli_results.csv` | `NLI_check.py` | NLI verification results with verdict and probabilities |

**Columns in `nli_results.csv`:**

| Column | Description |
|--------|-------------|
| `question_id`, `doc_id`, `category`, `question`, `answer` | Metadata |
| `retrieved_context` | Best-matching PDF chunk |
| `retrieved_pages` | Page numbers searched |
| `nli_label` | Raw NLI label: `entailment`, `neutral`, `contradiction` |
| `verdict` | `GROUNDED`, `UNCERTAIN`, `HALLUCINATION` |
| `entailment`, `neutral`, `contradiction` | NLI probabilities (0.0 to 1.0) |

---

## Source PDFs

The `10 pdfs/` folder contains 10 real product manuals used as ground-truth sources:

| PDF File | Product |
|----------|---------|
| `bosch_oven.pdf` | Bosch Oven |
| `dewalt_saw.pdf` | DeWalt Saw |
| `dyson_v12.pdf` | Dyson V12 Vacuum |
| `electrolux_oven.pdf` | Electrolux Oven |
| `hilti_hammer.pdf` | Hilti Hammer Drill |
| `laptop_lenovo_tc_x1.pdf` | Lenovo ThinkCentre X1 |
| `makita_drill.pdf` | Makita Drill |
| `omron_monitor.pdf` | Omron Blood Pressure Monitor |
| `prusa_3d-printer.pdf` | Prusa 3D Printer |
| `spot_boston_dynamics.pdf` | Boston Dynamics Spot |

The `nli_index/` directory caches pre-built FAISS indexes for these PDFs (one subfolder per document containing `index.faiss` and `meta.pkl`). These are created by `NLI_check.py` on first run and reused on subsequent runs.

---

## Models Used

### TinyLlama-1.1B-Chat-v1.0 (Whitebox Feature Extraction)

| Property | Value |
|----------|-------|
| Architecture | LlamaForCausalLM |
| Parameters | 1.1 billion |
| Hidden size | 2048 (dimensionality of extracted vectors) |
| Layers | 22 transformer layers |
| Attention heads | 32 (with 4 KV heads via Grouped Query Attention) |
| Vocabulary | 32,000 tokens (SentencePiece BPE) |
| Context window | 2048 tokens |
| Precision | float16 (half-precision for reduced VRAM) |
| License | Apache 2.0 |

### all-MiniLM-L6-v2 (Blackbox Chunk Embedding)

| Property | Value |
|----------|-------|
| Type | Sentence Transformer |
| Output dimension | 384 |
| Purpose | Encodes PDF text chunks for FAISS similarity search |

### cross-encoder/nli-deberta-v3-small (Blackbox NLI Classification)

| Property | Value |
|----------|-------|
| Type | Cross-Encoder (sequence classification) |
| Classes | 3: contradiction (0), entailment (1), neutral (2) |
| Purpose | Determines if source text supports or contradicts the answer |

### SVM / XGBoost (Whitebox Classification)

| Property | SVM | XGBoost |
|----------|-----|---------|
| Kernel / Type | RBF | Gradient boosted trees |
| Input features | PCA-reduced hidden states | PCA-reduced hidden states |
| Output | CORRECT / HALLUCINATED | CORRECT / HALLUCINATED |
| Probability output | Yes | Yes |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.10+ (developed on 3.12) |
| CUDA GPU | Required for TinyLlama, NLI, and embedding models |
| CUDA Toolkit | 12.x recommended (`nvcc --version` to check) |
| GPU VRAM | 2-3 GB minimum (TinyLlama at float16) |
| Disk Space | ~2 GB for TinyLlama weights + ~120 MB for PDFs |
| OS | Windows 10/11 (developed on), Linux, or macOS |

---

## Setup (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Hallucination-Main.git
cd Hallucination-Main
```

### 2. Create a Virtual Environment

```bash
python -m venv env
```

Activate it:

```bash
# Windows (Command Prompt)
.\env\Scripts\activate

# Windows (Git Bash)
source env/Scripts/activate

# Linux / macOS
source env/bin/activate
```

### 3. Install PyTorch with CUDA

PyTorch must be installed **first** and must match your CUDA version:

```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install All Other Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers accelerate huggingface_hub safetensors pandas openpyxl numpy tqdm PyMuPDF sentence-transformers faiss-cpu gradio joblib scikit-learn xgboost
```

### 5. Download TinyLlama

```bash
python download_model.py
```

Downloads ~2 GB to `./models/TinyLlama/`. Only needed once.

### 6. Verify GPU

```bash
python check.py
```

Should print GPU memory numbers without errors.

---

## Running the Application

```bash
python app.py
```

Opens at **http://127.0.0.1:7860**. Upload a PDF (or select a preloaded one), enter a question and answer, and click **Analyze Answer**.

---

## Reproducing the Pipeline from Scratch

If you want to retrain the models from scratch instead of using the provided PKL files:

```bash
# Step 1: Extract hidden-state features (requires GPU + TinyLlama)
python build_dataset_final.py

# Step 2: Shuffle the dataset
python shuffle.py

# Step 3: Train classifiers with grid search
python train.py

# Step 4: (Optional) Run NLI verification on the dataset
python NLI_check.py

# Step 5: Launch the web app
python app.py
```

### End-to-End Data Flow

```
correct answers.xlsx  +  incorrect answers.xlsx  +  10 PDFs
                         |
                         v
              build_dataset_final.py
                         |
                         v
            features_correct_incorrect.xlsx
                         |
                         v
                    shuffle.py
                         |
                         v
            features_shuffled_final.xlsx
                         |
                         v
                     train.py
                         |
                  +------+------+
                  |      |      |
                  v      v      v
           scaler   pca   variance_threshold   svm/xgb model
           .pkl     .pkl       .pkl               .pkl
                  |      |      |
                  +------+------+
                         |
                         v
                      app.py  <-- also uses TinyLlama + MiniLM + DeBERTa + PDFs
                         |
                         v
                   Gradio Web UI
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch -- GPU inference and neural network computation |
| `transformers` | Hugging Face -- loads TinyLlama and DeBERTa models |
| `accelerate` | Enables `device_map="cuda"` for automatic GPU placement |
| `huggingface_hub` | Downloads TinyLlama from Hugging Face |
| `safetensors` | Safe model weight loading format |
| `pandas` | DataFrame operations for CSV/Excel data |
| `openpyxl` | Excel engine for pandas `.xlsx` support |
| `numpy` | Numerical array operations |
| `tqdm` | Progress bars during batch processing |
| `PyMuPDF` (`fitz`) | PDF text extraction |
| `sentence-transformers` | `all-MiniLM-L6-v2` sentence embeddings |
| `faiss-cpu` | FAISS vector similarity search |
| `gradio` | Web UI framework |
| `joblib` | Serialization for sklearn/xgboost models |
| `scikit-learn` | SVM, PCA, StandardScaler, VarianceThreshold, metrics |
| `xgboost` | XGBoost classifier |

---

## License

This project is for research and educational purposes. TinyLlama model weights are distributed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
