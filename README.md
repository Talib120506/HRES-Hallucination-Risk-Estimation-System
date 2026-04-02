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

**Whitebox Approach** -- Opens up the LLM (Gemma) and inspects its **internal neural activations** (hidden-state vectors). When a model produces a hallucinated answer, its internal state differs from when it produces a grounded one. A downstream machine learning classifier—dynamically chosen through comprehensive cross-validation (typically Logistic Regression or SVM)—learns to distinguish these patterns.

**Blackbox Approach** -- Treats the LLM as a black box. Retrieves relevant text from the source PDF using semantic search (FAISS), then uses a Natural Language Inference model (DeBERTa) to check if the source **supports**, **contradicts**, or is **neutral** toward the answer.

Both results are combined into a single verdict: **CORRECT**, **HALLUCINATED**, or **UNCERTAIN**.

---

## How It Works

### Whitebox Pipeline (HRES)

```
PDF page text + Question + Answer
       |
       v
Gemma forward pass (no text generation, just processes the input)
       |
       v
Extract 2304-D hidden-state vector from the last transformer layer
at the last meaningful token position
       |
       v
VarianceThreshold  ->  StandardScaler  ->  PCA (dimensionality reduction)
       |
       v
Classifier (Logistic Regression, SVM, AdaBoost, etc. chosen via Grid Search)
       |
       v
Output: CORRECT or HALLUCINATED  (with confidence %)
```

**Why it works:** The 2304-dimensional vector from Gemma's final layer encodes a "fingerprint" of how the model internally processed the input. Hallucinated answers produce different activation patterns than correct ones. The classifier learns to tell them apart.

**What "whitebox" means:** You need access to the model's internal weights and hidden states -- you're looking *inside* the model.

### Blackbox Pipeline (NLI)

```
PDF document
       |
       v
Extract all pages  ->  Clean text  ->  Chunk into ~100-word segments
       |
       v
Embed all chunks with BAAI/bge-small-en-v1.5
       |
       v
Build FAISS index (cosine similarity search)
       |
       v
Retrieve top-3 or top-5 candidate chunks matching the question and answer
       |
       v
THRESHOLD CHECK: If maximum similarity < 0.35 -> Automatic HALLUCINATION
       |
       v
For each chunk passing threshold: run DeBERTa NLI directly on the full chunk
(Multi-step answers are split by newline and averaged across steps)
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

```
Hallucination test/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Main application & core scripts
│   ├── app.py                        # Gradio web interface (main entry point)
│   ├── chat.py                       # Interactive terminal chat with Gemma
│   ├── load_model.py                 # Shared Gemma model loader
│   ├── download_model.py             # Downloads Gemma from Hugging Face
│   ├── check.py                      # GPU CUDA verification
│   └── utils_internal/
│       ├── preview_input.py          # Preview formatted inputs (no GPU needed)
│       ├── tokenlog.py               # Chat with per-token probability logging
│       └── read.py                   # Early prototype (deprecated)
│
├── training/                         # Model training pipeline (run in order)
│   ├── build_dataset_final.py        # [STEP 1] Extract Gemma hidden-state features
│   ├── build_dataset_original.py     # Original version (deprecated)
│   ├── shuffle.py                    # [STEP 2] Shuffle feature dataset
│   ├── train.py                      # [STEP 3] Train SVM/XGBoost with grid search
│   ├── train_alt.py                  # Alternative training approach
│   └── NLI_check.py                  # [Independent] Run blackbox NLI verification
│
├── data/                             # Training data & results
│   ├── raw/
│   │   ├── TechManualQA_700.xlsx     # Extended Q&A dataset (700 pairs)
│   │   ├── TechManualQA_474 clean.xlsx  # Cleaned dataset (474 pairs)
│   │   ├── duplicates_removed_report.xlsx
│   │   └── cleanup.py
│   ├── training_data/
│   │   ├── correct answers.xlsx      # Ground-truth answers (label=0)
│   │   ├── Incorrect answers.xlsx    # Hallucinated answers (label=1)
│   │   ├── training_data_final.xlsx  # PCA-transformed features
│   │   └── hallucinated_generation.py  # Script to generate hallucinations
│   ├── processed/
│   │   ├── features_correct_incorrect.xlsx    # Raw features (2304-D vectors)
│   │   └── features_shuffled_final.xlsx       # Shuffled features
│   └── results/
│       ├── nli_results.csv           # NLI verification outputs
│       └── nli_results_baai.xlsx     # Alternative NLI results
│
├── models/                           # Trained ML models & embeddings
│   ├── gemma-2-2b-it/               # Gemma model weights (~2 GB, float16)
│   ├── svm_model_final.pkl          # Trained SVM classifier
│   ├── ada_model_final.pkl          # Alternative AdaBoost classifier
│   ├── lr_model_final.pkl           # Logistic regression model
│   ├── variance_threshold_final.pkl # Feature selector (removes low-variance)
│   ├── scaler_final.pkl             # StandardScaler for normalization
│   ├── reduction_final.pkl          # PCA dimensionality reducer
│   ├── reduction_metadata.csv       # PCA metadata
│   └── nli_index/                   # Cached FAISS indexes per document
│       ├── bosch_oven/
│       ├── dewalt_saw/
│       ├── dyson_v12/
│       ├── electrolux_oven/
│       ├── hilti_hammer/
│       └── ... (17 total PDFs indexed)
│
├── resources/                        # Source materials
│   └── pdfs/                        # 17 product manual PDFs
│       ├── apple_watch.pdf
│       ├── bosch_oven.pdf
│       ├── dji_mavic_pro.pdf
│       ├── dewalt_saw.pdf
│       ├── dyson_v12.pdf
│       ├── electrolux_oven.pdf
│       ├── ford_mach_e.pdf
│       ├── hilti_hammer.pdf
│       ├── laptop_lenovo_tc_x1.pdf
│       ├── lg_home_theater.pdf
│       ├── makita_drill.pdf
│       ├── nintendo_2ds_xl.pdf
│       ├── omron_monitor.pdf
│       ├── prusa_3d-printer.pdf
│       ├── samsung_phone_zfold.pdf
│       ├── spot_boston_dynamics.pdf
│       └── tesla_model_s.pdf
│
├── tests/                           # Testing infrastructure
│
└── env/                             # Python virtual environment (git-ignored)
```

---

## Complete File Reference

### Core Pipeline Scripts

#### `training/build_dataset_final.py` -- Feature Extraction

**What it does:** Extracts hidden-state features from Gemma for both correct and incorrect answers. This creates the training data for the whitebox classifiers.

**How it works:**
1. Reads `data/training_data/correct answers.xlsx` (label=0) and `data/training_data/Incorrect answers.xlsx` (label=1)
2. For each Q&A pair, extracts the relevant page from the source PDF using PyMuPDF
3. Builds input: `"{page_text}\n\nQuestion: {question}\nAnswer: {answer}"`
4. Runs a Gemma forward pass (no text generation)
5. Finds the last meaningful token (walks backwards, skipping EOS/punctuation)
6. Extracts the 2304-D hidden-state vector from the final transformer layer at that position
7. Saves all features to `data/processed/features_correct_incorrect.xlsx`

**Key functions:**
- `extract_page(pdf_path, page)` -- Extracts text from a specific PDF page (max 1600 chars)
- `find_last_meaningful_token(input_ids, tokenizer)` -- Locates the target token position
- `get_last_hidden_state(text, tokenizer, model)` -- Runs forward pass, returns 2304-D vector + metadata
- `process_answers(df, answer_type, label, ...)` -- Processes all rows for a given label
- `build_dataset(correct_xlsx, incorrect_xlsx, ...)` -- Main orchestrator

**CLI arguments:** `--correct`, `--incorrect`, `--pdfs_dir`, `--output`, `--model`

**Input:** `data/training_data/correct answers.xlsx`, `data/training_data/Incorrect answers.xlsx`, PDFs from `resources/pdfs/`, Gemma model
**Output:** `data/processed/features_correct_incorrect.xlsx` (each row = metadata + label + 2304 features + seq_len + target_index)

---

#### `training/shuffle.py` -- Dataset Shuffling

**What it does:** Randomly shuffles the rows of the feature dataset to prevent ordering bias during cross-validation.

```python
df = pd.read_excel("data/processed/features_correct_incorrect.xlsx")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_excel("data/processed/features_shuffled_final.xlsx", index=False)
```

**Input:** `data/processed/features_correct_incorrect.xlsx`
**Output:** `data/processed/features_shuffled_final.xlsx`

---

#### `training/train_unified.py` -- Unified Model Training (RECOMMENDED)

**What it does:** Comprehensive training script that tests multiple classifiers (SVM, XGBoost, AdaBoost, LogisticRegression, ExtraTrees, KNN) with both PCA and PCA+LDA dimensionality reduction. Automatically selects and saves only the single best-performing model.

**Training pipeline:**

1. **Load Data** -- Reads `data/processed/features_shuffled_final.xlsx`
2. **Preprocessing** -- VarianceThreshold (1e-6) → StandardScaler
3. **PCA Analysis** -- Analyzes variance retention at 80%, 85%, 90%, 95%, 99%
4. **Grid Search** -- Tests all combinations:
   - Reduction methods: PCA [150, 200, 250, 300 components] + PCA+LDA [150, 200, 250, 300 → 1]
   - Classifiers: SVM (8 configs), LogisticRegression (3), AdaBoost (6), ExtraTrees (4), KNN (5), XGBoost (2 if GPU available)
   - Total: ~200+ combinations with 5-fold cross-validation
5. **Select Winner** -- Chooses configuration with highest mean CV accuracy
6. **Detailed Evaluation** -- Reports precision, recall, F1, ROC-AUC across folds
7. **Train Final** -- Retrains winner on full dataset
8. **Save** -- Exports only the best model + metadata

**Output Files:**
- `models/best_model_final.pkl` - The winning classifier
- `models/scaler_final.pkl`, `models/reduction_final.pkl`, `models/variance_threshold_final.pkl`
- `models/reduction_metadata.csv` - Winner info (method, classifier type, accuracy, n_components, model_filename)
- `data/training_data/training_data_final.xlsx`

**Run:**
```bash
python training/train_unified.py
```

**Input:** `data/processed/features_shuffled_final.xlsx`
**Output:** Single best model + preprocessing pipeline in `models/`

---

#### `training/train.py` -- Original Training Script (ARCHIVED)

**Status:** Completely commented out. Original implementation trained SVM and XGBoost only. Superseded by `train_unified.py`.

---

#### `training/train_alt.py` -- Alternative Training Script (LEGACY)

**Status:** Legacy script that trained AdaBoost, ExtraTrees, and KNN. Functionality now merged into `train_unified.py`.

---

#### `training/NLI_check.py` -- Blackbox NLI Verification

**What it does:** Independently verifies whether each answer in the dataset is supported by its source PDF, using semantic retrieval (FAISS) and Natural Language Inference (DeBERTa).

**Pipeline:**
1. **PDF Processing** -- Extracts all pages with PyMuPDF; cleans text (removes page numbers, language headers, figure references); chunks into ~100-word overlapping segments (30-word overlap at sentence boundaries)
2. **Embedding** -- Encodes all chunks with `BAAI/bge-small-en-v1.5` (384-D vectors) on GPU
3. **FAISS Indexing** -- Builds one `IndexFlatIP` (cosine similarity) index per PDF document; caches indexes in `models/nli_index/` for reuse
4. **Retrieval** -- For each Q&A pair, retrieves candidates using three strategies:
   - Answer-based full-document search (top-5)
   - Question-based full-document search (top-5)
   - Page-filtered search using known page number (if available)
   - Results are merged and deduplicated
5. **NLI Scoring** -- For each candidate chunk, runs DeBERTa NLI directly on the entire chunk without sliding sentence windows. Multi-step answers are split by newline and evaluated per-step, with entailment probabilities averaged across all steps. Picks the highest overall entailment score.
6. **Verdict** -- Maps NLI label to: entailment=GROUNDED, neutral=UNCERTAIN, contradiction=HALLUCINATION

**Key class:** `FAISSDocIndex` -- Manages per-document FAISS indexes with `build()` and `retrieve()` methods.

**CLI arguments:** `--dataset`, `--pdfs_dir`, `--output`

**Input:** `data/raw/TechManualQA_700.xlsx` (or cleaned version), PDFs in `resources/pdfs/`
**Output:** `data/results/nli_results.csv`

---

### Web Application

#### `src/app.py` -- Gradio Web Interface

**What it does:** Provides a web UI that runs both pipelines on user-provided PDFs and displays combined results.

**User workflow:**
1. Upload a PDF or select a preloaded manual from the dropdown (17 manuals available)
2. Enter a question about the document
3. Enter an answer to verify
4. Click "Analyze Answer"
5. View the combined verdict, whitebox details, and blackbox details

**Backend pipeline:**

- **`whitebox_predict(pdf_path, question, answer)`**
  1. Extracts first 3 pages of PDF text (max 1600 chars combined)
  2. Runs Gemma forward pass, extracts 2304-D hidden-state vector
  3. Applies saved preprocessing: VarianceThreshold -> StandardScaler -> PCA (reduction)
  4. Classifies with SVM, AdaBoost, or Logistic Regression (only loads models whose feature dimensions match the PCA)
  5. Returns: CORRECT/HALLUCINATED with confidence and probability breakdown

- **`blackbox_predict(pdf_path, question, answer)`**
  1. Chunks all PDF pages, builds in-memory FAISS index (or uses cached index from `models/nli_index/`)
  2. Embeds the question and answer to retrieve the top-3 candidate chunks.
  3. **Threshold Check:** If the maximum FAISS similarity distance is below `0.35`, the system short-circuits and automatically flags the answer as a `HALLUCINATION` (since the semantic context is entirely missing from the source).
  4. **NLI Scoring:** Combines the query into a strict logical hypothesis (`"Question: {question} Answer: {answer}"`) and runs DeBERTa NLI directly on the full chunk.
  5. Returns: GROUNDED/UNCERTAIN/HALLUCINATION with probabilities and matching source text

- **`predict()`** -- Runs both pipelines, combines verdicts, formats results as styled HTML

**Model loading:** All models (Gemma from `models/gemma-2-2b-it/`, Embedder (BAAI), DeBERTa, SVM/Ada/LR classifiers, PCA reduction, scaler) are **lazy-loaded** on first use via singleton patterns (`get_llama()`, `get_embedder()`, `get_nli()`, `get_classifiers()`).

**Error handling:** Both pipelines are wrapped in try-except blocks. Failures display styled error messages instead of crashing the UI.

**UI features:**
- Gradient-styled header with pipeline badges
- Color-coded verdict cards (green/red/amber)
- Confidence progress bars and probability breakdowns
- Retrieved source chunk display with styled quote block
- Side-by-side pipeline comparison table
- Dark mode support
- "How does it work?" accordion with step-by-step comparison table
- Dropdown selector for 17 preloaded product manuals

**Run:**
```bash
python src/app.py
# Opens at http://127.0.0.1:7860
```

---

### Utility / Development Scripts

#### `src/download_model.py` -- Model Download

Downloads Gemma-2-2B-Chat from Hugging Face Hub to `models/gemma-2-2b-it/`. Run once before using any pipeline script.

```bash
python src/download_model.py
```

---

#### `src/check.py` -- GPU Check

Verifies CUDA is working by allocating a test tensor on the GPU and printing memory usage.

```bash
python src/check.py
```

---

#### `src/load_model.py` -- Shared Model Loader

Helper module (not run directly) that loads Gemma tokenizer + model in float16 on CUDA. Imported by `chat.py` and other scripts.

```python
from src.load_model import loadmodel
tokenizer, model = loadmodel()
```

**Note:** Model path should point to `models/gemma-2-2b-it/`. Update paths if your project structure differs.

---

#### `src/chat.py` -- Interactive Chat

Multi-turn terminal chat with Gemma. Maintains conversation history. Useful for manually testing model responses.

- Generation parameters: `max_new_tokens=256`, `temperature=0.7`, `top_p=0.9`
- Reports token count after each response
- Type `exit` to quit

```bash
python src/chat.py
```

---

#### `src/utils_internal/tokenlog.py` -- Token-Level Analysis

Extended chat that logs per-response introspection data:
- The target token (last meaningful token in the response)
- Its softmax probability (model confidence)
- The full 2304-D hidden-state vector at that position

Useful for understanding how hidden states differ between confident and uncertain responses.

```bash
python src/utils_internal/tokenlog.py
```

---

#### `src/utils_internal/preview_input.py` -- Input Preview (No GPU)

Prints the exact text that would be fed to Gemma for the first N answerable rows. **Does not load any model or use the GPU.** Useful for verifying PDF extraction and text concatenation.

```bash
python src/utils_internal/preview_input.py --n 10
```

---

#### `src/utils_internal/read.py` -- Early Prototype

First proof-of-concept for hidden-state extraction. Processes 5 rows from a CSV, extracts hidden states. Does NOT include PDF context -- only uses question + answer text. **Superseded by `training/build_dataset_final.py`.**

---

#### `training/build_dataset_original.py` -- Original Dataset Builder

Earlier version of `build_dataset_final.py`. Processes only correct answers (label=0) from the dataset, outputs CSV, includes resume support. Currently in test mode (3 rows). **Superseded by `training/build_dataset_final.py`.**

---

#### `training/train_alt.py` -- Alternative Training Script

Alternative training approach with different hyperparameter configurations or model types. Used for experimentation.

---

## Data Files

### Input Data (Excel)

| File | Rows | Description |
|------|------|-------------|
| `data/raw/TechManualQA_700.xlsx` | 700 | Extended master dataset with Q&A pairs from 17 product manuals |
| `data/raw/TechManualQA_474 clean.xlsx` | 474 | Cleaned and deduplicated version of the dataset |
| `data/training_data/correct answers.xlsx` | ~300 | Correct (grounded) answers with page references |
| `data/training_data/Incorrect answers.xlsx` | ~300 | Hallucinated/incorrect answers |

**Columns in `TechManualQA_700.xlsx`:**

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

**Columns in `data/training_data/correct answers.xlsx` / `Incorrect answers.xlsx`:**

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
| `data/processed/features_correct_incorrect.xlsx` | `training/build_dataset_final.py` | Raw features: metadata + label + 2304 hidden-state values + `seq_len` + `target_index` |
| `data/processed/features_shuffled_final.xlsx` | `training/shuffle.py` | Same as above, rows shuffled with `random_state=42` |
| `data/training_data/training_data_final.xlsx` | `training/train.py` | PCA-transformed features with metadata and labels |

**Feature columns in `data/processed/features_correct_incorrect.xlsx`:**

| Columns | Count | Description |
|---------|-------|-------------|
| `question_id`, `doc_id`, `question`, `answer`, `answer_type` | 5 | Metadata |
| `label` | 1 | 0 = correct, 1 = hallucinated |
| `v_0` through `v_2303` | 2304 | Hidden-state vector from Gemma's last layer |
| `seq_len` | 1 | Total number of tokens in the input |
| `target_index` | 1 | Token position of the last meaningful token |
| **Total** | **2312** | |

---

### Trained Model Artifacts (PKL)

| File | Contents | Saved by |
|------|----------|----------|
| `models/best_model_final.pkl` | Best performing classifier (SVM/AdaBoost/LR/XGB/ExtraTrees/KNN) | `training/train_unified.py` |
| `models/variance_threshold_final.pkl` | Fitted `VarianceThreshold` (removes near-zero-variance features) | `training/train_unified.py` |
| `models/scaler_final.pkl` | Fitted `StandardScaler` (zero mean, unit variance) | `training/train_unified.py` |
| `models/reduction_final.pkl` | Fitted `PCA` or `PCA+LDA` pipeline (dimensionality reduction) | `training/train_unified.py` |
| `models/reduction_metadata.csv` | Winner metadata (method, classifier_type, accuracy, n_components, model_filename) | `training/train_unified.py` |

**Legacy Model Files (backward compatibility):**
- `models/svm_model_final.pkl` - Trained SVM (if exists, app.py will use it as fallback)
- `models/ada_model_final.pkl` - Trained AdaBoost (legacy)
- `models/lr_model_final.pkl` - Trained Logistic Regression (legacy)

**Note:** `training/train_unified.py` saves only the single best model as `best_model_final.pkl`. The `src/app.py` dynamically loads whichever model won based on `reduction_metadata.csv`, or falls back to legacy model files for backward compatibility.

---

### Other Data Outputs

| File | Source | Description |
|------|--------|-------------|
| `data/results/nli_results.csv` | `training/NLI_check.py` | NLI verification results with verdict and probabilities |
| `data/results/nli_results_baai.xlsx` | `training/NLI_check.py` (alt) | Alternative NLI results using different embeddings |
| `data/training_data/hallucinated_generation.py` | Script | Generates hallucinated answers for training |

**Columns in `data/results/nli_results.csv`:**

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

The `resources/pdfs/` folder contains 17 real product manuals used as ground-truth sources:

| PDF File | Product |
|----------|---------|
| `apple_watch.pdf` | Apple Watch |
| `bosch_oven.pdf` | Bosch Oven |
| `dewalt_saw.pdf` | DeWalt Saw |
| `dji_mavic_pro.pdf` | DJI Mavic Pro Drone |
| `dyson_v12.pdf` | Dyson V12 Vacuum |
| `electrolux_oven.pdf` | Electrolux Oven |
| `ford_mach_e.pdf` | Ford Mustang Mach-E |
| `hilti_hammer.pdf` | Hilti Hammer Drill |
| `laptop_lenovo_tc_x1.pdf` | Lenovo ThinkCentre X1 |
| `lg_home_theater.pdf` | LG Home Theater System |
| `makita_drill.pdf` | Makita Drill |
| `nintendo_2ds_xl.pdf` | Nintendo 2DS XL |
| `omron_monitor.pdf` | Omron Blood Pressure Monitor |
| `prusa_3d-printer.pdf` | Prusa 3D Printer |
| `samsung_phone_zfold.pdf` | Samsung Galaxy Z Fold |
| `spot_boston_dynamics.pdf` | Boston Dynamics Spot Robot |
| `tesla_model_s.pdf` | Tesla Model S |

The `models/nli_index/` directory caches pre-built FAISS indexes for these PDFs (one subfolder per document containing `index.faiss` and `meta.pkl`). These are created by `training/NLI_check.py` on first run and reused on subsequent runs.

---

## Models Used

### Gemma-2-2B-Chat-v1.0 (Whitebox Feature Extraction)

| Property | Value |
|----------|-------|
| Architecture | LlamaForCausalLM |
| Parameters | 1.1 billion |
| Hidden size | 2304 (dimensionality of extracted vectors) |
| Layers | 22 transformer layers |
| Attention heads | 32 (with 4 KV heads via Grouped Query Attention) |
| Vocabulary | 32,000 tokens (SentencePiece BPE) |
| Context window | 2304 tokens |
| Precision | float16 (half-precision for reduced VRAM) |
| License | Apache 2.0 |

### BAAI/bge-small-en-v1.5 Models (Blackbox Chunk Embedding)

| Property | Value |
|----------|-------|
| Type | Sentence Transformer |
| Output dimension | 384 |
| Purpose | Encodes PDF text chunks for FAISS similarity search. BAAI models provide state-of-the-art dense retrieval. |

### cross-encoder/nli-deberta-v3-base (Blackbox NLI Classification)

| Property | Value |
|----------|-------|
| Type | Cross-Encoder (sequence classification) |
| Classes | 3: contradiction (0), entailment (1), neutral (2) |
| Purpose | Determines if source text supports or contradicts the answer |

### SVM / AdaBoost / Logistic Regression (Whitebox Classification)

| Property | SVM | AdaBoost | Logistic Regression |
|----------|-----|----------|---------------------|
| Kernel / Type | RBF | Ensemble boosting | Linear classifier |
| Input features | PCA-reduced hidden states | PCA-reduced hidden states | PCA-reduced hidden states |
| Output | CORRECT / HALLUCINATED | CORRECT / HALLUCINATED | CORRECT / HALLUCINATED |
| Probability output | Yes | Yes | Yes |

### Reasoning for Model Choices

**Why Whitebox (Hidden States)?**
Blackbox text-based methods rely on having a perfect ground-truth reference document and fail when an LLM is "confidently wrong." Whitebox inspection directly assesses the LLM's internal confidence and processing footprint. By analyzing varying activation patterns (2304-D feature space), we can detect uncertainty and fabrication at the source of generation before the text is even formed.

**Why Logistic Regression (with PCA) for the Whitebox classifier?**
Through a rigorous grid search of over 200 configurations, **Logistic Regression paired with PCA (150 components)** emerged as the strongest performer, achieving a peak cross-validation accuracy of **~68.36%** (with an F1 score of ~0.69). 
- **Linear Separability:** The fact that a straightforward linear model outperforms complex decision trees or neural networks indicates that Gemma's hidden states encode truthfulness in a mostly linearly separable manner.
- **Overfitting Resistance:** Given the modest dataset size (~950 samples), complex ensembles easily overfit. Logistic regression cleanly isolates the core mathematical signal of hallucination without memorizing noise in the 2304-dimensional space.

**Why Blackbox (BAAI + DeBERTa)?**
Because whitebox analysis alone can't strictly fact-check specific product specs, the Blackbox NLI pipeline acts as a semantic safety net. 
- Fast & Rich Embeddings: The **BAAI/bge-small-en-v1.5** model operates as our unified embedding solution. It leverages state-of-the-art dense retrieval capabilities to understand complex technical texts while remaining highly efficient. By unifying on BAAI entirely, the vector retrieval and offline cache pipelines are perfectly synchronized.
- High-Accuracy Verification: `cross-encoder/nli-deberta-v3-base` operates as the cross-encoder because its architecture yields state-of-the-art performance for Natural Language Inference (NLI), handling complex technical context and entailment reasoning well.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.10+ (developed on 3.12) |
| CUDA GPU | Required for Gemma, NLI, and embedding models |
| CUDA Toolkit | 12.x recommended (`nvcc --version` to check) |
| GPU VRAM | 2-3 GB minimum (Gemma at float16) |
| Disk Space | ~2 GB for Gemma weights + ~120 MB for PDFs |
| OS | Windows 10/11 (developed on), Linux, or macOS |

---

## Setup (Step-by-Step)

### 1. Navigate to Project Directory

```bash
cd "d:\Hallucination test"
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

### 5. Download Gemma

```bash
python src/download_model.py
```

Downloads ~2 GB to `models/gemma-2-2b-it/`. Only needed once.

### 6. Verify GPU

```bash
python src/check.py
```

Should print GPU memory numbers without errors.

---

## Running the Application

```bash
python src/app.py
```

Opens at **http://127.0.0.1:7860**. Upload a PDF (or select from 17 preloaded manuals), enter a question and answer, and click **Analyze Answer**.

---

## Reproducing the Pipeline from Scratch

If you want to retrain the models from scratch instead of using the provided PKL files:

```bash
# Step 1: Extract hidden-state features (requires GPU + Gemma)
python training/build_dataset_final.py

# Step 2: Shuffle the dataset
python training/shuffle.py

# Step 3: Train classifiers with comprehensive grid search (RECOMMENDED)
python training/train_unified.py
# Tests 200+ configurations, saves only the single best model

# Step 4: (Optional) Run NLI verification on the dataset
python training/NLI_check.py

# Step 5: Launch the web app
python src/app.py
```

### End-to-End Data Flow

```
data/training_data/
  correct answers.xlsx + Incorrect answers.xlsx
           +
resources/pdfs/ (17 PDFs)
           |
           v
training/build_dataset_final.py
           |
           v
data/processed/features_correct_incorrect.xlsx
           |
           v
training/shuffle.py
           |
           v
data/processed/features_shuffled_final.xlsx
           |
           v
training/train_unified.py
  ├─ Tests: SVM, XGBoost, AdaBoost, 
  │         LogisticRegression, ExtraTrees, KNN
  ├─ PCA vs PCA+LDA reduction
  └─ 200+ combinations with 5-fold CV
           |
           v
models/
  ├─ best_model_final.pkl     ← Winner saved here
  ├─ scaler_final.pkl
  ├─ reduction_final.pkl
  ├─ variance_threshold_final.pkl
  └─ reduction_metadata.csv   ← Stores which model won
           |
           v
src/app.py
  ├─ Loads best model dynamically from metadata
  ├─ Uses models/gemma-2-2b-it/
  ├─ Uses BAAI/bge-small-en-v1.5 embedder
  ├─ Uses DeBERTa NLI
  ├─ Reads resources/pdfs/
  └─ Caches models/nli_index/
           |
           v
     Gradio Web UI
  (http://127.0.0.1:7860)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch -- GPU inference and neural network computation |
| `transformers` | Hugging Face -- loads Gemma and DeBERTa models |
| `accelerate` | Enables `device_map="cuda"` for automatic GPU placement |
| `huggingface_hub` | Downloads Gemma from Hugging Face |
| `safetensors` | Safe model weight loading format |
| `pandas` | DataFrame operations for CSV/Excel data |
| `openpyxl` | Excel engine for pandas `.xlsx` support |
| `numpy` | Numerical array operations |
| `tqdm` | Progress bars during batch processing |
| `PyMuPDF` (`fitz`) | PDF text extraction |
| `sentence-transformers` | `BAAI/bge-small-en-v1.5` sentence embeddings |
| `faiss-cpu` | FAISS vector similarity search |
| `gradio` | Web UI framework |
| `joblib` | Serialization for sklearn/xgboost models |
| `scikit-learn` | SVM, PCA, StandardScaler, VarianceThreshold, metrics |
| `xgboost` | XGBoost classifier |

---

## License

This project is for research and educational purposes. Gemma model weights are distributed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
