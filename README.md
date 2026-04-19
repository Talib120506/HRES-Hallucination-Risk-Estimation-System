# HRES: Hallucination Risk Estimation System
*(Dual-Pipeline LLM Hallucination Detection)*

A comprehensive, state-of-the-art dual-pipeline system for detecting Large Language Model (LLM) hallucinations by uniquely combining **Whitebox** (internal hidden-state analysis) and **Blackbox** (Natural Language Inference text verification) methodologies. 

The project includes an interactive **React + FastAPI** full-stack web application, a robust model training pipeline, and a legacy Gradio interface for seamless fact-checking and model inspection.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Dual-Pipeline Architecture](#dual-pipeline-architecture)
   - [Whitebox Pipeline (HRES)](#whitebox-pipeline-hres)
   - [Blackbox Pipeline (NLI)](#blackbox-pipeline-nli)
   - [Combined Verdict Logic](#combined-verdict-logic)
3. [Technology Stack](#technology-stack)
4. [Project Structure & File Reference](#project-structure--file-reference)
5. [Data & Models Reference](#data--models-reference)
   - [Models Used](#models-used)
   - [Source PDFs](#source-pdfs)
6. [Prerequisites](#prerequisites)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
   - [Full-Stack Interfaces](#full-stack-interfaces)
   - [Legacy Gradio App](#legacy-gradio-app)
9. [Reproducing the ML Pipeline](#reproducing-the-ml-pipeline)
10. [REST API Documentation](#rest-api-documentation)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## 🎯 Project Overview

Large Language Models confidently generate answers that sound correct but are factually fabricated—a phenomenon known as **hallucination**. This project addresses the hallucination problem in Retrieval-Augmented Generation (RAG) and standard Q&A by aggressively cross-verifying outputs via two opposing paradigms:
1. **Inspecting the Model's Brain (Whitebox):** Capturing the actual neural uncertainty hidden in the final transformer layers before text is even generated.
2. **Fact-Checking the Output (Blackbox):** Treating the text as an empirical claim and running strict, semantic Natural Language Inference against verified source material.

By aggregating these techniques into a single, cohesive full-stack infrastructure, the **HRES** framework maximizes detection accuracy and minimizes false positives.

---

## 🧠 Dual-Pipeline Architecture: Deep Dive & Nuances

```text
PDF Source + Question + Answer

      [ WHITEBOX PIPELINE ]                          [ BLACKBOX PIPELINE ]
               │                                               │
     LLM Forward Pass (No Gen)                   Hybrid Retrieval (FAISS + BM25)
               │                                               │
   Extract Hidden-State Vector                   Cross-Encoder Re-ranking Top-5
               │                                               │
     PCA Dimensionality Reduc.                   Question-Aware Hypothesis Form.
               │                                               │
    ML Classifier (Logistic Reg.)                Domain-Tuned Strict DeBERTa NLI 
               │                                               │
          [ CORRECT /                              [ GROUNDED / UNCERTAIN / 
          HALLUCINATED ]                               HALLUCINATION ] 
               │                                               │
               └──────────────────────┬────────────────────────┘
                                      ▼
                            🎯 COMBINED VERDICT
```

### 1. Whitebox Pipeline: Internal Model State Analysis
Instead of judging the LLM purely on text output, this pipeline looks at the mathematical uncertainty in the neural network's very architecture (e.g., TinyLlama / Gemma-2).

**Techniques & Implementation Nuances:**
- **Capture Point:** We intercept the high-dimensional hidden-state vector (e.g., 2304-D) exactly at the last meaningful mathematical representation token of the final transformer block before text probability generation.
- **Variance Filtering & PCA:** Raw high-dimensional vectors contain excessive noise. We pipe the output through `VarianceThreshold` and scale it prior to passing through **Principal Component Analysis (PCA) configured at 150 components**. This radically distills the essential signals of "uncertainty" and "confidence".
- **Final Classification & Results:** The 150-D reduced mathematical footprint evaluates against an optimized **Logistic Regression layer**, achieving an isolated validation accuracy of **~68.4%**. While moderately accurate independently, it serves perfectly as an internal sanity check for the robust Blackbox.

### 2. Blackbox Pipeline: Advanced Retrieval & NLI Fact-checking
Evaluating standard text generation requires rigorous source anchoring. This pipeline treats the answer as an empirical claim against verified context using a heavily upgraded Natural Language Inference (NLI) matrix.

**Techniques & Implementation Nuances:**
- **Hybrid Source Retrieval:** Relying purely on Cosine Similarity was insufficient for technical datasets. We augmented dense embeddings (`BAAI/bge-small-en-v1.5` mapped computationally into a FAISS index) tightly overlapping with sparse **BM25 lexical retrieval**.
- **Context Re-ranking:** To resolve extraction failures in uniquely structured technical manuals (e.g., `tesla_model_s.pdf`), the system over-fetches 10 candidate chunks and scores them through a dedicated Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to serve the NLI model the definitive Top 5 contexts.
- **Question-Aware Hypothesis Engine:** Instead of weakly testing if the "Context" entails the "Answer", we computationally concatenate and build an aggregated unified hypothesis (`Question + Answer`) mitigating critical false-positive entailment errors.
- **Custom Fine-Tuned NLI Architecture:** Baseline `cross-encoder/nli-deberta-v3-base` fell victim to lexical spoofing (an 8.5% False Negative rate where it believed highly-repetitive hallucinated jargon). We successfully resolved this via targeted domain-specific **Contrastive Fine-Tuning** (`models/nli_finetuned/best`) which directly teaches the DeBERTa model the difference between nuanced vocabulary and absolute semantic contradictions.
- **Dynamic Thresholding:** Entailment and similarity barriers (`SIMILARITY_THRESHOLD`) underwent explicit tuning loops utilizing Youden's J matrix indexing to strictly identify the line between semantic `UNCERTAIN` outputs versus confident predictions.

### 3. Integrated System Impact & Results
The combined model infrastructure resolves high ambiguity, ultimately pushing overall system reliability strictly over an **87.4% baseline accuracy**, crushing False Positives to mere fractional margins (`< 4.0%`). By consolidating varying confidence intervals, this system radically outperforms standard isolated extraction systems.

### Combined Verdict Logic

| Whitebox says   | Blackbox says     | Combined Verdict |
|-----------------|-------------------|------------------|
| CORRECT         | GROUNDED          | **CORRECT**      |
| HALLUCINATED    | HALLUCINATION     | **HALLUCINATED** |
| HALLUCINATED    | GROUNDED/NEUTRAL  | **UNCERTAIN**    |
| CORRECT         | HALLUCINATION     | **UNCERTAIN**    |
| Mixed           | Mixed             | **UNCERTAIN**    |

---

## 💻 Technology Stack

### Backend Application
- **Python Framework:** FastAPI & Uvicorn (RESTful Architecture)
- **Deep Learning:** PyTorch, Transformers (Hugging Face), Accelerate, Safetensors
- **Information Retrieval:** Sentence-Transformers, FAISS 
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Data Engineering:** Pandas, NumPy, PyMuPDF (Fitz), Openpyxl

### Frontend Application
- **Framework:** React 18 & Vite
- **Styling:** Tailwind CSS, PostCSS
- **State/Routing:** React Router DOM
- **Network/Files:** Axios, React-Dropzone

---

## 📁 Project Structure & File Reference

```text
HRES/
├── backend/                       # 🚀 FastAPI Backend Microservice
│   ├── app/
│   │   ├── api/routes.py          # REST endpoints (/predict, /preloaded)
│   │   ├── services/
│   │   │   ├── detection.py       # Whitebox & Blackbox prediction logic
│   │   │   └── model_loader.py    # Singleton lazy-loading logic for all LLMs
│   │   ├── utils/pdf_utils.py     # PDF chunking/cleaning utilities
│   │   └── main.py                # Server instantiation & CORS config
│   ├── requirements.txt           # Backend-specific dependencies
│   └── run.py                     # Entry point (Port 8000)
│
├── frontend/                      # ⚛️ React UI Interface
│   ├── src/
│   │   ├── components/            # Isolated UI elements (TabContainers, Landing)
│   │   ├── pages/                 # Full Page layouts (AppPage.jsx, LandingPage.jsx)
│   │   ├── services/api.js        # Axios wrapper hitting Backend API
│   │   ├── utils/colors.js        # Conditional UI styling
│   │   ├── App.jsx & main.jsx     # Router mapping & DOM render
│   ├── tailwind.config.js         # Theme variables
│   └── vite.config.js             # Vite builder port proxying
│
├── training/                      # 🏋️ ML Data & Pipeline Training Scripts
│   ├── build_dataset_final.py     # [STEP 1] Extracts LLM hidden states & tokens
│   ├── shuffle.py                 # [STEP 2] Randomizes extracted dataset
│   ├── train_unified.py           # [STEP 3] Comprehensive Grid Search (SVM/XGB/LR/KNN)
│   └── NLI_check.py               # [INDEP] Batch verifies dataset with FAISS+DeBERTa
│
├── Architecture_Design&Working/   # 📝 Extrapolated Technical Design Specs
│
├── src/                           # 🦖 Legacy / Core Utility Scripts
│   ├── app.py                     # Legacy Gradio Web Interface
│   ├── chat.py                    # Interactive terminal chat with local LLM
│   ├── download_model.py          # Huggingface weight downloader
│   ├── check.py                   # PyTorch/CUDA verification script
│   └── utils_internal/            # Low-level token prob/state loggers
│
├── data/                          # 📊 Raw and Processed Datasets
│   ├── raw/                       # Original Q&A and PDF annotations
│   ├── training_data/             # Human-annotated correct/hallucinated data
│   ├── processed/                 # Extracted multidimensional feature Excel matrices
│   └── results/                   # Final evaluation metrics
│       └── nli_audit/             # Exhaustive Blackbox metric reports & visual graphs
│
├── models/                        # 🧠 Stored Pre-Trained Weights
│   ├── TinyLlama/ & gemma.../     # Local Large Language Model weights
│   ├── nli_index/                 # Cached FAISS indexes per PDF
│   ├── best_model_final.pkl       # Best performing downstream classifier
│   └── scaler/reduction/var.pkl   # Dimensionality transformation state chains
│
├── resources/pdfs/                # 📚 17 ground-truth testing instruction manuals
├── requirements.txt               # Root project-wide Python deps
└── README.md                      # Documentation (You are here)
```

---

## 🗄️ Data & Models Reference

### Models Used

| Model / Strategy | Role / Methodology | Scale / Size |
|------------------|--------------------|--------------|
| **TinyLlama / Gemma-2** | Base Causal LLM. Generates the initial hidden states representing neural processing. | ~1.1B - 2B Params, float16 |
| **BAAI/bge-small-en-v1.5** | Dense Retriever. Maps raw text to dense vectors for contextual cosine search. | 384-D Vector Output |
| **FAISS (IndexFlatIP)**    | Vector DB. Sub-millisecond KNN search against cached PDF embeddings. | C-binding Engine |
| **nli-deberta-v3-base**    | Cross-Encoder. Direct comparison of `[Premise]` vs `[Hypothesis]` for rigorous entailment scoring. | 3 labels (Ent/Neu/Con) |
| **Logistic Regression**    | Best-performing whitebox classifier after extensive grid search across 200+ configs. | PCA reduced to ~150D |

### Source PDFs
The system provides 17 out-of-the-box pre-indexed manuals (located in `resources/pdfs/`) for rapid simulation tests without waiting for FAISS encoding:
*Apple Watch, Bosch Oven, DeWalt Saw, Dyson V12, Tesla Model S, Boston Dynamics Spot, DJ Mavic Pro, Samsung Galaxy Fold... among others.*

---

### 🔬 Reasoning for Model Choices

#### Whitebox Methodology (Trial, Error, and Final Selection)

**What we tried:** To classify the high-dimensional (2304-D) hidden-state representations from the Gemma/TinyLlama transformer layers, we ran an exhaustive grid search using 5-fold cross-validation. Models tested included:
- **Support Vector Machines (SVM)** (RBF and Linear kernels)
- **XGBoost & AdaBoost** gradient boosting ensembles
- **ExtraTrees / Random Forests**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

We also iterated on dimensionality reduction strategies, testing raw attributes against **PCA** (150, 200, 250, 300 components) and a combined **PCA + LDA** pipeline. 

**Why we finally landed on Logistic Regression (using 150 PCA components):**
1. **Strong Linear Separability:** We discovered that LLMs inherently encode confidence and fabrication logically in their final hidden states. A simple linear hyperplane separates grounded logic from hallucination surprisingly well.
2. **Overfitting Resistance:** Our dataset consists of a modest ~950 annotated samples. Complex non-linear models like XGBoost and RBF SVMs achieved high training accuracies but memorized the noise, leading to massive overfitting. 
3. **The Final Strategy:** Logistic Regression, paired cleanly with 150 PCA components, yielded the highest out-of-sample validation accuracy without memorizing 2304-D noise.


#### Blackbox Methodology (Trial, Error, and Final Selection)

**What we tried:** The blackbox architecture evaluates textual logic, which meant finding the ideal Retriever and the ideal NLI classifier. 
- *Embedder testing:* We evaluated standard baseline models like `sentence-transformers/all-MiniLM-L6-v2` against newer state-of-the-art models like `BAAI/bge-small-en-v1.5`.
- *Retrieval strategies:* Pure dense retrieval (FAISS) vs **Hybrid Retrieval** (dense FAISS embeddings + BM25 lexical keyword matching) combined with cross-encoder re-ranking.
- *NLI Models:* Out-of-the-box zero-shot `cross-encoder/nli-deberta-v3-base` vs a fine-tuned, domain-specific iteration.

**Why we finally landed on the current stack:**
1. **Embedder (BAAI/bge-small-en-v1.5):** BAAI dominated the raw semantic similarity tests relative to its compact size, retrieving far better context chunks out of the gate compared to older `MiniLM` variants.
2. **Hybrid Retrieval Pipeline & Re-ranker:** Dense-only FAISS embeddings sometimes missed exact alphanumeric technical specs (e.g., "0.5mm clearance"). We introduced a hybrid strategy mixing BM25 keyword search with FAISS dense retrieval, then running the top ~10 results through `cross-encoder/ms-marco-MiniLM-L-6-v2` as an intermediary precision re-ranker to guarantee the best 5 chunks hit the NLI checks.
3. **Fine-Tuned DeBERTa-v3:** The pre-trained zero-shot DeBERTa heavily suffered from *domain shift*—it didn't understand technical product manuals and frequently threw "UNCERTAIN" predictions when confused by jargon. By running the `training/finetune_nli.py` pipeline, we successfully fine-tuned the base model (`models/nli_finetuned/best`) directly on our QA pairs. The fine-tuned architecture decisively recognizes strict semantic contradictions and correct groundings on technical text, effectively fixing the uncertainty leak.

---

## ⚙️ Prerequisites

To run this full-stack system, your environment needs:
- Python 3.8+ (Developed heavily on 3.12)
- Node.js 16+ & NPM (For Vite/React)
- **CUDA-Capable GPU** with VRAM >= 4GB (8GB Recommended for float16 inference)
- CUDA Toolkit (12.x+ Recommended)

---

## 🚀 Installation & Setup

### 1. Repository Setup
```bash
git clone <repository_url>
cd "Hallucination test"
```

### 2. Python Environment & PyTorch Setup (CRITICAL)
Create an isolated python ecosystem:
```bash
python -m venv env
# Activate it
.\env\Scriptsctivate      # Windows (CMD)
source env/Scripts/activate # Git Bash / Unix
```
Install PyTorch **specifically for your driver's CUDA version:**
```bash
# Example for CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install All Dependencies
Install the backend AI & server framework logic:
```bash
# Global deps
pip install -r requirements.txt 
# Backend API specific deps
cd backend && pip install -r requirements.txt
```
Install Frontend dependencies:
```bash
cd ../frontend
npm install
cd ..
```

### 4. Fetch the Local LLM & Verify CUDA
```bash
python src/download_model.py
python src/check.py  # Should confirm PyTorch detects physical hardware GPU
```

---

## 🌟 Running the Application

### Option A: The Full-Stack Interfaces (Recommended)

Start the FastAPI Python backend taking advantage of optimized routes and lazy-loading:
```bash
# Terminal 1
cd backend
python run.py
```
*API running silently on http://localhost:8000*

Start the React Frontend:
```bash
# Terminal 2
cd frontend
npm run dev
```
*App immediately available on http://localhost:5173*

**Workflow:**
1. Navigate browser to `http://localhost:5173`
2. Look at the styled landing page, click **Try It Now**.
3. Select a Preloaded Document (e.g. *Tesla Model S*) or select **Upload PDF**.
4. Enter an answer + question metric to test logic. Keep your eye on the generated Verdicts card logic for accurate, real-time whitebox/blackbox feedback!

### Option B: Automated Scripts (Windows Only)
If utilizing Windows, you may run the batch setup scripts in the root directory:
```cmd
start.bat           # Boots up simultaneous servers automatically
# OR
start_backend.bat   # Starts only the FastAPI server
start_frontend.bat  # Starts only the Vite React dev server
```

### Option C: Legacy Gradio App
For quick CLI tests isolated from the React UX:
```bash
python src/app.py 
# Opens at http://127.0.0.1:7860
```

---

## 🔁 Reproducing the ML Pipeline
Want to train your own hidden state logic or refine the Machine Learning models?
From the root directory with the virtual environment activated:

1. **Extract Feature Matrices** (Requires GPU)
   ```bash
   python training/build_dataset_final.py
   ```
2. **Prevent Ordering Bias**
   ```bash
   python training/shuffle.py
   ```
3. **Comprehensive ML Configuration Evaluation** 
   ```bash
   python training/train_unified.py
   ```
   *Note: This script automatically tests hyperparameter configurations (Logistic Regression, PCA variance steps, LDA steps) using 5-fold cross-validation. The absolute best pipeline is cached directly into `models/best_model_final.pkl`.*

4. **NLI Auditing System & Graph Generation**
   Evaluates system accuracy on test populations and outputs strict visual metrics (ROC Curves, Confusion Matrices, Threshold sweeps) heavily populated inside `data/results/nli_audit/graphs/`.
   ```bash
   python training/nli_audit_pipeline.py
   ```

5. **Fine-tuning the Blackbox NLI Classifier**
   Addresses localized jargon mapping failures and prevents false "UNCERTAIN" tags via task domain-shift alignment. Finalized weights are natively cached into `models/nli_finetuned/best`.
   ```bash
   python training/finetune_nli.py
   ```

---

## 📡 REST API Documentation

FastAPI guarantees complete OpenAPI compliant swagger documentation.
**Once the backend is running, visit:** `http://localhost:8000/docs`

Key Endpoints:
- `GET /api/preloaded-pdfs` → Retrieves available PDF context catalogs.
- `POST /api/predict` → Expects `multipart/form-data`. Triggers parallel async extraction pipelines for Hidden states and DeBERTa inferencing. 
  - **Response Format:** JSON payload mapping `"whitebox": { ... }` probability confidences and `"blackbox": { ... }` specific string entailments.

---

## 🚧 Troubleshooting

- **Models won't load or OOM (Out Of Memory) Crashes:** 
  You're likely exceeding available VRAM. Ensure no other applications are hogging the GPU. You can edit the Hugging Face `device_map='auto'` or precision (`torch.float16`) to downgrade weights further if absolutely necessary.
- **FastAPI / React CORS Errors:**
  Ensure the backend is strictly running on `localhost:8000` and Vite is on `5173`. Make sure the frontend `.env` points correctly to the API URL.
- **CUDA Not Found Error during training:**
  Your PyTorch installation defaulted to CPU context. Uninstall `torch` and strictly execute the CUDA index-url PIP command referencing your exact Nvidia driver schema.

---

## 📄 License
This project's infrastructure and unique logic configurations are intended for research, auditing, and educational LLM interpretability purposes. 

**Model licenses:**
- Model weight usage is bound by HuggingFace restrictions. Specifically, Google's `Gemma` models enforce the [Apache 2.0 / Gemma License](https://huggingface.co/google/gemma-2-2b-it) policies. 
- BAAI and DeBERTa weights conform to their specific open-source Hugging Face organizational licensing.
