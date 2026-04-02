# HRES Backend - FastAPI

FastAPI backend for the Hallucination Risk Estimation System (HRES).

## Features

- **Dual-Pipeline Detection**:
  - **Whitebox (HRES)**: TinyLlama hidden state extraction → PCA → SVM/XGBoost classification
  - **Blackbox (NLI)**: FAISS retrieval → DeBERTa NLI verification
- RESTful API with automatic documentation (Swagger UI)
- Support for uploaded PDFs and preloaded documents
- CORS enabled for React frontend integration

## Setup

### 1. Install PyTorch with CUDA

**IMPORTANT**: Install PyTorch FIRST before installing other dependencies.

```bash
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Verify Model Files

Ensure these files exist in the `models/` directory (one level up from backend):

- `TinyLlama/` (model directory)
- `svm_model_final.pkl`
- `xgb_model_final.pkl`
- `scaler_final.pkl`
- `reduction_final.pkl`
- `variance_threshold_final.pkl`
- `reduction_metadata.csv`

### 4. Verify PDF Resources

Ensure PDFs are in the `resources/pdfs/` directory (one level up from backend).

## Running the Server

```bash
cd backend
python run.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### GET `/api/preloaded-pdfs`

List all preloaded PDFs from `resources/pdfs/`.

**Response**:

```json
[
  {
    "filename": "example.pdf",
    "display_name": "example"
  }
]
```

### POST `/api/predict`

Run hallucination detection on a PDF with question and answer.

**Request** (multipart/form-data):

- `file`: PDF file (optional if using preloaded_pdf)
- `preloaded_pdf`: Filename of preloaded PDF (optional if uploading file)
- `question`: Question about the document
- `answer`: Answer to verify

**Response**:

```json
{
  "success": true,
  "whitebox": {
    "SVM": {
      "prediction": 0,
      "label": "CORRECT",
      "confidence": 0.95,
      "prob_correct": 0.95,
      "prob_hallucinated": 0.05
    },
    "XGBoost": { ... }
  },
  "whitebox_error": null,
  "blackbox": {
    "verdict": "GROUNDED",
    "entailment": 0.89,
    "neutral": 0.08,
    "contradiction": 0.03,
    "retrieved_context": "..."
  },
  "blackbox_error": null
}
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

The server runs with auto-reload enabled. Any changes to Python files will automatically restart the server.

## Configuration

- **Port**: 8000 (default)
- **CORS Origins**: Configured for React dev server on ports 3000 and 5173
- **File Size Limit**: 50MB for uploaded PDFs
- **Model Loading**: Lazy loading on first request

## Troubleshooting

### Models not loading

- Verify all model files exist in `../models/` directory
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### PDF extraction fails

- Ensure PyMuPDF is installed: `pip install PyMuPDF`
- Verify PDF is not corrupted or password-protected

### Out of memory errors

- Models run on CUDA by default - ensure sufficient GPU memory
- Reduce batch size in detection service if needed
