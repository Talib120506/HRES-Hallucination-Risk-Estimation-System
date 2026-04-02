# 🚀 Quick Start Guide - HRES Full-Stack Application

## Prerequisites

Before running the application, ensure you have:

1. **Python 3.8+** installed
2. **Node.js 16+** and **npm** installed
3. **CUDA-capable GPU** (recommended for model inference)
4. **PyTorch with CUDA** installed (see below)

## Installation Steps

### Step 1: Install PyTorch with CUDA (Required First!)

```bash
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Automated Startup (Easiest Method)

Simply double-click the `start.bat` file in the root directory. This will:

- Install all backend dependencies
- Install all frontend dependencies
- Start both servers automatically

```bash
# Or run from command line:
start.bat
```

### Step 3: Manual Startup (Alternative)

If you prefer to start servers separately:

**Terminal 1 - Backend:**

```bash
start_backend.bat
```

**Terminal 2 - Frontend:**

```bash
start_frontend.bat
```

## Access the Application

Once both servers are running:

- **Frontend (Main App)**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Using the Application

### Landing Page (/)

- View features and how HRES works
- Click "Try It Now" to go to the main application

### Main Application (/app)

#### Option 1: Use Preloaded PDFs

1. Select "Preloaded Documents" tab
2. Choose a PDF from the dropdown
3. Enter a question about the document
4. Enter an answer to verify
5. Click "Analyze Answer"

#### Option 2: Upload Your Own PDF

1. Select "Upload PDF" tab
2. Drag & drop a PDF file (or click to browse)
3. Enter a question about the document
4. Enter an answer to verify
5. Click "Analyze Answer"

### Understanding Results

The system will show three sections:

1. **🎯 Combined Verdict**
   - Overall decision based on both pipelines
   - High confidence if both agree
   - Uncertain if pipelines disagree

2. **🔬 Whitebox Details (HRES)**
   - SVM Classifier results
   - XGBoost Classifier results
   - Confidence scores and probabilities

3. **🔍 Blackbox Details (NLI)**
   - NLI Verdict (GROUNDED/UNCERTAIN/HALLUCINATION)
   - Entailment, Neutral, Contradiction scores
   - Retrieved source text from PDF

## Troubleshooting

### Backend won't start

- Ensure all model files exist in `models/` directory:
  - `TinyLlama/` (model directory)
  - `svm_model_final.pkl`
  - `xgb_model_final.pkl`
  - `scaler_final.pkl`
  - `reduction_final.pkl`
  - `variance_threshold_final.pkl`
  - `reduction_metadata.csv`

### Frontend won't start

- Check if Node.js is installed: `node --version`
- Check if npm is installed: `npm --version`
- Try manual install: `cd frontend && npm install`

### "Cannot connect to server" error

- Ensure backend is running on port 8000
- Check if another application is using port 8000
- Look for errors in the backend terminal

### Models not loading / CUDA errors

- Verify PyTorch is installed with CUDA:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should return True
  ```
- Check GPU memory: Models require ~8GB VRAM
- Ensure CUDA drivers are up to date

### PDF not processing

- Ensure PDF is not password-protected
- Check PDF file size (must be under 50MB)
- Verify PDF is not corrupted

## Project Structure

```
HRES/
├── start.bat              # One-click startup script
├── start_backend.bat      # Start backend only
├── start_frontend.bat     # Start frontend only
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── api/routes.py # API endpoints
│   │   ├── services/     # Detection logic
│   │   └── main.py       # FastAPI app
│   └── run.py            # Server launcher
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # UI components
│   │   ├── pages/        # Page components
│   │   └── services/     # API client
│   └── package.json
├── models/               # ML model files
└── resources/pdfs/       # Preloaded PDFs
```

## Development Mode

### Backend Development

```bash
cd backend
python run.py  # Auto-reload enabled
```

Changes to Python files will automatically restart the server.

### Frontend Development

```bash
cd frontend
npm run dev  # Hot Module Replacement enabled
```

Changes to React components update instantly without page reload.

## Building for Production

### Frontend

```bash
cd frontend
npm run build
npm run preview
```

This creates an optimized production build in `frontend/dist/`.

## Need Help?

- Check `FULLSTACK_ARCHITECTURE.md` for detailed architecture info
- Check `backend/README.md` for backend-specific documentation
- Check `frontend/README.md` for frontend-specific documentation
- Check `IMPLEMENTATION_COMPLETE.md` for full feature checklist

## Example Workflows

### Testing with Preloaded PDFs

1. Start the application
2. Go to http://localhost:5173/app
3. Select a preloaded PDF
4. Try the example cards at the bottom of the page
5. View results in the right panel

### Testing with Your Own PDF

1. Start the application
2. Go to http://localhost:5173/app
3. Switch to "Upload PDF" tab
4. Drag and drop your PDF
5. Enter a question and answer
6. Analyze and view results

## Performance Notes

- **First Request**: Takes 30-60 seconds (models loading)
- **Subsequent Requests**: ~15-30 seconds per analysis
- **Memory Usage**: ~8GB GPU VRAM, ~4GB system RAM
- **Concurrent Users**: Single-threaded by default

## License

See root README.md for license information.
