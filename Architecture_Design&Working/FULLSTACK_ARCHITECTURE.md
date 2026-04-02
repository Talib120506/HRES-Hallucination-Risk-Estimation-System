# Full-Stack Architecture: React + FastAPI

The project now includes a modern full-stack implementation with:

- **Backend**: FastAPI (Python) serving RESTful API endpoints
- **Frontend**: React + Vite with Tailwind CSS for a responsive UI

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│                  React + Vite + Tailwind                     │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Landing   │  │   Main App  │  │  Components │         │
│  │    Page    │  │    Page     │  │   Library   │         │
│  └────────────┘  └─────────────┘  └─────────────┘         │
│         │               │                  │                │
│         └───────────────┴──────────────────┘                │
│                         │                                    │
│                    HTTP/REST API                            │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                         │                                    │
│                    FastAPI Backend                          │
│                                                              │
│  ┌──────────────────────┴─────────────────────┐            │
│  │          API Routes Layer                  │            │
│  │  /api/predict  |  /api/preloaded-pdfs      │            │
│  └──────────────────┬─────────────────────────┘            │
│                     │                                       │
│  ┌──────────────────┴─────────────────────────┐            │
│  │         Detection Services                  │            │
│  │  whitebox_predict()  |  blackbox_predict()  │            │
│  └──────────────────┬─────────────────────────┘            │
│                     │                                       │
│  ┌──────────────────┴─────────────────────────┐            │
│  │          Model Loaders                      │            │
│  │  TinyLlama | Embedder | NLI | Classifiers  │            │
│  └──────────────────┬─────────────────────────┘            │
│                     │                                       │
└─────────────────────┼────────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
   ┌────▼────┐              ┌────────▼────────┐
   │ Models  │              │  PDF Resources  │
   │ (PyTorch│              │  (documents/)   │
   │  .pkl)  │              └─────────────────┘
   └─────────┘
```

## Directory Structure

```
HRES-Hallucination-Risk-Estimation-System/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py       # API endpoints
│   │   ├── services/
│   │   │   ├── detection.py    # Whitebox & Blackbox pipelines
│   │   │   └── model_loader.py # Lazy model loading
│   │   ├── utils/
│   │   │   └── pdf_utils.py    # PDF processing utilities
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI app initialization
│   ├── requirements.txt
│   ├── run.py                  # Server launcher
│   └── README.md
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── landing/        # Landing page components
│   │   │   │   ├── HeroSection.jsx
│   │   │   │   ├── AboutSection.jsx
│   │   │   │   ├── FeaturesSection.jsx
│   │   │   │   ├── HowItWorksSection.jsx
│   │   │   │   └── CTASection.jsx
│   │   │   ├── app/            # Main app components
│   │   │   │   ├── Header.jsx
│   │   │   │   ├── TabContainer.jsx
│   │   │   │   ├── PreloadedTab.jsx
│   │   │   │   ├── UploadTab.jsx
│   │   │   │   ├── VerdictCard.jsx
│   │   │   │   └── ResultsPanel.jsx
│   │   │   └── common/         # Shared components
│   │   │       └── LoadingSpinner.jsx
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx # Home page
│   │   │   └── AppPage.jsx     # Main application
│   │   ├── services/
│   │   │   └── api.js          # Axios API client
│   │   ├── utils/
│   │   │   └── colors.js       # Verdict styling utilities
│   │   ├── App.jsx             # React Router setup
│   │   ├── main.jsx            # Entry point
│   │   └── index.css           # Global styles + Tailwind
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env
│   └── README.md
│
├── src/                        # Original Gradio app (legacy)
│   └── app.py
├── models/                     # ML model files
├── resources/                  # PDF documents
├── data/                       # Training data
└── README.md                   # This file
```

## Quick Start

### 1. Install PyTorch (Required First!)

```bash
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python run.py
```

Backend will run on **http://localhost:8000**

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will run on **http://localhost:5173**

### 4. Open the Application

Navigate to **http://localhost:5173** in your browser.

## Features

### Frontend Features

- 🎨 **Beautiful Landing Page**: Hero section, features showcase, how it works explanation
- 📱 **Responsive Design**: Mobile-first, works on all screen sizes
- 🎯 **Interactive UI**: Tab-based interface with drag-and-drop file upload
- 📊 **Rich Results**: Color-coded verdict cards with expandable sections
- ⚡ **Real-time Validation**: Instant form validation feedback
- 💡 **Example Cards**: Pre-filled examples for quick testing

### Backend Features

- 🚀 **RESTful API**: Clean, documented endpoints (Swagger UI at /docs)
- 🔄 **Dual Pipelines**: Both whitebox (HRES) and blackbox (NLI) detection
- 📄 **PDF Support**: Upload files or use preloaded documents
- 🎯 **CORS Enabled**: Ready for React frontend integration
- 💾 **Lazy Loading**: Models load on first request for faster startup
- 🔒 **File Validation**: 50MB limit, PDF-only uploads

## API Endpoints

### GET `/api/preloaded-pdfs`

Returns list of available PDFs from `resources/pdfs/`.

**Response:**

```json
[
  {
    "filename": "example.pdf",
    "display_name": "example"
  }
]
```

### POST `/api/predict`

Analyzes answer for hallucinations.

**Request:** (multipart/form-data)

- `file`: PDF file (optional if using preloaded)
- `preloaded_pdf`: Filename (optional if uploading)
- `question`: Question text
- `answer`: Answer to verify

**Response:**

```json
{
  "success": true,
  "whitebox": {
    "SVM": {
      "label": "CORRECT",
      "confidence": 0.95,
      "prob_correct": 0.95,
      "prob_hallucinated": 0.05
    }
  },
  "blackbox": {
    "verdict": "GROUNDED",
    "entailment": 0.89,
    "neutral": 0.08,
    "contradiction": 0.03,
    "retrieved_context": "..."
  }
}
```

## Technology Stack

### Backend

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with auto-reload
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face models
- **FAISS**: Vector similarity search
- **Scikit-learn**: ML classifiers
- **XGBoost**: Gradient boosting

### Frontend

- **React 18**: UI library
- **Vite**: Build tool and dev server
- **React Router**: Client-side routing
- **Axios**: HTTP client
- **Tailwind CSS**: Utility-first CSS
- **React Dropzone**: Drag-and-drop file upload

## Development Workflow

### Backend Development

```bash
cd backend
python run.py  # Auto-reload enabled
```

Visit http://localhost:8000/docs for interactive API documentation.

### Frontend Development

```bash
cd frontend
npm run dev  # Hot Module Replacement enabled
```

Changes to React components update instantly without page reload.

## Production Build

### Backend

```bash
cd backend
# Set reload=False in run.py
python run.py
```

### Frontend

```bash
cd frontend
npm run build
npm run preview
```

Optimized build in `frontend/dist/` directory.

## Troubleshooting

### Backend Issues

- **Models not loading**: Verify model files in `../models/` directory
- **CUDA errors**: Check GPU availability with `torch.cuda.is_available()`
- **Port 8000 in use**: Change port in `backend/run.py`

### Frontend Issues

- **Cannot connect to backend**: Ensure backend is running on port 8000
- **Styles not loading**: Run `npm install` to install Tailwind CSS
- **Build fails**: Clear cache with `rm -rf node_modules/.vite && npm install`

## Legacy Gradio Interface

The original Gradio interface is still available in `src/app.py`:

```bash
python src/app.py
```

This launches the Gradio UI on http://localhost:7860

## Comparison: Gradio vs React + FastAPI

| Feature        | Gradio        | React + FastAPI |
| -------------- | ------------- | --------------- |
| Setup Time     | ⚡ Fast       | 🔧 Moderate     |
| Customization  | 🔒 Limited    | 🎨 Full Control |
| UI/UX          | 📊 Functional | ✨ Modern       |
| API Access     | ❌ No         | ✅ Yes          |
| Mobile Support | 📱 Basic      | 📱 Excellent    |
| Deployment     | 🚀 Simple     | 🏗️ Flexible     |

**Use Gradio for**: Quick prototypes, research demos, internal tools

**Use React + FastAPI for**: Production apps, public-facing tools, custom UX

## Next Steps

1. ✅ **Backend** - Complete FastAPI implementation
2. ✅ **Frontend** - Complete React UI with landing page
3. 🔄 **Testing** - Test full integration
4. 📚 **Documentation** - API and component docs
5. 🚢 **Deployment** - Docker containerization (optional)

---

For detailed setup instructions, see:

- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
