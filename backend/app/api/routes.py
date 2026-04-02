"""
API Routes
Defines all FastAPI endpoints for the HRES application
"""
import os
import tempfile
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from ..services.detection import whitebox_predict, blackbox_predict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PDFS_DIR = os.path.join(BASE_DIR, "resources", "pdfs")

router = APIRouter()


# ── Response Models ──────────────────────────────────────────────────────────

class WhiteboxModelResult(BaseModel):
    prediction: int
    label: str
    confidence: float
    prob_correct: float
    prob_hallucinated: float


class WhiteboxResult(BaseModel):
    SVM: Optional[WhiteboxModelResult] = None
    XGBoost: Optional[WhiteboxModelResult] = None


class BlackboxResult(BaseModel):
    verdict: str
    entailment: float
    neutral: float
    contradiction: float
    retrieved_context: str


class PredictResponse(BaseModel):
    success: bool
    whitebox: Optional[dict] = None
    whitebox_error: Optional[str] = None
    blackbox: Optional[dict] = None
    blackbox_error: Optional[str] = None


class PreloadedPDF(BaseModel):
    filename: str
    display_name: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/preloaded-pdfs", response_model=list[PreloadedPDF])
async def get_preloaded_pdfs():
    """
    List all preloaded PDFs from resources/pdfs/ directory
    """
    if not os.path.exists(PDFS_DIR):
        return []
    
    pdfs = []
    for filename in os.listdir(PDFS_DIR):
        if filename.lower().endswith('.pdf'):
            display_name = filename.replace('_', ' ').replace('.pdf', '')
            pdfs.append({
                "filename": filename,
                "display_name": display_name
            })
    
    return pdfs


@router.post("/predict", response_model=PredictResponse)
async def predict_hallucination(
    file: Optional[UploadFile] = File(None),
    preloaded_pdf: Optional[str] = Form(None),
    question: str = Form(...),
    answer: str = Form(...),
):
    """
    Run hallucination detection on the provided PDF, question, and answer.
    Either upload a file or specify a preloaded PDF filename.
    """
    # Validate inputs
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not answer or not answer.strip():
        raise HTTPException(status_code=400, detail="Answer is required")
    
    # Determine PDF path
    pdf_path = None
    temp_file = None
    
    try:
        if preloaded_pdf:
            # Use preloaded PDF
            pdf_path = os.path.join(PDFS_DIR, preloaded_pdf)
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail=f"Preloaded PDF '{preloaded_pdf}' not found")
        elif file:
            # Validate uploaded file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = await file.read()
            
            # Check file size (50MB limit)
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
            
            temp_file.write(content)
            temp_file.close()
            pdf_path = temp_file.name
        else:
            raise HTTPException(status_code=400, detail="Either upload a PDF or select a preloaded PDF")
        
        # Run whitebox pipeline
        wb_results = None
        wb_error = None
        try:
            wb_results, wb_error = whitebox_predict(pdf_path, question, answer)
        except Exception as e:
            wb_error = f"Whitebox pipeline failed: {str(e)}"
        
        # Run blackbox pipeline
        bb_results = None
        bb_error = None
        try:
            bb_results, bb_error = blackbox_predict(pdf_path, question, answer)
        except Exception as e:
            bb_error = f"Blackbox pipeline failed: {str(e)}"
        
        return {
            "success": True,
            "whitebox": wb_results,
            "whitebox_error": wb_error,
            "blackbox": bb_results,
            "blackbox_error": bb_error,
        }
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
