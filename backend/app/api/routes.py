"""
API Routes
Defines all FastAPI endpoints for the HRES application
"""
import os
import tempfile
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from ..services.detection import whitebox_predict, blackbox_predict, get_pdf_index
from ..services.model_loader import get_llama, get_embedder
from ..services.answer_generator import generate_answer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PDFS_DIR = os.path.join(BASE_DIR, "resources", "pdfs")

router = APIRouter()


# ── Response Models ──────────────────────────────────────────────────────────

class WhiteboxResult(BaseModel):
    """Single classifier result (matches app.py output format)"""
    model: str
    prediction: int
    label: str
    confidence: float
    prob_correct: float
    prob_hallucinated: float


class BlackboxResult(BaseModel):
    verdict: str
    entailment: float
    neutral: float
    contradiction: float
    retrieved_context: str


class PredictResponse(BaseModel):
    success: bool
    generated_answer: Optional[str] = None
    whitebox: Optional[WhiteboxResult] = None
    whitebox_error: Optional[str] = None
    blackbox: Optional[BlackboxResult] = None
    blackbox_error: Optional[str] = None


class GenerateAnswerResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    context: Optional[str] = None
    error: Optional[str] = None


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


@router.post("/generate-and-verify")
async def generate_and_verify_answer(
    file: Optional[UploadFile] = File(None),
    preloaded_pdf: Optional[str] = Form(None),
    question: str = Form(...),
):
    """
    Generate an answer using Gemma based on the PDF context, then verify it.
    """
    # Validate inputs
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Determine PDF path
    pdf_path = None
    temp_file = None
    
    try:
        if preloaded_pdf:
            pdf_path = os.path.join(PDFS_DIR, preloaded_pdf)
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail=f"Preloaded PDF '{preloaded_pdf}' not found")
        elif file:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = await file.read()
            
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
            
            temp_file.write(content)
            temp_file.close()
            pdf_path = temp_file.name
        else:
            raise HTTPException(status_code=400, detail="Either upload a PDF or select a preloaded PDF")
        
        # Step 1: Generate answer using Gemma
        generated_answer = None
        gen_error = None
        try:
            generated_answer, gen_error = generate_answer(pdf_path, question)
        except Exception as e:
            gen_error = f"Answer generation failed: {str(e)}"
        
        if gen_error or not generated_answer:
            return {
                "success": False,
                "generated_answer": None,
                "whitebox": None,
                "whitebox_error": gen_error or "Failed to generate answer",
                "blackbox": None,
                "blackbox_error": None,
            }
        
        # Step 2: Run whitebox pipeline on the generated answer
        wb_results = None
        wb_error = None
        try:
            wb_results, wb_error = whitebox_predict(pdf_path, question, generated_answer)
        except Exception as e:
            wb_error = f"Whitebox pipeline failed: {str(e)}"
        
        # Step 3: Run blackbox pipeline on the generated answer
        bb_results = None
        bb_error = None
        try:
            bb_results, bb_error = blackbox_predict(pdf_path, question, generated_answer)
        except Exception as e:
            bb_error = f"Blackbox pipeline failed: {str(e)}"
        
        return {
            "success": True,
            "generated_answer": generated_answer,
            "whitebox": wb_results,
            "whitebox_error": wb_error,
            "blackbox": bb_results,
            "blackbox_error": bb_error,
        }
    
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
