"""
FastAPI Main Application
HRES Hallucination Detection Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router

app = FastAPI(
    title="HRES Hallucination Detection API",
    description="Dual-pipeline hallucination detection system using whitebox (HRES) and blackbox (NLI) approaches",
    version="1.0.0",
)

# Configure CORS to allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HRES Hallucination Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Optional: Add startup event for model pre-loading
@app.on_event("startup")
async def startup_event():
    """
    Startup event - can be used to pre-load models if desired
    Currently models are loaded lazily on first request
    """
    print("🚀 HRES API Server starting...")
    print("📚 Models will be loaded on first request")
    print("📄 API documentation available at: http://localhost:8000/docs")
