@echo off
echo ================================================
echo HRES - Setup Verification
echo ================================================
echo.

echo [1/5] Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo [2/5] Checking CUDA availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] PyTorch is not installed. Install with:
    echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo.
)

echo [3/5] Checking required packages...
python -c "import transformers, gradio, sentence_transformers, faiss, joblib, sklearn, xgboost; print('All packages found!')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages are missing. Install with:
    echo    pip install -r requirements.txt
    echo.
)

echo [4/5] Checking TinyLlama model...
if exist "models\TinyLlama" (
    echo [OK] TinyLlama model directory exists
) else (
    echo [WARNING] TinyLlama not found. Download with:
    echo    python download_model.py
)
echo.

echo [5/5] Checking trained models...
if exist "models\svm_model_final.pkl" (
    echo [OK] Trained models found
) else (
    echo [WARNING] Trained models not found
)
echo.

echo ================================================
echo Setup check complete!
echo To run the app: run_app.bat
echo ================================================
pause
