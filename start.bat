@echo off
echo ====================================
echo HRES Full-Stack Application Startup
echo ====================================
echo.

echo [1/4] Installing Backend Dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)
echo Backend dependencies installed successfully!
echo.

cd ..

echo [2/4] Installing Frontend Dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    echo Make sure Node.js and npm are installed
    pause
    exit /b 1
)
echo Frontend dependencies installed successfully!
echo.

cd ..

echo [3/4] Starting Backend Server (Port 8000)...
echo.
start "HRES Backend" cmd /k "cd backend && python run.py"
timeout /t 3 /nobreak >nul
echo Backend server starting...
echo.

echo [4/4] Starting Frontend Server (Port 5173)...
echo.
start "HRES Frontend" cmd /k "cd frontend && npm run dev"
echo.

echo ====================================
echo HRES Application is starting!
echo ====================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Two terminal windows have been opened:
echo   1. Backend (FastAPI on port 8000)
echo   2. Frontend (React on port 5173)
echo.
echo Wait a few seconds for servers to start, then open:
echo   http://localhost:5173
echo.
echo Press Ctrl+C in each terminal window to stop the servers.
echo.
pause
