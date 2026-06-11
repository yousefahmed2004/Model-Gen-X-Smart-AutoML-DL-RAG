@echo off
REM Convenience launcher for Windows. Opens backend in one window and frontend in another.

cd /d "%~dp0"

if not exist backend\.env (
    echo Creating backend\.env from .env.example
    copy backend\.env.example backend\.env
)

echo Starting backend on http://localhost:8000
start "AutoML backend" cmd /k "cd backend && uvicorn app.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo Starting frontend on http://localhost:5500
echo Open http://localhost:5500 in your browser
start "AutoML frontend" cmd /k "cd frontend && python -m http.server 5500"
