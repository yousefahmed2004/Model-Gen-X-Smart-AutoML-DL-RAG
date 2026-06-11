#!/usr/bin/env bash
# Convenience launcher — starts backend (port 8000) and frontend (port 5500).
# Press Ctrl+C to stop both.

set -e

cd "$(dirname "$0")"

# Ensure .env exists
if [ ! -f backend/.env ]; then
    echo "→ Creating backend/.env from .env.example"
    cp backend/.env.example backend/.env
fi

# Start backend in background
echo "→ Starting backend on http://localhost:8000"
(cd backend && uvicorn app.main:app --reload --port 8000) &
BACKEND_PID=$!

# Trap to clean up
trap "echo '→ Shutting down'; kill $BACKEND_PID 2>/dev/null; exit" INT TERM

# Wait a moment for backend, then start frontend
sleep 2
echo "→ Starting frontend on http://localhost:5500"
echo "→ Open http://localhost:5500 in your browser"
(cd frontend && python -m http.server 5500)
