# Smart AutoML Agent

> AutoML, with words. Upload a dataset, describe what you want to predict, and the agent builds, trains, and evaluates a model end-to-end — no notebooks, no boilerplate.

This is the **graduation-project edition**. It runs locally on CPU and ships with a working AutoML pipeline plus a polished frontend; the heavier production paths (Kaggle GPU, Dask streaming for >10 GB datasets) are stubbed behind clean interfaces and clearly marked so they can be plugged in later.

---

## Table of contents

1. [What's in the box](#whats-in-the-box)
2. [Quick start (5 minutes)](#quick-start-5-minutes)
3. [Project structure](#project-structure)
4. [Configuration & API keys](#configuration--api-keys)
5. [Architecture](#architecture)
6. [What's real vs stubbed](#whats-real-vs-stubbed)
7. [Endpoints](#endpoints)
8. [Troubleshooting](#troubleshooting)

---

## What's in the box

**Fully working:**
- FastAPI backend with **22 endpoints** (auth, projects, datasets, training, chat, predict, download)
- **Google OAuth 2.0** + email/password fallback, JWT auth
- SQLAlchemy ORM with 7 tables (users, projects, datasets, models, chats, messages, token ledger)
- **AutoML engine** (scikit-learn) — runs cross-validation across 4 candidate models, picks the winner, saves a portable joblib pipeline
- Dataset inspection: CSV, TSV, Excel, Parquet, JSON
- **Gemini 1.5 Flash** agent for chat (with a sensible rule-based fallback when no API key is configured)
- Token economy (1,000 free on signup, spent per train/predict)
- Premium frontend with 9 pages: landing, login/register, dashboard, chat, upload, training, results, playground, pricing
- Dark/light mode, English/Arabic (RTL), GSAP-free CSS animations

**Stubbed with clear interface contracts:**
- Kaggle GPU dispatch (training runs locally for the demo)
- Dask big-data streaming (auto-detected for files >500 MB; falls back to pandas)
- PyCaret backend (the `train_with_pycaret_stub` function in `automl_engine.py` is where it slots in)

---

## Quick start (5 minutes)

### 0. Prerequisites
- Python **3.11 or 3.12**
- A way to serve static files (we use Python's built-in `http.server`)

### 1. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Slim install (skip heavy deps):** PyCaret and Dask are large. For the basic demo you only need:
> ```bash
> pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy \
>     'python-jose[cryptography]' bcrypt python-multipart httpx \
>     email-validator numpy pandas scikit-learn joblib openpyxl
> ```

### 2. Copy the env file

```bash
cp .env.example .env
```

The defaults work out of the box — Gemini and Google OAuth will be in **fallback mode** (email/password login + rule-based agent). Fill in real keys when you're ready (see below).

### 3. Run the backend

```bash
# From the backend/ directory, with the venv active:
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Initializing database (creating tables if missing)…
INFO:     Smart AutoML Agent is ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

API docs: <http://localhost:8000/docs>

### 4. Serve the frontend

In a second terminal:
```bash
cd frontend
python -m http.server 5500
```

Open <http://localhost:5500> in your browser.

### 5. Try the full flow

1. Click **Get started** → register with any email + password (≥ 6 chars).
2. Open **Upload** in the sidebar.
3. Drop in a CSV — there's a sample at `samples/iris.csv` (or use any CSV with a target column).
4. Click **Configure training →**.
5. Pick the target column, click **Start training**.
6. Watch the live log; when it's done, click **Try it in playground →** to make predictions.

---

## Project structure

```
smart-automl-agent/
├── backend/
│   ├── app/
│   │   ├── api/                  ← FastAPI routers (auth, projects, datasets, training, chat)
│   │   ├── core/                 ← settings, security (JWT, bcrypt)
│   │   ├── db/                   ← SQLAlchemy engine + session
│   │   ├── models/               ← ORM models
│   │   ├── schemas/              ← Pydantic schemas
│   │   ├── ml/                   ← AutoML engine, dataset loader
│   │   ├── services/             ← Gemini, Kaggle (stub), token accounting
│   │   └── main.py               ← FastAPI app
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── css/
│   │   ├── main.css              ← design system, theme tokens
│   │   ├── landing.css           ← landing-page styles
│   │   └── app.css               ← in-app layout (sidebar, dashboard, etc.)
│   ├── js/
│   │   ├── common.js             ← API client, auth, theme, i18n
│   │   ├── sidebar.js            ← sidebar rendering + user load
│   │   └── landing.js            ← terminal animation, stat counters
│   ├── pages/
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── chat.html
│   │   ├── upload.html
│   │   ├── training.html
│   │   ├── results.html
│   │   ├── playground.html
│   │   └── pricing.html
│   └── index.html                ← landing page
├── samples/
│   └── iris.csv                  ← demo dataset
├── uploads/                      ← user-uploaded datasets (created on first run)
├── trained_models/               ← saved joblib bundles (created on first run)
└── README.md
```

---

## Configuration & API keys

All configuration lives in `backend/.env`. The platform works without any of these — Gemini and Google OAuth gracefully fall back. Add them when you want the full experience.

### Google OAuth 2.0

1. Go to <https://console.cloud.google.com/apis/credentials>.
2. Create an **OAuth 2.0 Client ID** → application type: **Web application**.
3. Add this Authorized redirect URI:
   ```
   http://localhost:8000/api/auth/google/callback
   ```
4. Copy the **Client ID** and **Client Secret** into `.env`:
   ```env
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   GOOGLE_REDIRECT_URI=http://localhost:8000/api/auth/google/callback
   ```
5. Restart the backend.

### Gemini API

1. Get a key at <https://aistudio.google.com/app/apikey> (free tier available).
2. Add to `.env`:
   ```env
   GEMINI_API_KEY=...
   GEMINI_MODEL=gemini-1.5-flash
   ```
3. Install the SDK if you skipped the heavy deps:
   ```bash
   pip install google-generativeai
   ```
4. Restart the backend. The agent will now use Gemini instead of the rule-based fallback.

### Kaggle API (optional — stub)

The Kaggle GPU path is **not implemented**; this section is for when you want to enable it.

1. Get an API token at <https://www.kaggle.com/settings> → "Create New Token". You'll download `kaggle.json`.
2. Move it:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Open `backend/app/services/kaggle_service.py` and implement `dispatch()` / `poll()`. The interface contract is documented in the file.

### Database

Default is **SQLite** (`smart_automl.db` in the backend folder). Switch to PostgreSQL by changing `DATABASE_URL`:

```env
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/smart_automl
```

The schema is auto-created on startup via `init_db()`. For real production, use Alembic migrations.

---

## Architecture

```
┌──────────┐         HTTP (REST)         ┌──────────────────────┐
│ Browser  │ ◄────────────────────────► │  FastAPI Backend     │
│ (static  │                             │                      │
│  HTML/JS)│                             │  ┌────────────────┐  │
└──────────┘                             │  │ Auth (OAuth +  │  │
                                         │  │  email JWT)    │  │
                                         │  └───────┬────────┘  │
                                         │          │           │
                                         │  ┌───────▼────────┐  │
                                         │  │  AutoML engine │  │
                                         │  │  (scikit-learn │  │
                                         │  │   pipelines)   │  │
                                         │  └───────┬────────┘  │
                                         │          │           │
                                         │  ┌───────▼────────┐  │
                                         │  │  Persistence   │  │
                                         │  │  (SQLAlchemy   │  │
                                         │  │   + joblib)    │  │
                                         │  └────────────────┘  │
                                         │                      │
                                         │  ┌────────────────┐  │
                                         │  │ Gemini client  │──┼─► Google AI Studio
                                         │  └────────────────┘  │
                                         │  ┌────────────────┐  │
                                         │  │ Kaggle (stub)  │──┼─► Kaggle Notebooks (T4/P100)
                                         │  └────────────────┘  │
                                         └──────────────────────┘
```

**Training data flow:** `POST /api/training/train` →
1. Authenticate user, debit tokens
2. Load dataset via `dataset_loader.load_dataframe` (auto-detects CSV/Excel/Parquet)
3. Drop high-cardinality ID columns, build `ColumnTransformer` (impute + encode + scale)
4. 3-fold cross-validate across 4 candidate models (Random Forest, Logistic/Ridge, Gradient Boosting, Decision Tree)
5. Fit the winner on full training split, evaluate on held-out 20% test
6. Persist as `joblib` bundle (pipeline + feature columns + target + task type + class names)
7. Return metrics + confusion matrix + training log

The artifact is **fully self-contained** — you can `joblib.load()` it in any Python 3.11+ environment and call `bundle["pipeline"].predict(X)`.

---

## What's real vs stubbed

| Feature | Status | Where |
|---|---|---|
| Email/password auth + JWT | ✅ Works | `app/api/auth.py` |
| Google OAuth 2.0 | ✅ Works (needs keys) | `app/api/auth.py` |
| Dataset upload + profiling | ✅ Works | `app/api/datasets.py`, `app/ml/dataset_loader.py` |
| AutoML training (tabular) | ✅ Works, CV-tested | `app/ml/automl_engine.py` |
| Model persistence + download | ✅ Works | `app/api/training.py` |
| Prediction playground | ✅ Works | `app/api/training.py::predict_endpoint` |
| Gemini chat agent | ✅ Works (with fallback) | `app/services/gemini_service.py` |
| Token economy | ✅ Works | `app/services/token_service.py` |
| **Dask streaming (>10 GB)** | ⚠️ Stub — pandas-only for now | `app/ml/dataset_loader.py::BIG_FILE_THRESHOLD_MB` |
| **Kaggle GPU dispatch** | ⚠️ Stub — interface defined | `app/services/kaggle_service.py` |
| **PyCaret backend** | ⚠️ Stub — sklearn backend is the real engine | `app/ml/automl_engine.py::train_with_pycaret_stub` |
| **Deep learning (PyTorch/TF)** | ⚠️ Not in this build | (see Architecture above for the dispatch point) |

The stubs are intentional and documented in code. Enabling them is straightforward but not necessary for the graduation demo.

---

## Endpoints

Full OpenAPI docs at <http://localhost:8000/docs>. Summary:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info |
| GET | `/api/health` | Health check |
| POST | `/api/auth/register` | Email/password registration |
| POST | `/api/auth/login` | Email/password login |
| GET | `/api/auth/google/login` | Start OAuth flow |
| GET | `/api/auth/google/callback` | OAuth callback |
| GET | `/api/auth/me` | Current user |
| GET | `/api/projects` | List projects |
| POST | `/api/projects` | Create project |
| GET | `/api/projects/{id}` | Get project |
| DELETE | `/api/projects/{id}` | Delete project |
| POST | `/api/datasets/upload` | Upload dataset (multipart) |
| GET | `/api/datasets/project/{id}` | List datasets for project |
| GET | `/api/datasets/{id}/preview` | Get dataset preview |
| POST | `/api/training/train` | Train a model |
| GET | `/api/training/models/project/{id}` | List models |
| GET | `/api/training/models/{id}` | Model details |
| GET | `/api/training/models/{id}/download` | Download .joblib |
| POST | `/api/training/predict` | Make a prediction |
| GET | `/api/chat` | List chats |
| POST | `/api/chat` | New chat |
| GET | `/api/chat/{id}/messages` | Chat history |
| POST | `/api/chat/send` | Send a message |

---

## Troubleshooting

**`ImportError: bcrypt has no attribute __about__`**
You probably have `passlib` and a newer `bcrypt` installed. The code uses `bcrypt` directly, so just remove passlib:
```bash
pip uninstall passlib
```

**CORS errors in the browser**
Make sure the frontend origin in `.env` matches where you serve the frontend from:
```env
FRONTEND_ORIGIN=http://localhost:5500
```

**Google OAuth redirect mismatch**
The redirect URI in Google Cloud Console **must exactly match** `GOOGLE_REDIRECT_URI` in `.env`. Trailing slashes matter.

**Backend port already in use**
```bash
uvicorn app.main:app --port 8001
```
…and update the `<meta name="api-base">` tag in the frontend HTML files (or set it in `js/common.js`).

**“no such table: users”**
Delete `backend/smart_automl.db` and restart — the schema will be recreated on lifespan startup.

**Training takes too long on big datasets**
For files >100 MB on free-tier hardware, training can take minutes. The first request blocks the backend (training is synchronous in this build). For production you'd want a Celery/RQ worker.

---

## License

MIT — do whatever you want. This is graduation-project code; please don't deploy it to production without hardening (real Alembic migrations, async training workers, proper CSRF for OAuth state, secrets management, rate limiting).
