"""RAG Bot API — full professional config."""
from __future__ import annotations
import io, json, logging, re, shutil, threading, uuid, zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.rag_engine import build_vector_db, query_bot, SUPPORTED_EXTENSIONS
from app.models.rag_bot import RAGBot, RAGMessage
from app.models.user import User
from app.services import token_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])

TOKEN_BUILD_SMALL = 200
TOKEN_BUILD_LARGE = 500
TOKEN_QUERY       = 5


class CreateBotRequest(BaseModel):
    # Identity
    name:          str
    description:   str   = ""
    language:      str   = "auto"
    welcome_msg:   str   = "Hello! How can I help you today?"
    system_prompt: str   = "You are a helpful assistant. Answer based only on the provided documents."
    fallback_msg:  str   = "I don't have information about that in my documents."
    # Retrieval
    chunk_size:    int   = 500
    chunk_overlap: int   = 50
    top_k:         int   = 3
    similarity_threshold: float = 0.3
    # Generation
    temperature:   float = 0.7
    max_tokens:    int   = 1024
    show_sources:  bool  = False
    allow_general: bool  = False
    output_format: str   = "text"


class UpdateBotRequest(BaseModel):
    name:          str   | None = None
    description:   str   | None = None
    language:      str   | None = None
    welcome_msg:   str   | None = None
    system_prompt: str   | None = None
    fallback_msg:  str   | None = None
    chunk_size:    int   | None = None
    chunk_overlap: int   | None = None
    top_k:         int   | None = None
    similarity_threshold: float | None = None
    temperature:   float | None = None
    max_tokens:    int   | None = None
    show_sources:  bool  | None = None
    allow_general: bool  | None = None
    output_format: str   | None = None


class QueryRequest(BaseModel):
    question: str
    history:  list[dict] = []


@router.post("")
def create_bot(req: CreateBotRequest,
               db: Session = Depends(get_db),
               user: User  = Depends(get_current_user)):
    base     = Path(settings.upload_path) / "rag_bots" / str(user.id)
    bot_uuid = uuid.uuid4().hex[:12]
    docs_dir = base / bot_uuid / "docs"
    vdb_dir  = base / bot_uuid / "vectordb"
    docs_dir.mkdir(parents=True, exist_ok=True)
    vdb_dir.mkdir(parents=True, exist_ok=True)

    bot = RAGBot(
        user_id=user.id, name=req.name, description=req.description,
        language=req.language, welcome_msg=req.welcome_msg,
        system_prompt=req.system_prompt, fallback_msg=req.fallback_msg,
        docs_dir=str(docs_dir), vectordb_path=str(vdb_dir),
        chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
        top_k=req.top_k, similarity_threshold=req.similarity_threshold,
        temperature=req.temperature, max_tokens=req.max_tokens,
        show_sources=req.show_sources, allow_general=req.allow_general,
        output_format=req.output_format, status="empty",
    )
    db.add(bot); db.commit(); db.refresh(bot)
    return _fmt(bot)


@router.get("")
def list_bots(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bots = db.query(RAGBot).filter(RAGBot.user_id == user.id).order_by(RAGBot.created_at.desc()).all()
    return [_fmt(b) for b in bots]


@router.get("/{bot_id}")
def get_bot(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    return _fmt(bot)


@router.patch("/{bot_id}/update")
def update_bot(bot_id: int, req: UpdateBotRequest,
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    for field, val in req.dict(exclude_none=True).items():
        setattr(bot, field, val)
    db.commit(); db.refresh(bot)
    return _fmt(bot)


@router.post("/{bot_id}/upload")
async def upload_documents(bot_id: int, files: list[UploadFile] = File(...),
                            db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    saved = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS: continue
        dest = Path(bot.docs_dir) / f"{uuid.uuid4().hex[:8]}_{f.filename}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest.name)
    if not saved:
        raise HTTPException(400, f"No valid files. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}")
    return {"uploaded": saved, "message": f"Uploaded {len(saved)} file(s)"}


@router.post("/{bot_id}/build")
def build_bot(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    files = [f for f in Path(bot.docs_dir).iterdir() if f.is_file()]
    if not files: raise HTTPException(400, "No documents uploaded")
    total_mb = sum(f.stat().st_size for f in files) / (1024*1024)
    cost = TOKEN_BUILD_SMALL if total_mb < 5 else TOKEN_BUILD_LARGE
    try:
        token_service.spend(db, user, cost, f"rag_build_{bot_id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(402, str(exc))
    bot.status = "building"; db.commit()

    def _build():
        from app.db.session import SessionLocal
        local_db = SessionLocal()
        try:
            b = local_db.query(RAGBot).filter(RAGBot.id == bot_id).first()
            stats = build_vector_db(b.docs_dir, b.vectordb_path, b.chunk_size, b.chunk_overlap)
            b.n_documents = stats["n_documents"]
            b.n_chunks    = stats["n_chunks"]
            b.total_size_mb = stats["total_size_mb"]
            b.status = "ready"
            local_db.commit()
        except Exception as e:
            logger.error("RAG build failed: %s", e)
            b = local_db.query(RAGBot).filter(RAGBot.id == bot_id).first()
            if b: b.status = "failed"; local_db.commit()
        finally:
            local_db.close()

    threading.Thread(target=_build, daemon=True).start()
    return {"message": "Building started", "tokens_spent": cost}


@router.post("/{bot_id}/query")
def query(bot_id: int, req: QueryRequest,
          db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    if bot.status != "ready": raise HTTPException(400, f"Bot not ready ({bot.status})")
    try:
        token_service.spend(db, user, TOKEN_QUERY, f"rag_query_{bot_id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(402, str(exc))

    result = query_bot(
        vectordb_path=bot.vectordb_path, question=req.question,
        system_prompt=bot.system_prompt, top_k=bot.top_k,
        temperature=bot.temperature, history=req.history,
        max_tokens=bot.max_tokens, language=bot.language,
        fallback_msg=bot.fallback_msg, allow_general=bot.allow_general,
        similarity_threshold=bot.similarity_threshold,
    )

    db.add(RAGMessage(bot_id=bot_id, user_id=user.id, role="user", content=req.question))
    db.add(RAGMessage(bot_id=bot_id, user_id=user.id, role="assistant",
                      content=result["answer"], sources=json.dumps(result["sources"])))
    db.commit()

    if not bot.show_sources:
        result["sources"] = []
    return result


@router.get("/{bot_id}/messages")
def list_messages(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    msgs = db.query(RAGMessage).filter(RAGMessage.bot_id == bot_id, RAGMessage.user_id == user.id).order_by(RAGMessage.created_at).all()
    return [{"role": m.role, "content": m.content,
             "sources": json.loads(m.sources) if m.sources else [],
             "created_at": m.created_at.isoformat()} for m in msgs]


@router.delete("/{bot_id}")
def delete_bot(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    try:
        p = Path(bot.docs_dir)
        if p.parent.exists(): shutil.rmtree(p.parent)
    except: pass
    db.query(RAGMessage).filter(RAGMessage.bot_id == bot_id).delete()
    db.delete(bot); db.commit()
    return {"message": "Deleted"}


@router.get("/{bot_id}/download")
def download_package(bot_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bot = db.query(RAGBot).filter(RAGBot.id == bot_id, RAGBot.user_id == user.id).first()
    if not bot: raise HTTPException(404, "Bot not found")
    if bot.status != "ready": raise HTTPException(400, "Bot not ready")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in Path(bot.vectordb_path).rglob("*"):
            if f.is_file(): zf.write(f, f"vectordb/{f.relative_to(bot.vectordb_path)}")
        for f in Path(bot.docs_dir).iterdir():
            if f.is_file(): zf.write(f, f"docs/{f.name}")
        zf.writestr("config.json", json.dumps(_config_json(bot), ensure_ascii=False, indent=2))
        zf.writestr("server.py",     _gen_server(bot))
        zf.writestr("requirements.txt", _gen_requirements())
        zf.writestr("static/index.html", _gen_ui(bot))
        zf.writestr("README.md",     _gen_readme(bot))
        zf.writestr(".env.example",  "GROQ_API_KEY=your_key_here\n")

    buf.seek(0)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", bot.name)[:40]
    return StreamingResponse(iter([buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}_bot.zip"'})


def _config_json(b: RAGBot) -> dict:
    return {
        "name": b.name, "description": b.description,
        "language": b.language, "welcome_message": b.welcome_msg,
        "system_prompt": b.system_prompt, "fallback_message": b.fallback_msg,
        "retrieval": {"top_k": b.top_k, "chunk_size": b.chunk_size,
                      "chunk_overlap": b.chunk_overlap,
                      "similarity_threshold": b.similarity_threshold},
        "generation": {"temperature": b.temperature, "max_tokens": b.max_tokens,
                       "output_format": b.output_format},
        "behavior": {"show_sources": b.show_sources, "allow_general": b.allow_general},
        "stats": {"documents": b.n_documents, "chunks": b.n_chunks, "size_mb": b.total_size_mb},
    }


def _fmt(b: RAGBot) -> dict:
    return {
        "id": b.id, "name": b.name, "description": b.description,
        "language": b.language, "welcome_msg": b.welcome_msg,
        "system_prompt": b.system_prompt, "fallback_msg": b.fallback_msg,
        "n_documents": b.n_documents, "n_chunks": b.n_chunks,
        "total_size_mb": b.total_size_mb, "chunk_size": b.chunk_size,
        "chunk_overlap": b.chunk_overlap, "top_k": b.top_k,
        "similarity_threshold": b.similarity_threshold,
        "temperature": b.temperature, "max_tokens": b.max_tokens,
        "show_sources": b.show_sources, "allow_general": b.allow_general,
        "output_format": b.output_format, "status": b.status,
        "created_at": b.created_at.isoformat() if b.created_at else None,
    }


def _gen_server(b: RAGBot) -> str:
    return f'''"""RAG Bot Server — {b.name} | Generated by Model Gen X"""
import json, os, logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

with open("config.json") as f:
    CONFIG = json.load(f)

TOP_K         = CONFIG["retrieval"]["top_k"]
TEMPERATURE   = CONFIG["generation"]["temperature"]
MAX_TOKENS    = CONFIG["generation"]["max_tokens"]
SYSTEM_PROMPT = CONFIG["system_prompt"]
FALLBACK_MSG  = CONFIG["fallback_message"]
ALLOW_GENERAL = CONFIG["behavior"]["allow_general"]
LANGUAGE      = CONFIG["language"]
SHOW_SOURCES  = CONFIG["behavior"]["show_sources"]
EMBED_MODEL   = "all-MiniLM-L6-v2"

_embedder = None
_collection = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient("vectordb", settings=Settings(anonymized_telemetry=False))
        _collection = client.get_collection("documents")
    return _collection

app = FastAPI(title=CONFIG["name"])
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryReq(BaseModel):
    question: str
    history: list = []

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/api/config")
def config():
    return {{"name": CONFIG["name"], "welcome": CONFIG["welcome_message"], "language": LANGUAGE}}

@app.post("/api/query")
def query(req: QueryReq):
    if not GROQ_KEY:
        return {{"answer": "Set GROQ_API_KEY environment variable.", "sources": []}}
    q_emb = get_embedder().encode([req.question]).tolist()[0]
    results = get_collection().query(query_embeddings=[q_emb], n_results=TOP_K)
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return {{"answer": FALLBACK_MSG, "sources": []}}
    context = "\\n\\n---\\n\\n".join(f"[{{m.get('source','?')}}]\\n{{t}}" for t,m in zip(docs,metas))
    lang_instruction = ""
    if LANGUAGE == "ar": lang_instruction = "Always respond in Arabic."
    elif LANGUAGE == "en": lang_instruction = "Always respond in English."
    else: lang_instruction = "Respond in the same language as the user."
    strict = "" if ALLOW_GENERAL else "Answer ONLY from the context. If not found, say: " + FALLBACK_MSG
    messages = [{{"role": "system", "content": f"{{SYSTEM_PROMPT}}\\n{{lang_instruction}}\\n{{strict}}\\n\\nCONTEXT:\\n{{context}}"}}]
    for h in req.history[-6:]:
        if h.get("content"):
            messages.append({{"role": "assistant" if h.get("role") in ("assistant","model") else "user", "content": h["content"]}})
    messages.append({{"role": "user", "content": req.question}})
    r = httpx.post(GROQ_URL,
        headers={{"Authorization": f"Bearer {{GROQ_KEY}}", "Content-Type": "application/json"}},
        json={{"model": GROQ_MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}},
        timeout=30)
    if r.status_code != 200:
        return {{"answer": f"Error: {{r.text[:100]}}", "sources": []}}
    answer = r.json()["choices"][0]["message"]["content"]
    sources = [{{"source": m.get("source","?"), "text": t[:150]}} for t,m in zip(docs,metas)] if SHOW_SOURCES else []
    return {{"answer": answer, "sources": sources}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def _gen_requirements() -> str:
    return "fastapi==0.115.0\nuvicorn[standard]==0.32.0\nchromadb==0.4.24\nsentence-transformers==3.0.0\nhttpx==0.27.0\npydantic==2.9.0\npython-multipart==0.0.9\n"



def _gen_ui(b: RAGBot) -> str:
    name = b.name
    desc = b.description or "AI Assistant"
    lang = b.language or "auto"
    welcome = b.welcome_msg or "Hello!"
    is_ar = lang == "ar"
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"" + lang + "\" dir=\"" + ("rtl" if is_ar else "ltr") + "\">\n"
        "<head><meta charset=\"UTF-8\"/>\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n"
        "<title>" + name + "</title>\n"
        "<link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300..600&family=Cairo:wght@300..700&display=swap\" rel=\"stylesheet\"/>\n"
        "<style>\n"
        "*{box-sizing:border-box;margin:0;padding:0;}\n"
        "body{background:#06121A;color:#E6EDF3;font-family:" + ("Cairo" if is_ar else "Inter") + ",sans-serif;height:100vh;display:flex;flex-direction:column;}\n"
        ".top{background:#0D1B2A;border-bottom:1px solid #1E2D40;padding:16px 24px;display:flex;align-items:center;gap:12px;}\n"
        ".av{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#00E5FF,#0099CC);display:flex;align-items:center;justify-content:center;font-weight:700;color:#06121A;}\n"
        ".online{width:8px;height:8px;border-radius:50%;background:#3FB950;margin-left:auto;box-shadow:0 0 6px #3FB950;}\n"
        ".chat{flex:1;overflow-y:auto;padding:24px 16px;max-width:800px;margin:0 auto;width:100%;}\n"
        ".msg{margin-bottom:20px;display:flex;gap:12px;}\n"
        ".msg.u{flex-direction:row-reverse;}\n"
        ".mav{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0;}\n"
        ".bav{background:linear-gradient(135deg,#00E5FF,#0099CC);color:#06121A;}\n"
        ".uav{background:#1E2D40;color:#E6EDF3;}\n"
        ".bbl{background:#0D1B2A;border:1px solid #1E2D40;border-radius:16px;padding:12px 16px;font-size:14px;line-height:1.65;max-width:75%;white-space:pre-wrap;}\n"
        ".u .bbl{background:#00E5FF;color:#06121A;border:none;}\n"
        ".typing span{display:inline-block;width:6px;height:6px;border-radius:50%;background:#8B949E;margin-right:3px;animation:bn 1.4s infinite;}\n"
        ".typing span:nth-child(2){animation-delay:.15s;}.typing span:nth-child(3){animation-delay:.3s;}\n"
        "@keyframes bn{0%,60%,100%{transform:translateY(0);opacity:.4;}30%{transform:translateY(-4px);opacity:1;}}\n"
        ".bar{padding:16px;border-top:1px solid #1E2D40;background:#0D1B2A;}\n"
        ".wrap{max-width:800px;margin:0 auto;display:flex;gap:10px;}\n"
        ".wrap input{flex:1;background:#06121A;border:1px solid #1E2D40;border-radius:12px;padding:12px 16px;color:#E6EDF3;font-size:14px;font-family:inherit;}\n"
        ".wrap input:focus{outline:none;border-color:#00E5FF;}\n"
        ".wrap button{background:#00E5FF;color:#06121A;border:none;border-radius:12px;padding:12px 20px;cursor:pointer;font-weight:600;font-family:inherit;}\n"
        ".pw{text-align:center;font-size:11px;color:#6E7681;padding:8px;}\n"
        ".pw a{color:#00E5FF;text-decoration:none;}\n"
        "</style></head><body>\n"
        "<div class=\"top\">\n"
        "  <div class=\"av\">" + (name[0].upper() if name else "B") + "</div>\n"
        "  <div><div style=\"font-weight:600\">" + name + "</div><div style=\"font-size:12px;color:#8B949E\">" + desc + "</div></div>\n"
        "  <div class=\"online\"></div>\n"
        "</div>\n"
        "<div class=\"chat\" id=\"chat\">\n"
        "  <div id=\"w\" style=\"text-align:center;padding:40px 20px\"><h2 style=\"font-size:22px;margin-bottom:8px\">" + welcome + "</h2></div>\n"
        "</div>\n"
        "<div class=\"bar\"><div class=\"wrap\">\n"
        "  <input type=\"text\" id=\"inp\" placeholder=\"" + ("اكتب سؤالك..." if is_ar else "Type your message...") + "\" onkeydown=\"if(event.key===\'Enter\')send()\"/>\n"
        "  <button onclick=\"send()\">Send &rarr;</button>\n"
        "</div></div>\n"
        "<div class=\"pw\">Powered by <a href=\"https://modelgenx.site\" target=\"_blank\">Model Gen X</a></div>\n"
        "<script>\n"
        "const H=[];let C={};\n"
        "fetch(\'/api/config\').then(r=>r.json()).then(c=>{C=c;});\n"
        "function esc(s){return String(s).replace(/[&<>\"\'/]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[ c]||c);}\n"
        "function addMsg(role,txt){\n"
        "  document.getElementById(\'w\').style.display=\'none\';\n"
        "  const d=document.createElement(\'div\');\n"
        "  d.className=\'msg\'+(role===\'user\'?\' u\':\'\');\n"
        "  const av=role===\'user\'?\'U\':(C.name?C.name[0].toUpperCase():\'B\');\n"
        "  d.innerHTML=\'<div class=\"mav \'+(role===\'user\'?\'uav\':\'bav\')+\'\">\'+av+\'</div><div class=\"bbl\">\'+esc(txt)+\'</div>\';\n"
        "  document.getElementById(\'chat\').appendChild(d);\n"
        "  document.getElementById(\'chat\').scrollTop=99999;return d;\n"
        "}\n"
        "function addTyping(){\n"
        "  const d=document.createElement(\'div\');d.className=\'msg\';\n"
        "  const av=C.name?C.name[0].toUpperCase():\'B\';\n"
        "  d.innerHTML=\'<div class=\"mav bav\">\'+av+\'</div><div class=\"bbl typing\"><span></span><span></span><span></span></div>\';\n"
        "  document.getElementById(\'chat\').appendChild(d);\n"
        "  document.getElementById(\'chat\').scrollTop=99999;return d;\n"
        "}\n"
        "async function send(){\n"
        "  const inp=document.getElementById(\'inp\');\n"
        "  const q=inp.value.trim();if(!q)return;\n"
        "  inp.value=\'\';addMsg(\'user\',q);H.push({role:\'user\',content:q});\n"
        "  const t=addTyping();\n"
        "  try{\n"
        "    const r=await fetch(\'/api/query\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify({question:q,history:H.slice(0,-1)})});\n"
        "    const d=await r.json();t.remove();\n"
        "    addMsg(\'assistant\',d.answer);H.push({role:\'assistant\',content:d.answer});\n"
        "  }catch(e){t.remove();addMsg(\'assistant\',\'Error: \'+e.message);}\n"
        "}\n"
        "</script></body></html>"
    )



def _gen_readme(b: RAGBot) -> str:
    return f"""# {b.name}

> Generated by **Model Gen X** — https://modelgenx.site

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get free Groq API key
→ https://console.groq.com (takes 1 minute, free)

### 3. Run
```bash
export GROQ_API_KEY=your_key_here
python server.py
```

Open **http://localhost:8000** in your browser.

## Deploy to production

### Railway (recommended)
```bash
railway login && railway init && railway up
```
Add `GROQ_API_KEY` in Railway environment variables.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV GROQ_API_KEY=""
CMD ["python", "server.py"]
```

### Any VPS
```bash
pip install -r requirements.txt
GROQ_API_KEY=xxx python server.py
# Use nginx + systemd for production
```

## Configuration

Edit `config.json` to change bot behavior without touching code.

| Setting | Value | Description |
|---|---|---|
| language | `{b.language}` | Response language |
| top_k | `{b.top_k}` | Chunks retrieved per query |
| temperature | `{b.temperature}` | Response creativity |
| max_tokens | `{b.max_tokens}` | Max response length |
| show_sources | `{b.show_sources}` | Show source documents |
| allow_general | `{b.allow_general}` | Allow answers outside docs |

## Stats
- Documents: {b.n_documents}
- Knowledge chunks: {b.n_chunks}
- Knowledge base size: {b.total_size_mb} MB

---
*Built with Model Gen X — The AutoML & RAG Platform*
"""