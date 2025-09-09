from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ingestion import ingest_document, ensure_embeddings_ready
from retrieval import query_knowledge_base, LLAMA_MODEL
import torch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook:
    - Runs once at startup (great place to warm up models, caches, etc.)
    - Runs once at shutdown (optional cleanup)
    """
    # --- Startup ---
    print("[RAG] Warming up embeddings…")
    ensure_embeddings_ready()  # Download/load the embedding model once and run a tiny warmup call
    print("[RAG] Embeddings ready.")
    yield
    # --- Shutdown ---
    print("[RAG] Server shutting down.")

# Public FastAPI app with a startup warmup
app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)

# Allow browser apps (Vite/React) to call this API from other origins during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict this to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Upload endpoint:
    - Accepts a single file
    - Kicks off ingestion in the background so the request returns quickly
    """
    content = await file.read()
    # Background task will parse, chunk, embed, and append to FAISS index
    background_tasks.add_task(ingest_document, file.filename, content)
    return {"message": f"{file.filename} received. Indexing in background."}

@app.post("/query/")
async def query_assistant(query: str = Form(...)):
    """
    Query endpoint:
    - Receives a natural language question
    - Runs retrieval-augmented generation over uploaded documents
    - Returns the answer text plus a list of source filenames
    """
    answer, sources = query_knowledge_base(query)
    return {"answer": answer, "sources": sources}

@app.get("/")
async def root():
    """Simple healthcheck."""
    return {"message": "RAG Knowledge Assistant API is running!"}
