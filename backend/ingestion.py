# ingestion.py
import os, io, time, pickle, faiss, numpy as np, torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from bs4 import BeautifulSoup

# Path to the serialized FAISS index and accompanying docs list
VECTOR_STORE_PATH = "vector_store.pkl"

# Module-level cache for the embedding model so it loads only once
_embeddings = None

def get_embeddings():
    """
    Lazily load and cache the embedding model.
    Uses GPU if available and EMBED_ON_GPU=1 (default) and CUDA is present.
    """
    global _embeddings
    if _embeddings is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        device = "cuda" if (os.getenv("EMBED_ON_GPU", "1") == "1" and torch.cuda.is_available()) else "cpu"
        print(f"[INGEST] Loading embeddings {model_name} on {device} …")
        _embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return _embeddings

def ensure_embeddings_ready():
    """
    Run a tiny 'warmup' embedding to pay one-time model init cost at startup.
    This makes the first real request feel snappier.
    """
    t0 = time.time()
    get_embeddings().embed_query("warmup")
    print(f"[INGEST] Warmup done in {time.time()-t0:.1f}s")

def _extract_text(filename: str, content: bytes) -> str:
    """
    Convert bytes to plain text depending on file type.
    - PDF: text extraction via pypdf
    - HTML: strip tags & keep visible text with BeautifulSoup
    - Plain text / markdown: decode as UTF-8 (ignore bad bytes)
    """
    n = filename.lower()
    if n.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        # Concatenate text from all pages (skip None safely)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if n.endswith(".html") or n.endswith(".htm"):
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(" ", strip=True)
    # Default: try to decode as UTF-8 text
    return content.decode("utf-8", errors="ignore")

def ingest_document(filename, content):
    """
    Ingestion pipeline:
    1) Extract raw text from the uploaded file
    2) Split text into overlapping chunks (for better recall & context)
    3) Embed chunks with the selected embedding model
    4) Append vectors to FAISS index and persist {index, docs} to disk
    """
    print(f"[INGEST] Start: {filename}")
    t0 = time.time()

    # --- 1) Extract text ---
    text = _extract_text(filename, content)

    # --- 2) Chunking strategy ---
    # 500 token-ish chunks with 100 overlap is a good general baseline
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text(text)
    # Wrap into LangChain Document objects so we can keep metadata (source filename)
    docs = [Document(page_content=c, metadata={"source": filename}) for c in chunks]
    print(f"[INGEST] {len(chunks)} chunks")

    # --- 3) Embeddings ---
    t1 = time.time()
    vectors = get_embeddings().embed_documents([d.page_content for d in docs])
    print(f"[INGEST] Embedded {len(vectors)} chunks in {time.time()-t1:.1f}s")

    # Convert to float32 numpy array for FAISS
    X = np.array(vectors, dtype="float32")

    # --- 4) FAISS index append / create ---
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing index & docs and append
        with open(VECTOR_STORE_PATH, "rb") as f:
            data = pickle.load(f)
        index, all_docs = data["index"], data["docs"]
    else:
        # No prior index: build a new flat L2 index
        index, all_docs = faiss.IndexFlatL2(X.shape[1]), []

    # Add new vectors and extend the documents list
    index.add(X)
    all_docs.extend(docs)

    # Persist to disk for future queries
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": all_docs}, f)

    print(f"[INGEST] Done: {filename} | index size={index.ntotal} | {time.time()-t0:.1f}s total")
