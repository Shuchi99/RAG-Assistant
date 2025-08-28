# ingestion.py
import os, io, pickle, faiss, numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from bs4 import BeautifulSoup

VECTOR_STORE_PATH = "vector_store.pkl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".html") or name.endswith(".htm"):
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(" ", strip=True)
    # default: txt/md
    return content.decode("utf-8", errors="ignore")

def _load_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return None, []
    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["docs"]

def ingest_document(filename, content):
    text = _extract_text(filename, content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    new_docs = [Document(page_content=c, metadata={"source": filename}) for c in chunks]

    vectors = embeddings.embed_documents([d.page_content for d in new_docs])
    X = np.array(vectors, dtype="float32")  # FAISS requires float32

    index, docs = _load_store()
    if index is None:
        index = faiss.IndexFlatL2(X.shape[1])
        docs = []
    index.add(X)
    docs.extend(new_docs)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
