import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from prompts import RAG_PROMPT
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, AutoConfig)
from langchain.llms import HuggingFacePipeline
import torch

load_dotenv()

VECTOR_STORE_PATH = "vector_store.pkl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Choose device for embeddings (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device})

# Default generator model (small+fast by default). Can be changed via .env
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # used for gated models on Hugging Face

def load_llm():
    """
    Build a text-generation pipeline that works for both:
      - Encoder-decoder (seq2seq) models (e.g., FLAN-T5)
      - Decoder-only (causal) models (e.g., LLaMA, Mistral)
    We keep generation deterministic (temperature=0) to make outputs reproducible.
    """
    model_id = LLAMA_MODEL
    has_cuda = torch.cuda.is_available()

    # Inspect config to detect if the model is encoder-decoder (seq2seq) or causal
    cfg = AutoConfig.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

    if is_seq2seq:
        # --- Seq2Seq path (e.g., FLAN-T5) ---
        dtype = torch.float16 if has_cuda else torch.float32
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, use_auth_token=HF_TOKEN, torch_dtype=dtype
        )
        if has_cuda:
            mdl.to("cuda")

        gen = pipeline(
            "text2text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=32,  # keep concise; we already constrain via prompt
            temperature=0.0,
            do_sample=False,
            device=0 if has_cuda else -1,
        )
    else:
        # --- Causal path (decoder-only models) ---
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=HF_TOKEN,
            device_map="auto" if has_cuda else None,  # let HF place layers across GPUs if available
            torch_dtype="auto",
        )
        gen = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=64,   # causal models sometimes need a bit more
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.05,  # mild anti-repetition
            return_full_text=False,    # only return the generated completion
        )

    # LangChain wrapper so we can call llm.invoke(prompt)
    return HuggingFacePipeline(pipeline=gen)

# Initialize the local LLM once at import time
llm = load_llm()

def query_knowledge_base(query: str, k: int = 3):
    """
    Core RAG routine:
      1) Load FAISS index and stored docs (created by ingestion.py)
      2) Embed the incoming query
      3) Search top-k similar chunks
      4) Build the RAG prompt with those chunks as 'Context'
      5) Generate an answer with the local LLM
      6) Return the answer + unique list of source filenames

    Returns:
      (answer_text: str, sources: list[str])
    """
    # --- Load FAISS & docs ---
    if not os.path.exists(VECTOR_STORE_PATH):
        return "I don't have enough information. Please upload documents first.", []

    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
        index = data["index"]
        docs = data["docs"]

    # Handle empty or missing index
    if index is None or getattr(index, "ntotal", 0) == 0:
        return "I don't have enough information. Please upload documents first.", []

    # --- Embed the query (shape [1, d], dtype float32) ---
    query_vector = embeddings.embed_query(query)
    qv = np.array([query_vector], dtype="float32")

    # Clamp k to avoid asking FAISS for more neighbors than it has
    k = max(1, min(int(k), index.ntotal))

    # --- Nearest-neighbor search ---
    D, I = index.search(qv, k)  # D=distances, I=indices

    # Collect the top-k doc chunks; guard against out-of-range indices
    retrieved_docs = [docs[i] for i in I[0] if 0 <= i < len(docs)]

    # Build textual context by concatenating chunk contents
    context = "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""
    # Hard truncate context to keep prompt size reasonable for small models
    context = context[:3000]

    if not context.strip():
        return "I don't have enough information from the uploaded documents to answer that.", []

    # --- Prompt construction ---
    prompt = RAG_PROMPT.format(context=context, query=query)

    # --- Run generation ---
    try:
        response_text = llm.invoke(prompt).strip()
        # LangChain's HuggingFacePipeline returns a string
        answer = response_text if isinstance(response_text, str) else str(response_text)
    except Exception as e:
        # Surface generation errors without crashing the server
        answer = f"⚠️ Generation error: {e}"

    # --- Collect unique sources (preserve order) ---
    seen = set()
    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)

    return answer, sources
