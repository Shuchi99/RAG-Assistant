import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from prompts import RAG_PROMPT
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Configurations
# ----------------------------
VECTOR_STORE_PATH = "vector_store.pkl"
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-chat-hf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ----------------------------
# Embeddings (local / free)
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ----------------------------
# Load LLaMA-2 Model (Local)
# ----------------------------
def load_llm():
    print(f"[RAG] Loading open-source model: {LLAMA_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        use_auth_token=HF_TOKEN,
        device_map="auto",    # GPU if available
        torch_dtype="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

# Initialize the local model once
llm = load_llm()

# ----------------------------
# Query Knowledge Base
# ----------------------------
def query_knowledge_base(query: str, k: int = 3):
    """
    Searches FAISS index for relevant documents, builds a RAG prompt,
    and gets an answer from the local LLM.
    Returns: (answer_text: str, sources: list[str])
    """
    # Load FAISS store & docs
    if not os.path.exists(VECTOR_STORE_PATH):
        return "I don't have enough information. Please upload documents first.", []

    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
        index = data["index"]
        docs = data["docs"]

    # Handle empty index
    if index is None or getattr(index, "ntotal", 0) == 0:
        return "I don't have enough information. Please upload documents first.", []

    # Embed the query (ensure shape (1, d) and float32 dtype)
    query_vector = embeddings.embed_query(query)
    qv = np.array([query_vector], dtype="float32")

    # Clamp k to available vectors
    k = max(1, min(int(k), index.ntotal))

    # Semantic search
    D, I = index.search(qv, k)

    # Retrieve top-k chunks
    retrieved_docs = [docs[i] for i in I[0] if 0 <= i < len(docs)]
    context = "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""

    if not context.strip():
        return "I don't have enough information from the uploaded documents to answer that.", []

    # Format RAG prompt
    prompt = RAG_PROMPT.format(context=context, query=query)

    # Generate answer locally using the LLM
    try:
        response_text = llm.invoke(prompt)
        # langchain's HuggingFacePipeline returns a str
        answer = response_text if isinstance(response_text, str) else str(response_text)
    except Exception as e:
        answer = f"⚠️ Generation error: {e}"

    # Collect source filenames (deduped, preserve order)
    seen = set()
    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)

    return answer, sources
