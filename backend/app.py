# app.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ingestion import ingest_document
from retrieval import query_knowledge_base

app = FastAPI(title="RAG Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_document(file: UploadFile):
    content = await file.read()
    ingest_document(file.filename, content)
    return {"message": f"{file.filename} uploaded and processed successfully."}

@app.post("/query/")
async def query_assistant(query: str = Form(...)):
    answer, sources = query_knowledge_base(query)
    return {"answer": answer, "sources": sources}

@app.get("/")
async def root():
    return {"message": "RAG Knowledge Assistant API is running!"}
