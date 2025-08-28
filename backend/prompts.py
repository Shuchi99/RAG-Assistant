RAG_PROMPT = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If you don't know, say "I don't have enough information."

CONTEXT:
{context}

QUESTION:
{query}

Answer:
"""
