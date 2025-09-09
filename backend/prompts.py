RAG_PROMPT = """Answer the user's question using ONLY the Context.
If the answer is present in the Context, COPY the minimal text span that answers it (no extra words).
If the answer is not present, reply exactly: I don't have enough information.

CONTEXT:
{context}

QUESTION:
{query}

Answer:
"""
