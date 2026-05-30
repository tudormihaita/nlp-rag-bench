RAG_PROMPT = """\
Answer the question based ONLY on the context below. \
If the answer cannot be determined from the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

NO_RAG_PROMPT = """\
Answer the question concisely. If you don't know, say "I don't know."

Question: {question}

Answer:"""

DECOMPOSITION_PROMPT = """\
Decompose the following multi-hop question into 2-4 simple sub-questions \
that can be answered one at a time. Output ONLY the sub-questions, one per line, numbered.

Question: {question}

Sub-questions:"""